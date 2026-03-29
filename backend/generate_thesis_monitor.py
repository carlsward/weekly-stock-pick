import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from backend.generate_news_scores import init_models, score_symbol_news
    from backend.generate_pick import (
        HISTORY_PATH,
        MIN_CONFIDENCE_THRESHOLD,
        MODEL_VERSION,
        OVERALL_SELECTION_THRESHOLD,
        RISK_PICKS_PATH,
        SCHEMA_VERSION,
        StockCandidate,
        build_candidate,
        build_model_calibration,
        build_market_week,
        derive_data_as_of,
        deserialize_model_calibration,
        is_live_price_provider_failure,
        iso_utc,
        load_existing_history_entries,
        load_universe,
        now_utc,
        normalize_history_entry,
        resolve_universe_csv_path,
        serialize_candidate,
        write_json,
    )
    from backend.sector_utils import load_sector_map
    from backend.generate_sector_scores import build_sector_scores_payload, fetch_recent_global_articles
    from backend.pipeline_runtime import (
        build_data_quality_block,
        extract_payload_degraded_reasons,
        set_pipeline_scope,
    )
except ImportError:
    from generate_news_scores import init_models, score_symbol_news
    from generate_pick import (
        HISTORY_PATH,
        MIN_CONFIDENCE_THRESHOLD,
        MODEL_VERSION,
        OVERALL_SELECTION_THRESHOLD,
        RISK_PICKS_PATH,
        SCHEMA_VERSION,
        StockCandidate,
        build_candidate,
        build_model_calibration,
        build_market_week,
        derive_data_as_of,
        deserialize_model_calibration,
        is_live_price_provider_failure,
        iso_utc,
        load_existing_history_entries,
        load_universe,
        now_utc,
        normalize_history_entry,
        resolve_universe_csv_path,
        serialize_candidate,
        write_json,
    )
    from sector_utils import load_sector_map
    from generate_sector_scores import build_sector_scores_payload, fetch_recent_global_articles
    from pipeline_runtime import (
        build_data_quality_block,
        extract_payload_degraded_reasons,
        set_pipeline_scope,
    )


THESIS_MONITOR_PATH = Path("thesis_monitor.json")
THESIS_MONITOR_REFRESH_HOURS = 24
THESIS_MONITOR_STALE_HOURS = 36


def load_dashboard_source(path: Path = RISK_PICKS_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("risk_picks.json must contain an object")
    return payload


def monitor_refresh_window() -> tuple[str, str]:
    generated_at = now_utc()
    expected_next_refresh = generated_at + timedelta(hours=THESIS_MONITOR_REFRESH_HOURS)
    stale_after = generated_at + timedelta(hours=THESIS_MONITOR_STALE_HOURS)
    return iso_utc(expected_next_refresh), iso_utc(stale_after)


def source_threshold(selection: Dict[str, Any]) -> float:
    raw = selection.get("threshold_score")
    return float(raw) if isinstance(raw, (int, float)) else OVERALL_SELECTION_THRESHOLD


def source_confidence_threshold(selection: Dict[str, Any]) -> float:
    raw = selection.get("threshold_confidence")
    return float(raw) if isinstance(raw, (int, float)) else MIN_CONFIDENCE_THRESHOLD


def build_no_active_pick_payload(
    source_payload: Dict[str, Any],
    generated_at: str,
    expected_next_refresh_at: str,
    stale_after: str,
) -> Dict[str, Any]:
    market_context = source_payload.get("market_context")
    source_selection = source_payload.get("overall_selection", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at,
        "data_as_of": source_payload.get("data_as_of"),
        "expected_next_refresh_at": expected_next_refresh_at,
        "stale_after": stale_after,
        "market_context": market_context,
        "source_dashboard_generated_at": source_payload.get("generated_at"),
        "selection": {
            "status": "no_pick",
            "status_reason": str(source_selection.get("status_reason", "")).strip()
            or "No active weekly pick is currently available to monitor.",
            "threshold_score": round(source_threshold(source_selection), 2),
            "threshold_confidence": round(source_confidence_threshold(source_selection), 2),
        },
        "active_pick": None,
    }


def build_carried_forward_pick_payload(
    source_payload: Dict[str, Any],
    source_selection: Dict[str, Any],
    active_source_pick: Dict[str, Any],
    generated_at: str,
    expected_next_refresh_at: str,
    stale_after: str,
    status_reason: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at,
        "data_as_of": source_payload.get("data_as_of"),
        "expected_next_refresh_at": expected_next_refresh_at,
        "stale_after": stale_after,
        "market_context": source_payload.get("market_context"),
        "source_dashboard_generated_at": source_payload.get("generated_at"),
        "selection": {
            "status": "picked",
            "status_reason": status_reason,
            "threshold_score": round(source_threshold(source_selection), 2),
            "threshold_confidence": round(source_confidence_threshold(source_selection), 2),
        },
        "active_pick": dict(active_source_pick),
    }


def source_pick(source_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selection = source_payload.get("overall_selection")
    if not isinstance(selection, dict):
        return None
    if str(selection.get("status", "")).strip() != "picked":
        return None
    pick = selection.get("pick")
    return pick if isinstance(pick, dict) else None


def build_live_calibration(source_payload: Optional[Dict[str, Any]] = None):
    generation_summary = source_payload.get("generation_summary") if isinstance(source_payload, dict) else None
    if isinstance(generation_summary, dict):
        cached = deserialize_model_calibration(generation_summary.get("model_calibration"))
        if cached is not None:
            return cached

    universe_path = resolve_universe_csv_path()
    universe = load_universe(universe_path)
    sector_map = load_sector_map(universe_path)
    history_entries = [
        normalize_history_entry(entry)
        for entry in load_existing_history_entries(HISTORY_PATH)
    ]
    return build_model_calibration(universe, sector_map, history_entries)


def resolve_source_pick_sector(source_pick_payload: Dict[str, Any]) -> str:
    raw_sector = source_pick_payload.get("sector")
    if isinstance(raw_sector, str) and raw_sector.strip():
        return raw_sector.strip()

    symbol = str(source_pick_payload.get("symbol", "")).strip().upper()
    if not symbol:
        raise RuntimeError("Active pick is missing both sector and symbol.")

    universe_path = resolve_universe_csv_path()
    sector_map = load_sector_map(universe_path)
    resolved = sector_map.get(symbol)
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()

    raise RuntimeError(f"Unable to resolve sector for active pick {symbol}.")


def build_live_candidate(source_pick_payload: Dict[str, Any], source_payload: Optional[Dict[str, Any]] = None) -> StockCandidate:
    symbol = str(source_pick_payload["symbol"]).strip().upper()
    company_name = str(source_pick_payload["company_name"]).strip()
    sector = resolve_source_pick_sector(source_pick_payload)

    _, summarizer_pipe = init_models()
    news_info = score_symbol_news(
        symbol=symbol,
        company_name=company_name,
        summarizer_pipe=summarizer_pipe,
        llm_enabled=None,
        model_name=None,
    )

    symbol_metadata = {
        symbol: {
            "company_name": company_name,
            "sector": sector,
        }
    }
    sector_payload = build_sector_scores_payload(
        articles=fetch_recent_global_articles(),
        symbol_metadata=symbol_metadata,
        sectors=[sector],
        generated_at=now_utc(),
    )
    calibration = build_live_calibration(source_payload)

    return build_candidate(
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        news_info=news_info,
        sector_scores_payload=sector_payload,
        calibration=calibration,
    )


def main() -> None:
    set_pipeline_scope("thesis_monitor")
    source_payload = load_dashboard_source()
    generated_at = iso_utc(now_utc())
    expected_next_refresh_at, stale_after = monitor_refresh_window()
    source_selection = source_payload.get("overall_selection", {})
    active_source_pick = source_pick(source_payload)
    upstream_reasons = extract_payload_degraded_reasons(source_payload, "risk_picks")

    if active_source_pick is None:
        payload = build_no_active_pick_payload(
            source_payload=source_payload,
            generated_at=generated_at,
            expected_next_refresh_at=expected_next_refresh_at,
            stale_after=stale_after,
        )
        payload["data_quality"] = build_data_quality_block(
            scope="thesis_monitor",
            extra_reasons=upstream_reasons,
        )
        write_json(THESIS_MONITOR_PATH, payload)
        print("[INFO] thesis_monitor.json updated without an active pick.")
        return

    try:
        candidate = build_live_candidate(active_source_pick, source_payload)
    except Exception as exc:
        refresh_error = str(exc).strip() or "unknown refresh error"
        if is_live_price_provider_failure(refresh_error):
            reason = (
                "Live thesis monitor carried forward the last trusted weekly dashboard because "
                "live price providers failed for the active pick."
            )
        else:
            reason = (
                "Live thesis monitor carried forward the last trusted weekly dashboard because "
                "the refresh step failed."
            )

        payload = build_carried_forward_pick_payload(
            source_payload=source_payload,
            source_selection=source_selection,
            active_source_pick=active_source_pick,
            generated_at=generated_at,
            expected_next_refresh_at=expected_next_refresh_at,
            stale_after=stale_after,
            status_reason=reason,
        )
        payload["data_quality"] = build_data_quality_block(
            scope="thesis_monitor",
            extra_reasons=[*upstream_reasons, f"thesis_monitor: {refresh_error}"],
        )
        write_json(THESIS_MONITOR_PATH, payload)
        print("[WARN] thesis_monitor.json carried forward the last trusted weekly dashboard.")
        return

    data_as_of = derive_data_as_of([candidate])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at,
        "data_as_of": data_as_of,
        "expected_next_refresh_at": expected_next_refresh_at,
        "stale_after": stale_after,
        "market_context": source_payload.get("market_context"),
        "source_dashboard_generated_at": source_payload.get("generated_at"),
        "selection": {
            "status": "picked",
            "status_reason": "Live thesis monitor refreshed the active weekly pick.",
            "threshold_score": round(source_threshold(source_selection), 2),
            "threshold_confidence": round(source_confidence_threshold(source_selection), 2),
        },
        "active_pick": serialize_candidate(
            candidate,
            threshold_score=source_threshold(source_selection),
            threshold_confidence=source_confidence_threshold(source_selection),
        ),
    }
    payload["data_quality"] = build_data_quality_block(
        scope="thesis_monitor",
        extra_reasons=upstream_reasons,
    )
    write_json(THESIS_MONITOR_PATH, payload)
    print("[INFO] thesis_monitor.json updated.")


if __name__ == "__main__":
    main()
