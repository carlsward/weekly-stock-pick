from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

try:
    from backend.generate_pick import (
        HISTORY_PATH,
        MARKET_BENCHMARK_SYMBOL,
        MIN_CONFIDENCE_THRESHOLD,
        MODEL_VERSION,
        OVERALL_SELECTION_THRESHOLD,
        RISK_SELECTION_THRESHOLDS,
        SCHEMA_VERSION,
        SECTOR_ETF_BY_SECTOR,
        build_market_week,
        build_refresh_window,
        iso_utc,
        load_existing_history_entries,
        normalize_history_entry,
        now_utc,
        realized_forward_return,
        resolve_universe_csv_path,
        write_json,
    )
    from backend.pipeline_runtime import (
        build_data_quality_block,
        extract_payload_degraded_reasons,
        set_pipeline_scope,
    )
    from backend.sector_utils import load_sector_map
except ImportError:
    from generate_pick import (
        HISTORY_PATH,
        MARKET_BENCHMARK_SYMBOL,
        MIN_CONFIDENCE_THRESHOLD,
        MODEL_VERSION,
        OVERALL_SELECTION_THRESHOLD,
        RISK_SELECTION_THRESHOLDS,
        SCHEMA_VERSION,
        SECTOR_ETF_BY_SECTOR,
        build_market_week,
        build_refresh_window,
        iso_utc,
        load_existing_history_entries,
        normalize_history_entry,
        now_utc,
        realized_forward_return,
        resolve_universe_csv_path,
        write_json,
    )
    from pipeline_runtime import (
        build_data_quality_block,
        extract_payload_degraded_reasons,
        set_pipeline_scope,
    )
    from sector_utils import load_sector_map


TRACK_RECORD_PATH = Path("track_record.json")


def round_optional(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def safe_average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def safe_compounded_return(values: List[float]) -> Optional[float]:
    if not values:
        return None
    compounded = 1.0
    for value in values:
        compounded *= (1.0 + value)
    return compounded - 1.0


def safe_correlation(left: List[float], right: List[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    left_variance = sum((x - mean_left) ** 2 for x in left)
    right_variance = sum((y - mean_right) ** 2 for y in right)
    if left_variance <= 1e-12 or right_variance <= 1e-12:
        return None
    return numerator / ((left_variance * right_variance) ** 0.5)


def outcome_label(realized_return: Optional[float]) -> str:
    if realized_return is None:
        return "open"
    if realized_return > 0:
        return "win"
    if realized_return < 0:
        return "loss"
    return "flat"


def enrich_entry_metrics(
    entry: Dict[str, Any],
    sector_map: Dict[str, str],
) -> Dict[str, Any]:
    enriched = dict(entry)
    symbol = enriched.get("symbol")
    week_end = enriched.get("week_end")
    if not isinstance(symbol, str) or not symbol.strip():
        enriched["sector"] = None
        enriched["realized_5d_sector_excess_return"] = None
        enriched["realized_5d_sector_return"] = None
        enriched["outcome"] = "no_pick"
        return enriched

    stored_sector = enriched.get("sector")
    sector = (
        str(stored_sector).strip()
        if isinstance(stored_sector, str) and str(stored_sector).strip()
        else sector_map.get(symbol.upper())
    )
    enriched["sector"] = sector

    realized_return = enriched.get("realized_5d_return")
    realized_excess = enriched.get("realized_5d_excess_return")
    if not isinstance(realized_return, (int, float)) or not isinstance(realized_excess, (int, float)):
        fetched_return, fetched_excess = realized_forward_return(symbol, week_end, MARKET_BENCHMARK_SYMBOL)
        if fetched_return is not None:
            realized_return = fetched_return
            enriched["realized_5d_return"] = round_optional(fetched_return)
        if fetched_excess is not None:
            realized_excess = fetched_excess
            enriched["realized_5d_excess_return"] = round_optional(fetched_excess)

    sector_benchmark_symbol = SECTOR_ETF_BY_SECTOR.get(sector or "")
    sector_return: Optional[float] = None
    sector_excess: Optional[float] = None
    if sector_benchmark_symbol and isinstance(week_end, str):
        fetched_sector_return, fetched_sector_excess = realized_forward_return(
            symbol,
            week_end,
            sector_benchmark_symbol,
        )
        sector_return = fetched_sector_return
        sector_excess = fetched_sector_excess

    enriched["realized_5d_sector_return"] = round_optional(sector_return)
    enriched["realized_5d_sector_excess_return"] = round_optional(sector_excess)
    enriched["outcome"] = outcome_label(realized_return if isinstance(realized_return, (int, float)) else None)
    return enriched


def build_summary_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    picked_entries = [entry for entry in entries if entry.get("status") == "picked" and entry.get("symbol")]
    closed_entries = [
        entry
        for entry in picked_entries
        if isinstance(entry.get("realized_5d_return"), (int, float))
    ]
    realized_returns = [float(entry["realized_5d_return"]) for entry in closed_entries]
    spy_excess_returns = [
        float(entry["realized_5d_excess_return"])
        for entry in closed_entries
        if isinstance(entry.get("realized_5d_excess_return"), (int, float))
    ]
    sector_excess_returns = [
        float(entry["realized_5d_sector_excess_return"])
        for entry in closed_entries
        if isinstance(entry.get("realized_5d_sector_excess_return"), (int, float))
    ]

    return {
        "total_weeks": len(entries),
        "total_picks": len(picked_entries),
        "no_pick_weeks": len([entry for entry in entries if entry.get("status") == "no_pick"]),
        "closed_picks": len(closed_entries),
        "open_picks": len(picked_entries) - len(closed_entries),
        "win_rate": round_optional(
            len([value for value in realized_returns if value > 0]) / len(realized_returns)
            if realized_returns
            else None,
            4,
        ),
        "beat_spy_rate": round_optional(
            len([value for value in spy_excess_returns if value > 0]) / len(spy_excess_returns)
            if spy_excess_returns
            else None,
            4,
        ),
        "beat_sector_rate": round_optional(
            len([value for value in sector_excess_returns if value > 0]) / len(sector_excess_returns)
            if sector_excess_returns
            else None,
            4,
        ),
        "average_5d_return": round_optional(safe_average(realized_returns)),
        "median_5d_return": round_optional(median(realized_returns) if realized_returns else None),
        "average_5d_excess_return": round_optional(safe_average(spy_excess_returns)),
        "average_5d_sector_excess_return": round_optional(safe_average(sector_excess_returns)),
        "compounded_5d_return": round_optional(safe_compounded_return(realized_returns)),
    }


def build_risk_breakdown(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    breakdown: Dict[str, Dict[str, Any]] = {}
    for risk in ("low", "medium", "high"):
        risk_entries = [
            entry for entry in entries
            if entry.get("status") == "picked" and str(entry.get("risk", "")).lower() == risk
        ]
        closed_entries = [
            entry for entry in risk_entries
            if isinstance(entry.get("realized_5d_return"), (int, float))
        ]
        realized_returns = [float(entry["realized_5d_return"]) for entry in closed_entries]
        excess_returns = [
            float(entry["realized_5d_excess_return"])
            for entry in closed_entries
            if isinstance(entry.get("realized_5d_excess_return"), (int, float))
        ]
        breakdown[risk] = {
            "pick_count": len(risk_entries),
            "closed_pick_count": len(closed_entries),
            "win_rate": round_optional(
                len([value for value in realized_returns if value > 0]) / len(realized_returns)
                if realized_returns
                else None,
                4,
            ),
            "average_5d_return": round_optional(safe_average(realized_returns)),
            "average_5d_excess_return": round_optional(safe_average(excess_returns)),
        }
    return breakdown


def build_signal_block_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    samples: List[Dict[str, float]] = []
    for entry in entries:
        diagnostics = entry.get("diagnostics")
        realized = entry.get("realized_5d_excess_return")
        if not isinstance(diagnostics, dict) or not isinstance(realized, (int, float)):
            continue

        technical_total = diagnostics.get("technical_total")
        news_adjustment = diagnostics.get("news_adjustment")
        macro_adjustment = diagnostics.get("macro_adjustment")
        sector_adjustment = diagnostics.get("sector_adjustment")
        if not all(
            isinstance(value, (int, float))
            for value in (technical_total, news_adjustment, macro_adjustment, sector_adjustment)
        ):
            continue

        samples.append(
            {
                "technical_total": float(technical_total),
                "full_model_score": float(technical_total + news_adjustment + macro_adjustment + sector_adjustment),
                "block_adjustment": float(news_adjustment + macro_adjustment + sector_adjustment),
                "news_signal_input": float(diagnostics.get("news_signal_input", 0.0)),
                "macro_signal_input": float(diagnostics.get("macro_signal_input", 0.0)),
                "sector_signal_input": float(diagnostics.get("sector_signal_input", 0.0)),
                "realized_excess": float(realized),
            }
        )

    if len(samples) < 6:
        return {
            "status": "insufficient_data",
            "scope": "picked_entries_only",
            "sample_count": len(samples),
            "summary": "Not enough closed weekly picks are available yet to compare technical-only conviction against block-adjusted conviction.",
        }

    realized = [sample["realized_excess"] for sample in samples]
    technical_scores = [sample["technical_total"] for sample in samples]
    full_scores = [sample["full_model_score"] for sample in samples]
    block_adjustments = [sample["block_adjustment"] for sample in samples]

    technical_ic = safe_correlation(technical_scores, realized)
    full_ic = safe_correlation(full_scores, realized)
    block_ic = safe_correlation(block_adjustments, realized)
    improvement = None
    if technical_ic is not None and full_ic is not None:
        improvement = full_ic - technical_ic

    if improvement is None:
        summary = "Closed picks now have enough samples for monitoring, but the score dispersion is still too narrow for a stable correlation read."
    elif improvement > 0.03:
        summary = "On closed picked weeks, news/macro/sector adjustments are currently improving the realized excess-return fit versus technical-only conviction."
    elif improvement < -0.03:
        summary = "On closed picked weeks, block adjustments are currently hurting the realized excess-return fit versus technical-only conviction."
    else:
        summary = "On closed picked weeks, block adjustments are currently close to neutral versus technical-only conviction."

    return {
        "status": "ok",
        "scope": "picked_entries_only",
        "sample_count": len(samples),
        "technical_only_ic": round_optional(technical_ic),
        "full_model_ic": round_optional(full_ic),
        "block_adjustment_ic": round_optional(block_ic),
        "ic_improvement_vs_technical": round_optional(improvement),
        "average_block_adjustment": round_optional(safe_average(block_adjustments)),
        "average_news_signal_input": round_optional(safe_average([sample["news_signal_input"] for sample in samples])),
        "average_macro_signal_input": round_optional(safe_average([sample["macro_signal_input"] for sample in samples])),
        "average_sector_signal_input": round_optional(safe_average([sample["sector_signal_input"] for sample in samples])),
        "summary": summary,
    }


def build_candidate_ranking_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    for entry in entries:
        week_end = entry.get("week_end")
        top_candidates = entry.get("top_candidates")
        if not isinstance(week_end, str) or not isinstance(top_candidates, list):
            continue
        for candidate in top_candidates:
            if not isinstance(candidate, dict):
                continue
            symbol = str(candidate.get("symbol", "")).strip().upper()
            rank = candidate.get("rank")
            if not symbol or not isinstance(rank, int):
                continue
            realized_return, realized_excess = realized_forward_return(
                symbol,
                week_end,
                MARKET_BENCHMARK_SYMBOL,
            )
            if not isinstance(realized_return, (int, float)) or not isinstance(realized_excess, (int, float)):
                continue
            samples.append(
                {
                    "week_id": entry.get("week_id"),
                    "week_end": week_end,
                    "rank": rank,
                    "symbol": symbol,
                    "model_score": candidate.get("model_score"),
                    "realized_5d_return": realized_return,
                    "realized_5d_excess_return": realized_excess,
                }
            )

    if len(samples) < 10:
        return {
            "status": "insufficient_data",
            "sample_count": len(samples),
            "summary": "Not enough closed stored candidate rankings are available yet to evaluate rank quality.",
        }

    rank_buckets: Dict[str, Dict[str, Any]] = {}
    for rank in range(1, 11):
        rank_values = [
            float(sample["realized_5d_excess_return"])
            for sample in samples
            if sample["rank"] == rank
        ]
        if not rank_values:
            continue
        rank_buckets[str(rank)] = {
            "sample_count": len(rank_values),
            "average_5d_excess_return": round_optional(safe_average(rank_values)),
            "beat_spy_rate": round_optional(
                len([value for value in rank_values if value > 0]) / len(rank_values),
                4,
            ),
        }

    top_3_values = [
        float(sample["realized_5d_excess_return"])
        for sample in samples
        if int(sample["rank"]) <= 3
    ]
    top_10_values = [float(sample["realized_5d_excess_return"]) for sample in samples]

    return {
        "status": "ok",
        "scope": "stored_weekly_top_candidates",
        "sample_count": len(samples),
        "rank_buckets": rank_buckets,
        "top_3_average_5d_excess_return": round_optional(safe_average(top_3_values)),
        "top_3_beat_spy_rate": round_optional(
            len([value for value in top_3_values if value > 0]) / len(top_3_values)
            if top_3_values
            else None,
            4,
        ),
        "top_10_average_5d_excess_return": round_optional(safe_average(top_10_values)),
        "summary": "Stored weekly top-candidate rankings now have enough closed samples to monitor rank quality.",
    }


def build_no_pick_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    spy_returns: List[float] = []
    for entry in entries:
        if entry.get("status") != "no_pick":
            continue
        week_end = entry.get("week_end")
        if not isinstance(week_end, str):
            continue
        spy_return, _ = realized_forward_return(MARKET_BENCHMARK_SYMBOL, week_end)
        if isinstance(spy_return, (int, float)):
            spy_returns.append(float(spy_return))

    if not spy_returns:
        return {
            "status": "insufficient_data",
            "sample_count": 0,
            "summary": "No closed no-pick weeks are available yet for cash-versus-SPY comparison.",
        }

    return {
        "status": "ok",
        "sample_count": len(spy_returns),
        "average_spy_5d_return_during_no_pick": round_optional(safe_average(spy_returns)),
        "spy_up_rate_during_no_pick": round_optional(
            len([value for value in spy_returns if value > 0]) / len(spy_returns),
            4,
        ),
        "avoided_loss_rate": round_optional(
            len([value for value in spy_returns if value < 0]) / len(spy_returns),
            4,
        ),
        "summary": "No-pick weeks are compared against holding SPY over the same forward 5-trading-day window.",
    }


def serialize_track_record_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "week_id": entry.get("week_id"),
        "week_start": entry.get("week_start"),
        "week_end": entry.get("week_end"),
        "week_label": entry.get("week_label"),
        "logged_at": entry.get("logged_at"),
        "status": entry.get("status"),
        "status_reason": entry.get("status_reason"),
        "symbol": entry.get("symbol"),
        "company_name": entry.get("company_name"),
        "sector": entry.get("sector"),
        "risk": entry.get("risk"),
        "model_score": entry.get("model_score"),
        "confidence_score": entry.get("confidence_score"),
        "confidence_label": entry.get("confidence_label"),
        "data_as_of": entry.get("data_as_of"),
        "realized_5d_return": entry.get("realized_5d_return"),
        "realized_5d_excess_return": entry.get("realized_5d_excess_return"),
        "realized_5d_sector_return": entry.get("realized_5d_sector_return"),
        "realized_5d_sector_excess_return": entry.get("realized_5d_sector_excess_return"),
        "outcome": entry.get("outcome"),
    }


def main() -> None:
    set_pipeline_scope("track_record")
    universe_path = resolve_universe_csv_path()
    sector_map = load_sector_map(universe_path)
    generated_at = iso_utc(now_utc())
    market_week = build_market_week()
    expected_next_refresh_at, stale_after = build_refresh_window(market_week)

    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as handle:
            history_payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        history_payload = {}

    normalized_entries = [
        normalize_history_entry(entry)
        for entry in load_existing_history_entries(HISTORY_PATH)
    ]
    upstream_reasons = extract_payload_degraded_reasons(history_payload, "history")
    enriched_entries = [enrich_entry_metrics(entry, sector_map) for entry in normalized_entries]
    serialized_entries = [serialize_track_record_entry(entry) for entry in reversed(enriched_entries)]
    closed_dates = [
        str(entry.get("week_end"))
        for entry in enriched_entries
        if isinstance(entry.get("realized_5d_return"), (int, float)) and isinstance(entry.get("week_end"), str)
    ]
    data_as_of = max(closed_dates) if closed_dates else market_week.week_end.isoformat()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at,
        "data_as_of": data_as_of,
        "expected_next_refresh_at": expected_next_refresh_at,
        "stale_after": stale_after,
        "market_context": {
            "timezone": "America/New_York",
            "week_id": market_week.week_id,
            "week_label": market_week.week_label,
            "week_start": market_week.week_start.isoformat(),
            "week_end": market_week.week_end.isoformat(),
        },
        "selection_thresholds": {
            "overall_score": OVERALL_SELECTION_THRESHOLD,
            "risk_scores": RISK_SELECTION_THRESHOLDS,
            "minimum_confidence": MIN_CONFIDENCE_THRESHOLD,
        },
        "summary": build_summary_metrics(enriched_entries),
        "signal_block_report": build_signal_block_report(enriched_entries),
        "candidate_ranking_report": build_candidate_ranking_report(enriched_entries),
        "no_pick_report": build_no_pick_report(enriched_entries),
        "risk_breakdown": build_risk_breakdown(enriched_entries),
        "entries": serialized_entries,
        "data_quality": build_data_quality_block(
            scope="track_record",
            extra_reasons=upstream_reasons,
        ),
    }
    write_json(TRACK_RECORD_PATH, payload)
    print("[INFO] track_record.json updated.")


if __name__ == "__main__":
    main()
