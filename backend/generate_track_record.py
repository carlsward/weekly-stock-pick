from __future__ import annotations

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
    universe_path = resolve_universe_csv_path()
    sector_map = load_sector_map(universe_path)
    generated_at = iso_utc(now_utc())
    market_week = build_market_week()
    expected_next_refresh_at, stale_after = build_refresh_window(market_week)

    normalized_entries = [
        normalize_history_entry(entry)
        for entry in load_existing_history_entries(HISTORY_PATH)
    ]
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
        "risk_breakdown": build_risk_breakdown(enriched_entries),
        "entries": serialized_entries,
    }
    write_json(TRACK_RECORD_PATH, payload)
    print("[INFO] track_record.json updated.")


if __name__ == "__main__":
    main()
