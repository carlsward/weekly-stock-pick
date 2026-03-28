import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from backend.sector_utils import load_active_sectors, load_symbol_metadata
except ImportError:
    from sector_utils import load_active_sectors, load_symbol_metadata

SCHEMA_VERSION = 2
RISK_BUCKETS = ("low", "medium", "high")


def parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    require(isinstance(payload, dict), f"{path.name} must contain a top-level JSON object")
    return payload


def validate_common_payload(payload: Dict[str, Any], filename: str) -> None:
    require(payload.get("schema_version") == SCHEMA_VERSION, f"{filename} must use schema_version={SCHEMA_VERSION}")
    require(isinstance(payload.get("model_version"), str), f"{filename} is missing model_version")

    generated_at = parse_iso_datetime(payload["generated_at"])
    data_as_of = parse_iso_date(payload["data_as_of"])
    expected_refresh = parse_iso_datetime(payload["expected_next_refresh_at"])
    stale_after = parse_iso_datetime(payload["stale_after"])

    require(stale_after > expected_refresh, f"{filename} stale_after must be after expected_next_refresh_at")
    require(data_as_of <= generated_at.date(), f"{filename} data_as_of cannot be in the future")

    market_context = payload.get("market_context")
    require(isinstance(market_context, dict), f"{filename} is missing market_context")
    week_start = parse_iso_date(market_context["week_start"])
    week_end = parse_iso_date(market_context["week_end"])
    require(week_start.weekday() == 0, f"{filename} week_start must be a Monday")
    require(week_end.weekday() == 4, f"{filename} week_end must be a Friday")
    require((week_end - week_start).days == 4, f"{filename} week_end must be the Friday of the same market week")
    require(isinstance(market_context.get("week_id"), str), f"{filename} is missing market_context.week_id")
    require(isinstance(market_context.get("week_label"), str), f"{filename} is missing market_context.week_label")

    thresholds = payload.get("selection_thresholds")
    require(isinstance(thresholds, dict), f"{filename} is missing selection_thresholds")
    risk_scores = thresholds.get("risk_scores")
    require(isinstance(risk_scores, dict), f"{filename} selection_thresholds.risk_scores must be an object")
    require(all(risk in risk_scores for risk in RISK_BUCKETS), f"{filename} must expose all risk score thresholds")


def validate_candidate(candidate: Dict[str, Any], context: str) -> None:
    require(isinstance(candidate.get("symbol"), str), f"{context} is missing symbol")
    require(isinstance(candidate.get("company_name"), str), f"{context} is missing company_name")
    require(isinstance(candidate.get("sector"), str), f"{context} is missing sector")
    require(candidate.get("risk") in RISK_BUCKETS, f"{context} risk must be one of {RISK_BUCKETS}")
    require(isinstance(candidate.get("model_score"), (int, float)), f"{context} is missing model_score")
    require(isinstance(candidate.get("confidence_score"), (int, float)), f"{context} is missing confidence_score")
    require(candidate.get("confidence_label") in ("low", "medium", "high"), f"{context} confidence_label is invalid")
    require(isinstance(candidate.get("price_as_of"), str), f"{context} is missing price_as_of")
    parse_iso_date(candidate["price_as_of"])
    macro_as_of = candidate.get("macro_as_of")
    if macro_as_of is not None:
        require(isinstance(macro_as_of, str), f"{context}.macro_as_of must be a string when present")
        parse_iso_date(macro_as_of)
    sector_as_of = candidate.get("sector_as_of")
    if sector_as_of is not None:
        require(isinstance(sector_as_of, str), f"{context}.sector_as_of must be a string when present")
        parse_iso_date(sector_as_of)

    metrics = candidate.get("metrics")
    require(isinstance(metrics, dict), f"{context} is missing metrics")
    for key in (
        "momentum_5d",
        "daily_volatility",
        "news_sentiment",
        "raw_news_sentiment",
        "macro_sentiment",
        "macro_confidence",
        "sector_sentiment",
        "sector_confidence",
        "market_relative_5d",
        "market_relative_20d",
        "sector_relative_5d",
        "sector_relative_20d",
    ):
        require(isinstance(metrics.get(key), (int, float)), f"{context} metrics.{key} must be numeric")

    score_breakdown = candidate.get("score_breakdown")
    require(isinstance(score_breakdown, dict), f"{context} is missing score_breakdown")
    for key in (
        "momentum",
        "market_relative_strength",
        "sector_relative_strength",
        "volatility_penalty",
        "news_adjustment",
        "macro_adjustment",
        "sector_adjustment",
        "total",
    ):
        require(isinstance(score_breakdown.get(key), (int, float)), f"{context} score_breakdown.{key} must be numeric")

    reasons = candidate.get("reasons")
    require(isinstance(reasons, list) and reasons, f"{context} must include non-empty reasons")

    news_evidence = candidate.get("news_evidence")
    if news_evidence is not None:
        require(isinstance(news_evidence, list), f"{context} news_evidence must be a list when present")
        for index, article in enumerate(news_evidence):
            article_context = f"{context}.news_evidence[{index}]"
            require(isinstance(article, dict), f"{article_context} must be an object")
            require(isinstance(article.get("title"), str), f"{article_context} must include title")

    macro_evidence = candidate.get("macro_evidence")
    if macro_evidence is not None:
        require(isinstance(macro_evidence, list), f"{context} macro_evidence must be a list when present")
        for index, article in enumerate(macro_evidence):
            article_context = f"{context}.macro_evidence[{index}]"
            require(isinstance(article, dict), f"{article_context} must be an object")
            require(isinstance(article.get("title"), str), f"{article_context} must include title")

    thesis_monitor = candidate.get("thesis_monitor")
    if thesis_monitor is not None:
        require(isinstance(thesis_monitor, dict), f"{context} thesis_monitor must be an object")
        require(thesis_monitor.get("status") in ("healthy", "watch", "risk"), f"{context} thesis_monitor.status is invalid")
        require(isinstance(thesis_monitor.get("headline"), str), f"{context} thesis_monitor.headline must be present")
        require(isinstance(thesis_monitor.get("summary"), str), f"{context} thesis_monitor.summary must be present")
        alerts = thesis_monitor.get("alerts")
        require(isinstance(alerts, list), f"{context} thesis_monitor.alerts must be a list")
        signals = thesis_monitor.get("signals")
        require(isinstance(signals, list) and signals, f"{context} thesis_monitor.signals must be a non-empty list")
        for index, signal in enumerate(signals):
            signal_context = f"{context}.thesis_monitor.signals[{index}]"
            require(isinstance(signal, dict), f"{signal_context} must be an object")
            require(isinstance(signal.get("label"), str), f"{signal_context} must include label")
            require(signal.get("state") in ("positive", "watch", "risk"), f"{signal_context} state is invalid")
            require(isinstance(signal.get("value"), str), f"{signal_context} must include value")
            require(isinstance(signal.get("detail"), str), f"{signal_context} must include detail")


def validate_selection(selection: Dict[str, Any], context: str) -> None:
    require(selection.get("status") in ("picked", "no_pick"), f"{context} status must be picked or no_pick")
    require(isinstance(selection.get("status_reason"), str), f"{context} is missing status_reason")
    require(isinstance(selection.get("threshold_score"), (int, float)), f"{context} is missing threshold_score")
    require(isinstance(selection.get("threshold_confidence"), (int, float)), f"{context} is missing threshold_confidence")

    if selection["status"] == "picked":
        require(selection.get("pick") is not None, f"{context} must include pick when status is picked")
        validate_candidate(selection["pick"], f"{context}.pick")
    else:
        require(selection.get("pick") is None, f"{context} pick must be null when status is no_pick")
        best_candidate = selection.get("best_candidate")
        if best_candidate is not None:
            require(isinstance(best_candidate.get("symbol"), str), f"{context}.best_candidate must include symbol")
            require(isinstance(best_candidate.get("model_score"), (int, float)), f"{context}.best_candidate must include model_score")


def validate_current_pick_payload(payload: Dict[str, Any]) -> None:
    validate_common_payload(payload, "current_pick.json")
    selection = payload.get("selection")
    require(isinstance(selection, dict), "current_pick.json is missing selection")
    validate_selection(selection, "current_pick.json.selection")


def validate_risk_picks_payload(payload: Dict[str, Any]) -> None:
    validate_common_payload(payload, "risk_picks.json")
    overall = payload.get("overall_selection")
    require(isinstance(overall, dict), "risk_picks.json is missing overall_selection")
    validate_selection(overall, "risk_picks.json.overall_selection")

    risk_selections = payload.get("risk_selections")
    require(isinstance(risk_selections, dict), "risk_picks.json is missing risk_selections")
    for risk in RISK_BUCKETS:
        require(risk in risk_selections, f"risk_picks.json must include {risk} risk selection")
        validate_selection(risk_selections[risk], f"risk_picks.json.risk_selections.{risk}")


def validate_history_payload(payload: Dict[str, Any]) -> None:
    require(payload.get("schema_version") == SCHEMA_VERSION, "history.json must use the current schema version")
    require(isinstance(payload.get("model_version"), str), "history.json is missing model_version")
    parse_iso_datetime(payload["generated_at"])

    entries = payload.get("entries")
    require(isinstance(entries, list), "history.json entries must be a list")

    seen_week_ids = set()
    previous_week_start = None
    for index, entry in enumerate(entries):
        context = f"history.json.entries[{index}]"
        require(isinstance(entry.get("week_id"), str), f"{context} is missing week_id")
        require(entry["week_id"] not in seen_week_ids, f"{context} duplicates an existing week_id")
        seen_week_ids.add(entry["week_id"])

        week_start = parse_iso_date(entry["week_start"])
        week_end = parse_iso_date(entry["week_end"])
        require(week_start <= week_end, f"{context} week_start must be on or before week_end")

        if previous_week_start is not None:
            require(week_start >= previous_week_start, f"{context} entries must be sorted by week_start")
        previous_week_start = week_start

        require(entry.get("status") in ("picked", "no_pick"), f"{context} status must be picked or no_pick")
        require(isinstance(entry.get("status_reason"), str), f"{context} is missing status_reason")
        require(isinstance(entry.get("model_version"), str), f"{context} is missing model_version")


def validate_sector_scores_payload(payload: Dict[str, Any], universe_path: Path) -> None:
    parse_iso_datetime(payload["generated_at"])
    parse_iso_date(payload["last_updated"])
    require(isinstance(payload.get("lookback_days"), int), "sector_scores.json must include lookback_days")
    require(isinstance(payload.get("article_count"), int), "sector_scores.json must include article_count")
    require(isinstance(payload.get("source_count"), int), "sector_scores.json must include source_count")
    require(isinstance(payload.get("llm_model"), str), "sector_scores.json must include llm_model")
    require(isinstance(payload.get("summary"), str), "sector_scores.json must include summary")

    sector_scores = payload.get("sector_scores")
    require(isinstance(sector_scores, dict) and sector_scores, "sector_scores.json must include non-empty sector_scores")
    expected_sectors = load_active_sectors(universe_path)
    for sector in expected_sectors:
        require(sector in sector_scores, f"sector_scores.json is missing sector {sector}")
        sector_payload = sector_scores[sector]
        context = f"sector_scores.json.sector_scores.{sector}"
        require(isinstance(sector_payload, dict), f"{context} must be an object")
        require(isinstance(sector_payload.get("display_name"), str), f"{context} must include display_name")
        require(isinstance(sector_payload.get("score"), (int, float)), f"{context} score must be numeric")
        require(isinstance(sector_payload.get("confidence"), (int, float)), f"{context} confidence must be numeric")
        require(
            sector_payload.get("direction") in ("bullish", "bearish", "neutral"),
            f"{context} direction is invalid",
        )
        reasons = sector_payload.get("reasons")
        require(isinstance(reasons, list) and reasons, f"{context} reasons must be a non-empty list")
        last_updated = sector_payload.get("last_updated")
        if last_updated is not None:
            require(isinstance(last_updated, str), f"{context}.last_updated must be a string when present")
            parse_iso_date(last_updated)

    symbol_scores = payload.get("symbol_scores")
    require(isinstance(symbol_scores, dict) and symbol_scores, "sector_scores.json must include non-empty symbol_scores")
    expected_symbols = load_symbol_metadata(universe_path)
    for symbol, metadata in expected_symbols.items():
        require(symbol in symbol_scores, f"sector_scores.json is missing symbol {symbol}")
        symbol_payload = symbol_scores[symbol]
        context = f"sector_scores.json.symbol_scores.{symbol}"
        require(isinstance(symbol_payload, dict), f"{context} must be an object")
        require(symbol_payload.get("symbol") == symbol, f"{context} must include symbol")
        require(isinstance(symbol_payload.get("company_name"), str), f"{context} must include company_name")
        require(symbol_payload.get("sector") == metadata["sector"], f"{context} sector must match universe.csv")
        require(isinstance(symbol_payload.get("score"), (int, float)), f"{context} score must be numeric")
        require(isinstance(symbol_payload.get("confidence"), (int, float)), f"{context} confidence must be numeric")
        require(
            symbol_payload.get("direction") in ("bullish", "bearish", "neutral"),
            f"{context} direction is invalid",
        )
        reasons = symbol_payload.get("reasons")
        require(isinstance(reasons, list) and reasons, f"{context} reasons must be a non-empty list")
        last_updated = symbol_payload.get("last_updated")
        if last_updated is not None:
            require(isinstance(last_updated, str), f"{context}.last_updated must be a string when present")
            parse_iso_date(last_updated)

    events = payload.get("events")
    require(isinstance(events, list), "sector_scores.json events must be a list")


def validate_news_scores_payload(payload: Dict[str, Any], universe_path: Path) -> None:
    expected_symbols = load_symbol_metadata(universe_path)
    require(isinstance(payload, dict) and payload, "news_scores.json must contain symbol entries")
    for symbol in expected_symbols:
        require(symbol in payload, f"news_scores.json is missing symbol {symbol}")
        symbol_payload = payload[symbol]
        context = f"news_scores.json.{symbol}"
        require(isinstance(symbol_payload, dict), f"{context} must be an object")
        for key in (
            "news_score",
            "news_confidence",
            "raw_sentiment",
            "calibrated_sentiment",
            "effective_article_count",
            "average_relevance",
        ):
            require(isinstance(symbol_payload.get(key), (int, float)), f"{context}.{key} must be numeric")
        require(isinstance(symbol_payload.get("article_count"), int), f"{context}.article_count must be an integer")
        require(isinstance(symbol_payload.get("source_count"), int), f"{context}.source_count must be an integer")
        require(isinstance(symbol_payload.get("dominant_signal"), str), f"{context}.dominant_signal must be a string")
        require(isinstance(symbol_payload.get("analysis_method"), str), f"{context}.analysis_method must be a string")
        reasons = symbol_payload.get("news_reasons")
        require(isinstance(reasons, list) and reasons, f"{context}.news_reasons must be a non-empty list")
        top_articles = symbol_payload.get("top_articles")
        require(isinstance(top_articles, list), f"{context}.top_articles must be a list")
        last_updated = symbol_payload.get("last_updated")
        if last_updated is not None:
            require(isinstance(last_updated, str), f"{context}.last_updated must be a string when present")
            parse_iso_date(last_updated)


def validate_repository_outputs(base_dir: Path = Path(".")) -> None:
    current_pick = load_json(base_dir / "current_pick.json")
    risk_picks = load_json(base_dir / "risk_picks.json")
    history = load_json(base_dir / "history.json")
    news_scores = load_json(base_dir / "news_scores.json")
    sector_scores = load_json(base_dir / "sector_scores.json")

    validate_current_pick_payload(current_pick)
    validate_risk_picks_payload(risk_picks)
    validate_history_payload(history)
    validate_news_scores_payload(news_scores, base_dir / "universe.csv")
    validate_sector_scores_payload(sector_scores, base_dir / "universe.csv")

    overall_current = current_pick["selection"]["status"]
    overall_risk = risk_picks["overall_selection"]["status"]
    require(overall_current == overall_risk, "current_pick.json and risk_picks.json must agree on overall selection status")

    current_pick_symbol = None
    overall_pick = current_pick["selection"].get("pick")
    if overall_pick is not None:
        current_pick_symbol = overall_pick.get("symbol")

    risk_pick_symbol = None
    risk_overall_pick = risk_picks["overall_selection"].get("pick")
    if risk_overall_pick is not None:
        risk_pick_symbol = risk_overall_pick.get("symbol")

    require(current_pick_symbol == risk_pick_symbol, "current_pick.json and risk_picks.json must agree on the overall symbol")


def main() -> None:
    validate_repository_outputs(Path("."))
    print("Output contracts are valid.")


if __name__ == "__main__":
    main()
