import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    require(candidate.get("risk") in RISK_BUCKETS, f"{context} risk must be one of {RISK_BUCKETS}")
    require(isinstance(candidate.get("model_score"), (int, float)), f"{context} is missing model_score")
    require(isinstance(candidate.get("confidence_score"), (int, float)), f"{context} is missing confidence_score")
    require(candidate.get("confidence_label") in ("low", "medium", "high"), f"{context} confidence_label is invalid")
    require(isinstance(candidate.get("price_as_of"), str), f"{context} is missing price_as_of")
    parse_iso_date(candidate["price_as_of"])

    metrics = candidate.get("metrics")
    require(isinstance(metrics, dict), f"{context} is missing metrics")
    for key in ("momentum_5d", "daily_volatility", "news_sentiment", "raw_news_sentiment"):
        require(isinstance(metrics.get(key), (int, float)), f"{context} metrics.{key} must be numeric")

    score_breakdown = candidate.get("score_breakdown")
    require(isinstance(score_breakdown, dict), f"{context} is missing score_breakdown")
    for key in ("momentum", "volatility_penalty", "news_adjustment", "total"):
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


def validate_repository_outputs(base_dir: Path = Path(".")) -> None:
    current_pick = load_json(base_dir / "current_pick.json")
    risk_picks = load_json(base_dir / "risk_picks.json")
    history = load_json(base_dir / "history.json")

    validate_current_pick_payload(current_pick)
    validate_risk_picks_payload(risk_picks)
    validate_history_payload(history)

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
