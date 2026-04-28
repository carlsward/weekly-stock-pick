import calendar
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from backend.generate_pick import (
        MARKET_BENCHMARK_SYMBOL,
        MARKET_TIMEZONE,
        MARKET_TZ,
        SCHEMA_VERSION,
        GenerationStats,
        ModelCalibration,
        NewsSnapshot,
        SectorSnapshot,
        MacroSnapshot,
        blend_weight_maps,
        clamp,
        confidence_label,
        compute_confidence_score,
        compute_max_drawdown,
        daily_returns,
        dedupe_layer_penalties,
        fetch_price_frame_cached,
        fetch_price_series_cached,
        format_pct,
        format_score,
        information_coefficient,
        iso_utc,
        load_news_scores,
        load_sector_scores,
        load_universe,
        market_today,
        latest_payload_article_date,
        normalize_news_evidence,
        normalize_sector_supporting_articles,
        now_utc,
        parse_iso_datetime,
        price_failure_no_pick_reason,
        relative_return,
        resolve_universe_csv_path,
        safe_divide,
        std_dev,
        write_json,
        is_live_price_provider_failure,
        PRICE_SERIES_CACHE,
        PRICE_FRAME_CACHE,
        SECTOR_ETF_BY_SECTOR,
    )
    from backend.sector_utils import load_sector_map, sector_display_name
    from backend.pipeline_runtime import (
        build_data_quality_block,
        extract_payload_degraded_reasons,
        set_pipeline_scope,
    )
except ImportError:
    from generate_pick import (
        MARKET_BENCHMARK_SYMBOL,
        MARKET_TIMEZONE,
        MARKET_TZ,
        SCHEMA_VERSION,
        GenerationStats,
        ModelCalibration,
        NewsSnapshot,
        SectorSnapshot,
        MacroSnapshot,
        blend_weight_maps,
        clamp,
        confidence_label,
        compute_confidence_score,
        compute_max_drawdown,
        daily_returns,
        dedupe_layer_penalties,
        fetch_price_frame_cached,
        fetch_price_series_cached,
        format_pct,
        format_score,
        information_coefficient,
        iso_utc,
        load_news_scores,
        load_sector_scores,
        load_universe,
        market_today,
        latest_payload_article_date,
        normalize_news_evidence,
        normalize_sector_supporting_articles,
        now_utc,
        parse_iso_datetime,
        price_failure_no_pick_reason,
        relative_return,
        resolve_universe_csv_path,
        safe_divide,
        std_dev,
        write_json,
        is_live_price_provider_failure,
        PRICE_SERIES_CACHE,
        PRICE_FRAME_CACHE,
        SECTOR_ETF_BY_SECTOR,
    )
    from sector_utils import load_sector_map, sector_display_name
    from pipeline_runtime import (
        build_data_quality_block,
        extract_payload_degraded_reasons,
        set_pipeline_scope,
    )


MONTHLY_MODEL_VERSION = "v3.4-monthly"
MONTHLY_PICK_PATH = Path("monthly_pick.json")
MONTHLY_HISTORY_PATH = Path("monthly_history.json")
ENABLE_MONTHLY_HISTORY_REALIZED_ENRICHMENT_ENV = "ENABLE_MONTHLY_HISTORY_REALIZED_ENRICHMENT"

MONTHLY_HORIZON_TRADING_DAYS = 20
MONTHLY_PRICE_WINDOW_DAYS = 90
MONTHLY_SELECTION_THRESHOLD = 0.12
MONTHLY_MIN_CONFIDENCE_THRESHOLD = 0.60
MONTHLY_STALE_GRACE_HOURS = 72
MONTHLY_NEWS_STALE_AFTER_DAYS = 7
MONTHLY_GLOBAL_STALE_AFTER_DAYS = 10

MONTHLY_MOMENTUM_20D_SCALE = 0.16
MONTHLY_MOMENTUM_60D_SCALE = 0.30
MONTHLY_TREND_GAP_SCALE = 0.12
MONTHLY_VOLUME_TREND_SCALE = 0.65
MONTHLY_VOLATILITY_SCALE = 0.05
MONTHLY_DOWNSIDE_SCALE = 0.035
MONTHLY_MAX_DRAWDOWN_SCALE = 0.18
MONTHLY_RELATIVE_STRENGTH_SCALE = 0.14

MONTHLY_CALIBRATION_LOOKBACK_DAYS = 540
MONTHLY_CALIBRATION_STEP_DAYS = 10
MONTHLY_CALIBRATION_MIN_ROWS = 120
MONTHLY_CALIBRATION_RIDGE_ALPHA = 0.75
MONTHLY_CALIBRATION_PREDICTION_PCTL = 0.90
MONTHLY_CALIBRATION_MIN_ABS_IC = 0.03
MONTHLY_TECHNICAL_SCORE_TARGET_SCALE = 0.34

MONTHLY_BLOCK_BASE_PRIORS = {
    "news": 0.14,
    "macro": 0.12,
    "sector": 0.09,
}
MONTHLY_BLOCK_WEIGHT_FLOORS = {
    "news": 0.09,
    "macro": 0.07,
    "sector": 0.05,
}
MONTHLY_BLOCK_WEIGHT_CAPS = {
    "news": 0.18,
    "macro": 0.15,
    "sector": 0.12,
}
MONTHLY_BLOCK_CALIBRATION_MIN_ROWS = 6
MONTHLY_BLOCK_CALIBRATION_RIDGE_ALPHA = 0.45
MONTHLY_BLOCK_CALIBRATION_FULL_TRUST_ROWS = 24

MONTHLY_HORIZON_MULTIPLIERS = {
    "1-3d": 0.45,
    "1w": 0.68,
    "1-2w": 0.86,
    "2-4w": 1.0,
    "unclear": 0.82,
}

MONTHLY_TECHNICAL_FEATURE_ORDER = (
    "monthly_momentum_short",
    "monthly_momentum_medium",
    "monthly_trend_gap",
    "monthly_positive_ratio",
    "monthly_volume_confirmation",
    "monthly_market_relative",
    "monthly_sector_relative",
    "monthly_inverse_volatility",
    "monthly_inverse_downside",
    "monthly_inverse_drawdown",
)

MONTHLY_TECHNICAL_GROUPS = {
    "trend_strength": (
        "monthly_momentum_short",
        "monthly_momentum_medium",
        "monthly_trend_gap",
        "monthly_positive_ratio",
    ),
    "relative_strength": ("monthly_market_relative", "monthly_sector_relative"),
    "participation": ("monthly_volume_confirmation",),
    "risk_control": (
        "monthly_inverse_volatility",
        "monthly_inverse_downside",
        "monthly_inverse_drawdown",
    ),
}

MONTHLY_DEFAULT_TECHNICAL_FEATURE_WEIGHTS = {
    "monthly_momentum_short": 0.24,
    "monthly_momentum_medium": 0.24,
    "monthly_trend_gap": 0.08,
    "monthly_positive_ratio": 0.08,
    "monthly_volume_confirmation": 0.07,
    "monthly_market_relative": 0.10,
    "monthly_sector_relative": 0.06,
    "monthly_inverse_volatility": 0.06,
    "monthly_inverse_downside": 0.04,
    "monthly_inverse_drawdown": 0.03,
}


@dataclass(frozen=True)
class MarketMonth:
    month_id: str
    month_label: str
    month_start: date
    month_end: date
    rebalance_date: date
    next_rebalance_date: date
    horizon_trading_days: int


@dataclass
class MonthlyCandidate:
    symbol: str
    company_name: str
    sector: str
    reasons: List[str]
    risk_level: str
    total_score: float
    confidence_score: float
    confidence_label: str
    price_as_of: str
    news_as_of: Optional[str]
    macro_as_of: Optional[str]
    sector_as_of: Optional[str]
    article_count: int
    effective_article_count: float
    source_count: int
    average_relevance: float
    momentum_20d: float
    momentum_60d: float
    volatility_20d: float
    downside_volatility_20d: float
    max_drawdown_60d: float
    trend_gap_50d: float
    positive_day_ratio_20d: float
    volume_trend_10d: float
    market_relative_20d: float
    market_relative_60d: float
    sector_relative_20d: float
    sector_relative_60d: float
    news_score: float
    news_confidence: float
    macro_score: float
    macro_confidence: float
    sector_score: float
    sector_confidence: float
    raw_sentiment: float
    calibrated_sentiment: float
    dominant_signal: str
    score_breakdown: Dict[str, float]
    news_evidence: List[Dict[str, Any]]
    macro_evidence: List[Dict[str, Any]]


@dataclass(frozen=True)
class MonthlySelectionDecision:
    status: str
    status_reason: str
    threshold_score: float
    threshold_confidence: float
    pick: Optional[MonthlyCandidate]
    best_candidate: Optional[MonthlyCandidate]


def monthly_history_realized_enrichment_enabled() -> bool:
    raw = os.getenv(ENABLE_MONTHLY_HISTORY_REALIZED_ENRICHMENT_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def next_month_anchor(year: int, month: int) -> Tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def build_market_month(now: Optional[datetime] = None) -> MarketMonth:
    market_now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    reference_date = market_now.date()
    year = reference_date.year
    month = reference_date.month
    month_start = date(year, month, 1)
    month_end = date(year, month, calendar.monthrange(year, month)[1])
    rebalance_date = month_start
    next_year, next_month = next_month_anchor(year, month)
    next_rebalance_date = date(next_year, next_month, 1)
    month_id = f"{year}-{month:02d}"
    month_label = reference_date.strftime("%B %Y")
    return MarketMonth(
        month_id=month_id,
        month_label=month_label,
        month_start=month_start,
        month_end=month_end,
        rebalance_date=rebalance_date,
        next_rebalance_date=next_rebalance_date,
        horizon_trading_days=MONTHLY_HORIZON_TRADING_DAYS,
    )


def build_monthly_refresh_window(market_month: MarketMonth) -> Tuple[str, str]:
    next_refresh = datetime.combine(
        market_month.next_rebalance_date,
        time(hour=12, tzinfo=timezone.utc),
    )
    stale_after = next_refresh + timedelta(hours=MONTHLY_STALE_GRACE_HOURS)
    return iso_utc(next_refresh), iso_utc(stale_after)


def build_monthly_relative_strength_metrics(
    stock_metrics: Dict[str, float],
    market_metrics: Dict[str, float],
    sector_metrics: Dict[str, float],
) -> Dict[str, float]:
    return {
        "market_relative_20d": relative_return(stock_metrics["momentum_20d"], market_metrics["momentum_20d"]),
        "market_relative_60d": relative_return(stock_metrics["momentum_60d"], market_metrics["momentum_60d"]),
        "sector_relative_20d": relative_return(stock_metrics["momentum_20d"], sector_metrics["momentum_20d"]),
        "sector_relative_60d": relative_return(stock_metrics["momentum_60d"], sector_metrics["momentum_60d"]),
    }


def classify_risk(volatility: float) -> str:
    if volatility < 0.01:
        return "low"
    if volatility < 0.02:
        return "medium"
    return "high"


def compute_monthly_metrics(price_series) -> Dict[str, float]:
    closes = price_series.closes
    volumes = price_series.volumes
    returns = daily_returns(closes)

    if len(closes) < 61 or len(returns) < 60:
        raise ValueError("At least 61 daily closes are required for monthly scoring")

    momentum_20d = safe_divide(closes[0], closes[20], 1.0) - 1.0
    momentum_60d = safe_divide(closes[0], closes[60], 1.0) - 1.0
    volatility_20d = std_dev(returns[:20])
    downside_returns = [item for item in returns[:20] if item < 0]
    downside_volatility_20d = (
        math.sqrt(sum(item * item for item in downside_returns) / len(downside_returns))
        if downside_returns
        else 0.0
    )
    average_recent_50 = sum(closes[:50]) / 50
    trend_gap_50d = safe_divide(closes[0], average_recent_50, 1.0) - 1.0
    positive_day_ratio_20d = sum(1 for item in returns[:20] if item > 0) / 20
    max_drawdown_60d = compute_max_drawdown(closes[:61])

    recent_volume = sum(volumes[:10]) / 10 if len(volumes) >= 10 else 0.0
    earlier_volume_window = volumes[10:40]
    baseline_volume = (
        sum(earlier_volume_window) / len(earlier_volume_window)
        if earlier_volume_window
        else recent_volume
    )
    volume_trend_10d = safe_divide(recent_volume, baseline_volume, 1.0) - 1.0 if baseline_volume > 0 else 0.0

    return {
        "momentum_20d": momentum_20d,
        "momentum_60d": momentum_60d,
        "volatility_20d": volatility_20d,
        "downside_volatility_20d": downside_volatility_20d,
        "trend_gap_50d": trend_gap_50d,
        "positive_day_ratio_20d": positive_day_ratio_20d,
        "max_drawdown_60d": max_drawdown_60d,
        "volume_trend_10d": volume_trend_10d,
    }


def build_monthly_normalized_features(
    *,
    momentum_20d: float,
    momentum_60d: float,
    trend_gap_50d: float,
    positive_day_ratio_20d: float,
    volume_trend_10d: float,
    volatility_20d: float,
    downside_volatility_20d: float,
    max_drawdown_60d: float,
    market_relative_20d: float,
    market_relative_60d: float,
    sector_relative_20d: float,
    sector_relative_60d: float,
) -> Dict[str, float]:
    normalized_market_relative = clamp(
        ((market_relative_20d + market_relative_60d) / 2.0) / MONTHLY_RELATIVE_STRENGTH_SCALE,
        -1.0,
        1.0,
    )
    normalized_sector_relative = clamp(
        ((sector_relative_20d + sector_relative_60d) / 2.0) / MONTHLY_RELATIVE_STRENGTH_SCALE,
        -1.0,
        1.0,
    )
    return {
        "monthly_momentum_short": clamp(momentum_20d / MONTHLY_MOMENTUM_20D_SCALE, -1.0, 1.0),
        "monthly_momentum_medium": clamp(momentum_60d / MONTHLY_MOMENTUM_60D_SCALE, -1.0, 1.0),
        "monthly_trend_gap": clamp(trend_gap_50d / MONTHLY_TREND_GAP_SCALE, -1.0, 1.0),
        "monthly_positive_ratio": clamp((positive_day_ratio_20d - 0.5) * 2.0, -1.0, 1.0),
        "monthly_volume_confirmation": clamp(volume_trend_10d / MONTHLY_VOLUME_TREND_SCALE, -1.0, 1.0),
        "monthly_market_relative": normalized_market_relative,
        "monthly_sector_relative": normalized_sector_relative,
        "monthly_inverse_volatility": -clamp(volatility_20d / MONTHLY_VOLATILITY_SCALE, 0.0, 1.0),
        "monthly_inverse_downside": -clamp(downside_volatility_20d / MONTHLY_DOWNSIDE_SCALE, 0.0, 1.0),
        "monthly_inverse_drawdown": -clamp(max_drawdown_60d / MONTHLY_MAX_DRAWDOWN_SCALE, 0.0, 1.0),
    }


def normalize_monthly_weight_map(weight_map: Dict[str, float]) -> Dict[str, float]:
    ordered = {
        name: max(0.0, float(weight_map.get(name, 0.0)))
        for name in MONTHLY_TECHNICAL_FEATURE_ORDER
    }
    total = sum(ordered.values())
    if total <= 1e-9:
        fallback_total = sum(MONTHLY_DEFAULT_TECHNICAL_FEATURE_WEIGHTS.values())
        return {
            name: MONTHLY_DEFAULT_TECHNICAL_FEATURE_WEIGHTS[name] / fallback_total
            for name in MONTHLY_TECHNICAL_FEATURE_ORDER
        }
    return {name: value / total for name, value in ordered.items()}


def monthly_horizon_multiplier(supporting_articles: Sequence[Dict[str, Any]]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for article in supporting_articles:
        if not isinstance(article, dict):
            continue
        horizon = str(article.get("horizon", "")).strip() or "unclear"
        multiplier = MONTHLY_HORIZON_MULTIPLIERS.get(horizon, MONTHLY_HORIZON_MULTIPLIERS["unclear"])
        weight = float(article.get("weight", 1.0)) if isinstance(article.get("weight"), (int, float)) else 1.0
        if weight <= 0:
            continue
        weighted_sum += multiplier * weight
        total_weight += weight
    if total_weight <= 1e-9:
        return 1.0
    return clamp(weighted_sum / total_weight, 0.35, 1.0)


def build_monthly_news_snapshot(news_info: Optional[dict]) -> NewsSnapshot:
    if not news_info:
        return NewsSnapshot(
            news_score=0.5,
            reasons=[
                "No medium-term company catalyst was available, so the monthly company-news overlay stayed neutral."
            ],
            raw_sentiment=0.0,
            calibrated_sentiment=0.0,
            article_count=0,
            effective_article_count=0.0,
            source_count=0,
            average_relevance=0.0,
            news_confidence=0.18,
            dominant_signal="neutral",
            last_updated=None,
            news_evidence=[],
        )

    reasons = news_info.get("news_reasons")
    normalized_reasons = [str(item) for item in reasons] if isinstance(reasons, list) else []
    raw_sentiment = float(news_info.get("raw_sentiment", 0.0))
    base_calibrated_sentiment = float(news_info.get("calibrated_sentiment", raw_sentiment))
    article_count = int(news_info.get("article_count", 0))
    base_news_confidence = float(news_info.get("news_confidence", 0.24 if article_count else 0.18))
    base_news_score = float(news_info.get("news_score", 0.5))
    last_updated = news_info.get("last_updated") or latest_payload_article_date(news_info.get("top_articles"))
    now = now_utc()
    freshness_multiplier = 1.0

    parsed_last_updated = parse_iso_datetime(last_updated)
    if parsed_last_updated and parsed_last_updated < now - timedelta(days=MONTHLY_NEWS_STALE_AFTER_DAYS):
        age_days = max(1, (now - parsed_last_updated).days)
        freshness_multiplier = 0.72 if age_days <= 14 else 0.52
        normalized_reasons.append(
            f"Company-news evidence is {age_days} days old, so the monthly overlay was reduced."
        )

    calibrated_sentiment = base_calibrated_sentiment * freshness_multiplier
    adjusted_news_score = 0.5 + (base_news_score - 0.5) * freshness_multiplier
    adjusted_news_confidence = clamp(
        base_news_confidence * freshness_multiplier,
        0.16 if article_count else 0.18,
        0.95,
    )
    dominant_signal = str(news_info.get("dominant_signal", "neutral"))
    if freshness_multiplier < 1.0 and abs(calibrated_sentiment) < 0.12:
        dominant_signal = "neutral"

    return NewsSnapshot(
        news_score=adjusted_news_score,
        reasons=normalized_reasons,
        raw_sentiment=raw_sentiment,
        calibrated_sentiment=calibrated_sentiment,
        article_count=article_count,
        effective_article_count=float(news_info.get("effective_article_count", article_count)),
        source_count=int(news_info.get("source_count", 0)),
        average_relevance=float(news_info.get("average_relevance", 0.0)),
        news_confidence=adjusted_news_confidence,
        dominant_signal=dominant_signal,
        last_updated=last_updated,
        news_evidence=normalize_news_evidence(news_info),
    )


def build_monthly_sector_snapshot(
    sector: str,
    sector_scores_payload: Optional[Dict[str, Any]],
) -> SectorSnapshot:
    neutral_reason = (
        f"No durable broad catalyst clearly favored the {sector_display_name(sector)} sector, so the monthly sector overlay stayed neutral."
    )
    if not sector_scores_payload:
        return SectorSnapshot(
            sector=sector,
            sector_score=0.5,
            sector_confidence=0.18,
            direction="neutral",
            reasons=[neutral_reason],
            supporting_articles=[],
            last_updated=None,
        )

    sector_scores = sector_scores_payload.get("sector_scores")
    sector_info = sector_scores.get(sector) if isinstance(sector_scores, dict) else None
    if not isinstance(sector_info, dict):
        return SectorSnapshot(
            sector=sector,
            sector_score=0.5,
            sector_confidence=0.18,
            direction="neutral",
            reasons=[neutral_reason],
            supporting_articles=[],
            last_updated=str(sector_scores_payload.get("last_updated")) if sector_scores_payload.get("last_updated") else None,
        )

    reasons = sector_info.get("reasons")
    normalized_reasons = [str(item) for item in reasons] if isinstance(reasons, list) and reasons else [neutral_reason]
    base_score = float(sector_info.get("score", 0.5))
    base_confidence = float(sector_info.get("confidence", 0.18))
    supporting_articles = normalize_sector_supporting_articles(sector_info)
    last_updated = (
        str(sector_info.get("last_updated")).strip()
        if sector_info.get("last_updated")
        else latest_payload_article_date(supporting_articles)
    )
    if not last_updated and supporting_articles:
        last_updated = str(sector_scores_payload.get("last_updated")).strip() if sector_scores_payload.get("last_updated") else None

    freshness_multiplier = 1.0
    parsed_last_updated = parse_iso_datetime(last_updated)
    now = now_utc()
    if parsed_last_updated and parsed_last_updated < now - timedelta(days=MONTHLY_GLOBAL_STALE_AFTER_DAYS):
        age_days = max(1, (now - parsed_last_updated).days)
        freshness_multiplier = 0.76 if age_days <= 14 else 0.58
        normalized_reasons.append(
            f"Sector catalyst evidence is {age_days} days old, so the monthly overlay was reduced."
        )

    horizon_multiplier = monthly_horizon_multiplier(supporting_articles)
    if supporting_articles and horizon_multiplier < 0.95:
        normalized_reasons.append(
            "Most sector catalysts were short-lived, so the monthly sector overlay was downweighted."
        )

    total_multiplier = freshness_multiplier * horizon_multiplier
    score = 0.5 + (base_score - 0.5) * total_multiplier
    confidence = clamp(base_confidence * total_multiplier, 0.16, 0.95)
    direction = str(sector_info.get("direction", "neutral"))
    if total_multiplier < 0.85 and abs(score - 0.5) < 0.08:
        direction = "neutral"

    return SectorSnapshot(
        sector=sector,
        sector_score=score,
        sector_confidence=confidence,
        direction=direction,
        reasons=normalized_reasons,
        supporting_articles=supporting_articles,
        last_updated=str(last_updated).strip() if last_updated else None,
    )


def build_monthly_macro_snapshot(
    symbol: str,
    company_name: str,
    sector: str,
    sector_scores_payload: Optional[Dict[str, Any]],
) -> MacroSnapshot:
    neutral_reason = (
        f"No durable world-news catalyst clearly favored {company_name}, so the monthly global overlay stayed neutral."
    )
    if not sector_scores_payload:
        return MacroSnapshot(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            macro_score=0.5,
            macro_confidence=0.18,
            direction="neutral",
            reasons=[neutral_reason],
            supporting_articles=[],
            last_updated=None,
        )

    symbol_scores = sector_scores_payload.get("symbol_scores")
    symbol_info = symbol_scores.get(symbol) if isinstance(symbol_scores, dict) else None
    if not isinstance(symbol_info, dict):
        return MacroSnapshot(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            macro_score=0.5,
            macro_confidence=0.18,
            direction="neutral",
            reasons=[neutral_reason],
            supporting_articles=[],
            last_updated=str(sector_scores_payload.get("last_updated")) if sector_scores_payload.get("last_updated") else None,
        )

    reasons = symbol_info.get("reasons")
    normalized_reasons = [str(item) for item in reasons] if isinstance(reasons, list) and reasons else [neutral_reason]
    base_score = float(symbol_info.get("score", 0.5))
    base_confidence = float(symbol_info.get("confidence", 0.18))
    supporting_articles = normalize_sector_supporting_articles(symbol_info)
    last_updated = (
        str(symbol_info.get("last_updated")).strip()
        if symbol_info.get("last_updated")
        else latest_payload_article_date(supporting_articles)
    )
    if not last_updated and supporting_articles:
        last_updated = str(sector_scores_payload.get("last_updated")).strip() if sector_scores_payload.get("last_updated") else None

    freshness_multiplier = 1.0
    parsed_last_updated = parse_iso_datetime(last_updated)
    now = now_utc()
    if parsed_last_updated and parsed_last_updated < now - timedelta(days=MONTHLY_GLOBAL_STALE_AFTER_DAYS):
        age_days = max(1, (now - parsed_last_updated).days)
        freshness_multiplier = 0.76 if age_days <= 14 else 0.58
        normalized_reasons.append(
            f"World-news catalyst evidence is {age_days} days old, so the monthly overlay was reduced."
        )

    horizon_multiplier = monthly_horizon_multiplier(supporting_articles)
    if supporting_articles and horizon_multiplier < 0.95:
        normalized_reasons.append(
            "Most world-news catalysts were short-lived, so the monthly global overlay was downweighted."
        )

    total_multiplier = freshness_multiplier * horizon_multiplier
    score = 0.5 + (base_score - 0.5) * total_multiplier
    confidence = clamp(base_confidence * total_multiplier, 0.16, 0.95)
    direction = str(symbol_info.get("direction", "neutral"))
    if total_multiplier < 0.85 and abs(score - 0.5) < 0.08:
        direction = "neutral"

    return MacroSnapshot(
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        macro_score=score,
        macro_confidence=confidence,
        direction=direction,
        reasons=normalized_reasons,
        supporting_articles=supporting_articles,
        last_updated=str(last_updated).strip() if last_updated else None,
    )


def default_monthly_calibration() -> ModelCalibration:
    return ModelCalibration(
        technical_weights=normalize_monthly_weight_map(MONTHLY_DEFAULT_TECHNICAL_FEATURE_WEIGHTS),
        block_weights=dict(MONTHLY_BLOCK_BASE_PRIORS),
        technical_scale=MONTHLY_TECHNICAL_SCORE_TARGET_SCALE,
        training_row_count=0,
        training_ic=0.0,
        block_row_count=0,
        source="monthly_priors",
    )


def build_monthly_calibration_training_rows(
    universe: List[Tuple[str, str]],
    sector_map: Dict[str, str],
) -> List[Tuple[np.ndarray, float]]:
    training_rows: List[Tuple[np.ndarray, float]] = []
    market_frame = fetch_price_frame_cached(MARKET_BENCHMARK_SYMBOL)[["Date", "Close"]].rename(
        columns={"Close": "MarketClose"}
    )
    sector_frames = {
        sector: fetch_price_frame_cached(etf)[["Date", "Close"]].rename(columns={"Close": "SectorClose"})
        for sector, etf in SECTOR_ETF_BY_SECTOR.items()
    }

    for symbol, _company_name in universe:
        sector = sector_map[symbol]
        stock_frame = fetch_price_frame_cached(symbol)[["Date", "Close", "Volume"]]
        merged = stock_frame.merge(market_frame, on="Date", how="inner")
        merged = merged.merge(sector_frames[sector], on="Date", how="inner")
        merged = merged.sort_values("Date").reset_index(drop=True)
        if len(merged) < 85:
            continue

        merged = merged.tail(MONTHLY_CALIBRATION_LOOKBACK_DAYS).reset_index(drop=True)
        for end_index in range(60, len(merged) - MONTHLY_HORIZON_TRADING_DAYS, MONTHLY_CALIBRATION_STEP_DAYS):
            window = merged.iloc[end_index - 60:end_index + 1]
            if len(window) < 61:
                continue

            current_row = merged.iloc[end_index]
            future_index = end_index + MONTHLY_HORIZON_TRADING_DAYS
            future_row = merged.iloc[future_index]

            stock_series = fetch_price_series_cached(symbol, max_days=MONTHLY_PRICE_WINDOW_DAYS)
            del stock_series  # keep cache warm and rely on merged windows below

            stock_closes = window["Close"].tolist()[::-1]
            market_closes = window["MarketClose"].tolist()[::-1]
            sector_closes = window["SectorClose"].tolist()[::-1]
            stock_volumes = window["Volume"].tolist()[::-1]

            stock_metrics = compute_monthly_metrics(
                type("Series", (), {
                    "closes": stock_closes,
                    "volumes": stock_volumes,
                    "latest_trading_date": current_row["Date"].date().isoformat(),
                    "observations": len(stock_closes),
                })()
            )
            market_metrics = compute_monthly_metrics(
                type("Series", (), {
                    "closes": market_closes,
                    "volumes": [0.0] * len(market_closes),
                    "latest_trading_date": current_row["Date"].date().isoformat(),
                    "observations": len(market_closes),
                })()
            )
            sector_metrics = compute_monthly_metrics(
                type("Series", (), {
                    "closes": sector_closes,
                    "volumes": [0.0] * len(sector_closes),
                    "latest_trading_date": current_row["Date"].date().isoformat(),
                    "observations": len(sector_closes),
                })()
            )
            relative_metrics = build_monthly_relative_strength_metrics(stock_metrics, market_metrics, sector_metrics)
            technical_features = build_monthly_normalized_features(
                momentum_20d=stock_metrics["momentum_20d"],
                momentum_60d=stock_metrics["momentum_60d"],
                trend_gap_50d=stock_metrics["trend_gap_50d"],
                positive_day_ratio_20d=stock_metrics["positive_day_ratio_20d"],
                volume_trend_10d=stock_metrics["volume_trend_10d"],
                volatility_20d=stock_metrics["volatility_20d"],
                downside_volatility_20d=stock_metrics["downside_volatility_20d"],
                max_drawdown_60d=stock_metrics["max_drawdown_60d"],
                market_relative_20d=relative_metrics["market_relative_20d"],
                market_relative_60d=relative_metrics["market_relative_60d"],
                sector_relative_20d=relative_metrics["sector_relative_20d"],
                sector_relative_60d=relative_metrics["sector_relative_60d"],
            )

            stock_forward_return = safe_divide(float(future_row["Close"]), float(current_row["Close"]), 1.0) - 1.0
            market_forward_return = safe_divide(float(future_row["MarketClose"]), float(current_row["MarketClose"]), 1.0) - 1.0
            forward_excess_return = clamp(
                relative_return(stock_forward_return, market_forward_return),
                -0.35,
                0.35,
            )
            feature_vector = np.array(
                [technical_features[name] for name in MONTHLY_TECHNICAL_FEATURE_ORDER],
                dtype=float,
            )
            training_rows.append((feature_vector, forward_excess_return))

    return training_rows


def build_monthly_block_weight_sample(entry: Dict[str, Any]) -> Optional[Tuple[np.ndarray, float]]:
    diagnostics = entry.get("diagnostics")
    realized = entry.get("realized_20d_excess_return")
    if not isinstance(diagnostics, dict) or not isinstance(realized, (int, float)):
        return None

    news_signal = diagnostics.get("news_signal_input")
    macro_signal = diagnostics.get("macro_signal_input")
    sector_signal = diagnostics.get("sector_signal_input")
    if not all(isinstance(value, (int, float)) for value in (news_signal, macro_signal, sector_signal)):
        return None

    return (
        np.array([float(news_signal), float(macro_signal), float(sector_signal)], dtype=float),
        clamp(float(realized), -0.35, 0.35),
    )


def calibrate_monthly_block_weights(history_entries: List[Dict[str, Any]]) -> Tuple[Dict[str, float], int, str]:
    samples = [
        sample
        for sample in (build_monthly_block_weight_sample(entry) for entry in history_entries)
        if sample is not None
    ]
    if len(samples) < MONTHLY_BLOCK_CALIBRATION_MIN_ROWS:
        return dict(MONTHLY_BLOCK_BASE_PRIORS), len(samples), "monthly_block_priors"

    x = np.vstack([sample[0] for sample in samples])
    y = np.array([sample[1] for sample in samples], dtype=float)
    beta = np.linalg.solve(
        x.T @ x + MONTHLY_BLOCK_CALIBRATION_RIDGE_ALPHA * np.eye(x.shape[1]),
        x.T @ y,
    )
    beta = np.clip(beta, 0.0, None)
    if float(beta.sum()) <= 1e-9:
        return dict(MONTHLY_BLOCK_BASE_PRIORS), len(samples), "monthly_block_priors"

    budget = sum(MONTHLY_BLOCK_BASE_PRIORS.values())
    weights = {
        "news": clamp(
            budget * float(beta[0] / beta.sum()),
            MONTHLY_BLOCK_WEIGHT_FLOORS["news"],
            MONTHLY_BLOCK_WEIGHT_CAPS["news"],
        ),
        "macro": clamp(
            budget * float(beta[1] / beta.sum()),
            MONTHLY_BLOCK_WEIGHT_FLOORS["macro"],
            MONTHLY_BLOCK_WEIGHT_CAPS["macro"],
        ),
        "sector": clamp(
            budget * float(beta[2] / beta.sum()),
            MONTHLY_BLOCK_WEIGHT_FLOORS["sector"],
            MONTHLY_BLOCK_WEIGHT_CAPS["sector"],
        ),
    }
    blended_weights, learned_share = blend_weight_maps(
        MONTHLY_BLOCK_BASE_PRIORS,
        weights,
        sample_count=len(samples),
        min_rows=MONTHLY_BLOCK_CALIBRATION_MIN_ROWS,
        full_trust_rows=MONTHLY_BLOCK_CALIBRATION_FULL_TRUST_ROWS,
    )
    source = (
        "monthly_history_realized_returns"
        if learned_share >= 0.999
        else f"monthly_history_realized_returns_blended_{learned_share:.2f}"
    )
    return blended_weights, len(samples), source


def build_monthly_calibration(
    universe: List[Tuple[str, str]],
    sector_map: Dict[str, str],
    history_entries: List[Dict[str, Any]],
) -> ModelCalibration:
    calibration = default_monthly_calibration()
    source = calibration.source
    try:
        training_rows = build_monthly_calibration_training_rows(universe, sector_map)
    except Exception as exc:
        print(f"[WARN] Monthly technical calibration failed, using default priors: {exc}")
        training_rows = []

    if len(training_rows) < MONTHLY_CALIBRATION_MIN_ROWS:
        technical_calibration = calibration
    else:
        x = np.vstack([row[0] for row in training_rows])
        y = np.array([row[1] for row in training_rows], dtype=float)
        beta = np.linalg.solve(
            x.T @ x + MONTHLY_CALIBRATION_RIDGE_ALPHA * np.eye(x.shape[1]),
            x.T @ y,
        )
        beta = np.clip(beta, 0.0, None)
        if float(beta.sum()) <= 1e-9:
            technical_calibration = calibration
        else:
            normalized_weights = normalize_monthly_weight_map(
                {name: float(value) for name, value in zip(MONTHLY_TECHNICAL_FEATURE_ORDER, beta.tolist())}
            )
            raw_predictions = x @ np.array(
                [normalized_weights[name] for name in MONTHLY_TECHNICAL_FEATURE_ORDER],
                dtype=float,
            )
            training_ic = information_coefficient(raw_predictions, y)
            robust_scale = float(np.quantile(np.abs(raw_predictions), MONTHLY_CALIBRATION_PREDICTION_PCTL)) if len(raw_predictions) else 0.0
            technical_scale = clamp(
                MONTHLY_TECHNICAL_SCORE_TARGET_SCALE / max(robust_scale, 1e-6),
                0.16,
                0.48,
            )
            if abs(training_ic) >= MONTHLY_CALIBRATION_MIN_ABS_IC:
                technical_calibration = ModelCalibration(
                    technical_weights=normalized_weights,
                    block_weights=dict(MONTHLY_BLOCK_BASE_PRIORS),
                    technical_scale=technical_scale,
                    training_row_count=len(training_rows),
                    training_ic=training_ic,
                    block_row_count=0,
                    source="monthly_price_history_20d_excess_return",
                )
            else:
                technical_calibration = ModelCalibration(
                    technical_weights=calibration.technical_weights,
                    block_weights=dict(MONTHLY_BLOCK_BASE_PRIORS),
                    technical_scale=calibration.technical_scale,
                    training_row_count=len(training_rows),
                    training_ic=training_ic,
                    block_row_count=0,
                    source="monthly_price_history_low_ic_default_priors",
                )
            source = technical_calibration.source

    block_weights, block_row_count, block_source = calibrate_monthly_block_weights(history_entries)
    if technical_calibration.source == "monthly_priors" and block_source == "monthly_block_priors":
        return technical_calibration

    return ModelCalibration(
        technical_weights=technical_calibration.technical_weights,
        block_weights=block_weights,
        technical_scale=technical_calibration.technical_scale,
        training_row_count=technical_calibration.training_row_count,
        training_ic=technical_calibration.training_ic,
        block_row_count=block_row_count,
        source=f"{source}+{block_source}",
    )


def compute_monthly_score_breakdown(
    *,
    technical_features: Dict[str, float],
    news_score: float,
    news_confidence: float,
    macro_score: float,
    macro_confidence: float,
    sector_score: float,
    sector_confidence: float,
    calibration: ModelCalibration,
    layer_penalties: Dict[str, Any],
) -> Dict[str, float]:
    centered_news = clamp((news_score - 0.5) * 2.0, -1.0, 1.0)
    centered_macro = clamp((macro_score - 0.5) * 2.0, -1.0, 1.0)
    centered_sector = clamp((sector_score - 0.5) * 2.0, -1.0, 1.0)

    feature_weights = calibration.technical_weights
    raw_feature_contributions = {
        name: technical_features[name] * feature_weights.get(name, 0.0)
        for name in MONTHLY_TECHNICAL_FEATURE_ORDER
    }
    trend_strength = sum(
        raw_feature_contributions[name] * calibration.technical_scale
        for name in MONTHLY_TECHNICAL_GROUPS["trend_strength"]
    )
    relative_strength = sum(
        raw_feature_contributions[name] * calibration.technical_scale
        for name in MONTHLY_TECHNICAL_GROUPS["relative_strength"]
    )
    participation = sum(
        raw_feature_contributions[name] * calibration.technical_scale
        for name in MONTHLY_TECHNICAL_GROUPS["participation"]
    )
    risk_control = sum(
        raw_feature_contributions[name] * calibration.technical_scale
        for name in MONTHLY_TECHNICAL_GROUPS["risk_control"]
    )
    technical_total = trend_strength + relative_strength + participation + risk_control

    news_multiplier = calibration.block_weights["news"] * (0.42 + 0.58 * clamp(news_confidence, 0.0, 1.0))
    macro_multiplier = calibration.block_weights["macro"] * (0.34 + 0.66 * clamp(macro_confidence, 0.0, 1.0)) * float(layer_penalties["macro_penalty"])
    sector_multiplier = calibration.block_weights["sector"] * (0.32 + 0.68 * clamp(sector_confidence, 0.0, 1.0)) * float(layer_penalties["sector_penalty"])

    weighted_news = centered_news * news_multiplier
    weighted_macro = centered_macro * macro_multiplier
    weighted_sector = centered_sector * sector_multiplier

    technical_direction = clamp(technical_total / 0.34, -1.0, 1.0)
    alignment_signal = technical_direction * centered_news if abs(centered_news) >= 0.10 else 0.0
    weighted_signal_alignment = alignment_signal * (0.025 + 0.035 * clamp(news_confidence, 0.0, 1.0))
    total_score = technical_total + weighted_news + weighted_macro + weighted_sector + weighted_signal_alignment

    return {
        "technical_total": technical_total,
        "trend_strength": trend_strength,
        "relative_strength": relative_strength,
        "participation": participation,
        "risk_control": risk_control,
        "weighted_news": weighted_news,
        "weighted_macro": weighted_macro,
        "weighted_sector": weighted_sector,
        "weighted_signal_alignment": weighted_signal_alignment,
        "centered_news": centered_news,
        "centered_macro": centered_macro,
        "centered_sector": centered_sector,
        "technical_scale": calibration.technical_scale,
        "technical_training_rows": calibration.training_row_count,
        "technical_training_ic": calibration.training_ic,
        "block_training_rows": calibration.block_row_count,
        "block_weight_news": calibration.block_weights["news"],
        "block_weight_macro": calibration.block_weights["macro"],
        "block_weight_sector": calibration.block_weights["sector"],
        "macro_dedup_penalty": float(layer_penalties["macro_penalty"]),
        "sector_dedup_penalty": float(layer_penalties["sector_penalty"]),
        "news_macro_overlap": float(layer_penalties["news_macro_overlap"]),
        "news_sector_overlap": float(layer_penalties["news_sector_overlap"]),
        "macro_sector_overlap": float(layer_penalties["macro_sector_overlap"]),
        "total": total_score,
    }


def build_monthly_candidate(
    *,
    symbol: str,
    company_name: str,
    sector: str,
    news_info: Optional[dict],
    sector_scores_payload: Optional[Dict[str, Any]],
    calibration: ModelCalibration,
) -> MonthlyCandidate:
    price_series = fetch_price_series_cached(symbol, max_days=MONTHLY_PRICE_WINDOW_DAYS)
    technical_metrics = compute_monthly_metrics(price_series)
    market_metrics = compute_monthly_metrics(fetch_price_series_cached(MARKET_BENCHMARK_SYMBOL, max_days=MONTHLY_PRICE_WINDOW_DAYS))
    sector_benchmark_symbol = SECTOR_ETF_BY_SECTOR[sector]
    sector_benchmark_metrics = compute_monthly_metrics(fetch_price_series_cached(sector_benchmark_symbol, max_days=MONTHLY_PRICE_WINDOW_DAYS))
    relative_metrics = build_monthly_relative_strength_metrics(technical_metrics, market_metrics, sector_benchmark_metrics)
    technical_features = build_monthly_normalized_features(
        momentum_20d=technical_metrics["momentum_20d"],
        momentum_60d=technical_metrics["momentum_60d"],
        trend_gap_50d=technical_metrics["trend_gap_50d"],
        positive_day_ratio_20d=technical_metrics["positive_day_ratio_20d"],
        volume_trend_10d=technical_metrics["volume_trend_10d"],
        volatility_20d=technical_metrics["volatility_20d"],
        downside_volatility_20d=technical_metrics["downside_volatility_20d"],
        max_drawdown_60d=technical_metrics["max_drawdown_60d"],
        market_relative_20d=relative_metrics["market_relative_20d"],
        market_relative_60d=relative_metrics["market_relative_60d"],
        sector_relative_20d=relative_metrics["sector_relative_20d"],
        sector_relative_60d=relative_metrics["sector_relative_60d"],
    )

    news_snapshot = build_monthly_news_snapshot(news_info)
    macro_snapshot = build_monthly_macro_snapshot(symbol, company_name, sector, sector_scores_payload)
    sector_snapshot = build_monthly_sector_snapshot(sector, sector_scores_payload)
    layer_penalties = dedupe_layer_penalties(news_snapshot, macro_snapshot, sector_snapshot)
    score_breakdown = compute_monthly_score_breakdown(
        technical_features=technical_features,
        news_score=news_snapshot.news_score,
        news_confidence=news_snapshot.news_confidence,
        macro_score=macro_snapshot.macro_score,
        macro_confidence=macro_snapshot.macro_confidence,
        sector_score=sector_snapshot.sector_score,
        sector_confidence=sector_snapshot.sector_confidence,
        calibration=calibration,
        layer_penalties=layer_penalties,
    )
    confidence_score = compute_confidence_score(
        total_score=score_breakdown["total"],
        article_count=news_snapshot.article_count,
        price_observations=price_series.observations,
        news_score=news_snapshot.news_score,
        positive_day_ratio=technical_metrics["positive_day_ratio_20d"],
        max_drawdown=technical_metrics["max_drawdown_60d"],
        news_confidence=news_snapshot.news_confidence,
        effective_article_count=news_snapshot.effective_article_count,
        signal_alignment=score_breakdown["weighted_signal_alignment"],
        macro_score=macro_snapshot.macro_score,
        macro_confidence=macro_snapshot.macro_confidence,
        sector_score=sector_snapshot.sector_score,
        sector_confidence=sector_snapshot.sector_confidence,
    )

    reasons = [
        (
            f"Medium-term trend stayed constructive: 20-day momentum is {format_pct(technical_metrics['momentum_20d'])} "
            f"and 60-day momentum is {format_pct(technical_metrics['momentum_60d'])}."
        ),
        (
            f"Trend quality held up over the last month with {technical_metrics['positive_day_ratio_20d'] * 100:.0f}% positive sessions "
            f"and price trading {format_pct(technical_metrics['trend_gap_50d'])} versus its 50-day average."
        ),
        (
            f"Relative strength is {format_pct(relative_metrics['market_relative_20d'])} versus SPY over 20 days "
            f"and {format_pct(relative_metrics['sector_relative_20d'])} versus the {sector_display_name(sector)} benchmark."
        ),
        (
            f"Risk profile shows {format_pct(technical_metrics['volatility_20d'])} daily volatility and "
            f"a {format_pct(technical_metrics['max_drawdown_60d'])} drawdown over the last 60 trading days."
        ),
    ]
    if news_snapshot.article_count > 0:
        reasons.append(
            (
                f"Company news contributed a {news_snapshot.dominant_signal} overlay at {news_snapshot.news_score:.2f} "
                f"with confidence {news_snapshot.news_confidence:.2f}."
            )
        )
    else:
        reasons.append("Company-news impact was neutral because no recent relevant article set passed the filter.")

    if macro_snapshot.macro_confidence >= 0.30 and abs(macro_snapshot.macro_score - 0.5) >= 0.05:
        reasons.append(
            f"World-news catalysts leaned {macro_snapshot.direction} for {company_name} with score {macro_snapshot.macro_score:.2f}."
        )
    if sector_snapshot.sector_confidence >= 0.30 and abs(sector_snapshot.sector_score - 0.5) >= 0.05:
        reasons.append(
            f"The {sector_display_name(sector)} sector overlay leaned {sector_snapshot.direction} at {sector_snapshot.sector_score:.2f}."
        )
    if calibration.training_row_count >= MONTHLY_CALIBRATION_MIN_ROWS:
        reasons.append(
            f"Technical weights were calibrated on {calibration.training_row_count} historical symbol-month windows against 20-day excess return (IC {calibration.training_ic:.2f})."
        )
    if calibration.block_row_count >= MONTHLY_BLOCK_CALIBRATION_MIN_ROWS:
        reasons.append(
            f"News, macro, and sector block weights were calibrated on {calibration.block_row_count} realized monthly outcomes."
        )
    reasons.append(
        f"Monthly model score is {score_breakdown['total']:.3f} with {confidence_label(confidence_score)} confidence."
    )

    return MonthlyCandidate(
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        reasons=reasons,
        risk_level=classify_risk(technical_metrics["volatility_20d"]),
        total_score=score_breakdown["total"],
        confidence_score=confidence_score,
        confidence_label=confidence_label(confidence_score),
        price_as_of=price_series.latest_trading_date,
        news_as_of=news_snapshot.last_updated,
        macro_as_of=macro_snapshot.last_updated,
        sector_as_of=sector_snapshot.last_updated,
        article_count=news_snapshot.article_count,
        effective_article_count=news_snapshot.effective_article_count,
        source_count=news_snapshot.source_count,
        average_relevance=news_snapshot.average_relevance,
        momentum_20d=technical_metrics["momentum_20d"],
        momentum_60d=technical_metrics["momentum_60d"],
        volatility_20d=technical_metrics["volatility_20d"],
        downside_volatility_20d=technical_metrics["downside_volatility_20d"],
        max_drawdown_60d=technical_metrics["max_drawdown_60d"],
        trend_gap_50d=technical_metrics["trend_gap_50d"],
        positive_day_ratio_20d=technical_metrics["positive_day_ratio_20d"],
        volume_trend_10d=technical_metrics["volume_trend_10d"],
        market_relative_20d=relative_metrics["market_relative_20d"],
        market_relative_60d=relative_metrics["market_relative_60d"],
        sector_relative_20d=relative_metrics["sector_relative_20d"],
        sector_relative_60d=relative_metrics["sector_relative_60d"],
        news_score=news_snapshot.news_score,
        news_confidence=news_snapshot.news_confidence,
        macro_score=macro_snapshot.macro_score,
        macro_confidence=macro_snapshot.macro_confidence,
        sector_score=sector_snapshot.sector_score,
        sector_confidence=sector_snapshot.sector_confidence,
        raw_sentiment=news_snapshot.raw_sentiment,
        calibrated_sentiment=news_snapshot.calibrated_sentiment,
        dominant_signal=news_snapshot.dominant_signal,
        score_breakdown=score_breakdown,
        news_evidence=news_snapshot.news_evidence,
        macro_evidence=macro_snapshot.supporting_articles,
    )


def serialize_monthly_ranked_candidate_snapshot(candidate: MonthlyCandidate, rank: int) -> Dict[str, Any]:
    return {
        "rank": rank,
        "symbol": candidate.symbol,
        "company_name": candidate.company_name,
        "sector": candidate.sector,
        "risk": candidate.risk_level,
        "model_score": round(candidate.total_score, 4),
        "confidence_score": round(candidate.confidence_score, 2),
        "confidence_label": candidate.confidence_label,
        "price_as_of": candidate.price_as_of,
        "news_as_of": candidate.news_as_of,
        "article_count": candidate.article_count,
        "diagnostics": {
            "technical_total": round(candidate.score_breakdown["technical_total"], 4),
            "news_adjustment": round(candidate.score_breakdown["weighted_news"], 4),
            "macro_adjustment": round(candidate.score_breakdown["weighted_macro"], 4),
            "sector_adjustment": round(candidate.score_breakdown["weighted_sector"], 4),
            "signal_alignment": round(candidate.score_breakdown["weighted_signal_alignment"], 4),
        },
    }


def serialize_monthly_top_candidate_snapshots(
    candidates: Sequence[MonthlyCandidate],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: item.total_score, reverse=True)
    return [
        serialize_monthly_ranked_candidate_snapshot(candidate, rank)
        for rank, candidate in enumerate(ranked[:limit], start=1)
    ]


def get_monthly_candidates() -> Tuple[List[MonthlyCandidate], GenerationStats]:
    universe_path = resolve_universe_csv_path()
    universe = load_universe(universe_path)
    news_scores = load_news_scores()
    sector_scores = load_sector_scores()
    sector_map = load_sector_map(universe_path)
    history_entries = [
        normalize_monthly_history_entry(entry)
        for entry in load_existing_monthly_history_entries(MONTHLY_HISTORY_PATH)
    ]
    calibration = build_monthly_calibration(universe, sector_map, history_entries)
    degraded_reasons = []
    degraded_reasons.extend(extract_payload_degraded_reasons(news_scores, "news_scores"))
    degraded_reasons.extend(extract_payload_degraded_reasons(sector_scores, "sector_scores"))

    candidates: List[MonthlyCandidate] = []
    skipped_details: List[Dict[str, str]] = []
    price_provider_failures = 0
    for symbol, company_name in universe:
        try:
            candidates.append(
                build_monthly_candidate(
                    symbol=symbol,
                    company_name=company_name,
                    sector=sector_map[symbol],
                    news_info=news_scores.get(symbol),
                    sector_scores_payload=sector_scores,
                    calibration=calibration,
                )
            )
        except Exception as exc:
            print(f"[WARN] Skipping monthly candidate {symbol}: {exc}")
            reason = str(exc)
            failure_category = "price_provider_failure" if is_live_price_provider_failure(reason) else "other"
            if failure_category == "price_provider_failure":
                price_provider_failures += 1
            skipped_details.append({"symbol": symbol, "reason": reason, "category": failure_category})

    if skipped_details:
        degraded_reasons.append(
            f"{len(skipped_details)} symbols were skipped during monthly candidate generation."
        )
    if price_provider_failures:
        degraded_reasons.append(
            f"Live price data failed for {price_provider_failures} symbols during monthly candidate generation."
        )
    if universe and not candidates and price_provider_failures >= len(universe):
        degraded_reasons.append(
            "All live price providers failed across the monthly universe, so the pipeline published no_pick instead of ranking from fabricated prices."
        )

    return candidates, GenerationStats(
        universe_size=len(universe),
        evaluated_candidates=len(candidates),
        skipped_symbols=len(skipped_details),
        skipped_details=skipped_details[:10],
        price_provider_failures=price_provider_failures,
        degraded_reasons=degraded_reasons,
        model_calibration=None,
        top_candidates=serialize_monthly_top_candidate_snapshots(candidates),
    )


def qualifies(candidate: MonthlyCandidate) -> bool:
    return (
        candidate.total_score >= MONTHLY_SELECTION_THRESHOLD
        and candidate.confidence_score >= MONTHLY_MIN_CONFIDENCE_THRESHOLD
    )


def select_monthly_candidate(
    candidates: List[MonthlyCandidate],
    stats: Optional[GenerationStats] = None,
) -> MonthlySelectionDecision:
    ranked = sorted(candidates, key=lambda item: item.total_score, reverse=True)
    qualified = [candidate for candidate in ranked if qualifies(candidate)]
    if qualified:
        winner = qualified[0]
        return MonthlySelectionDecision(
            status="picked",
            status_reason=(
                f"{winner.symbol} cleared the monthly release thresholds for {winner.company_name} "
                f"with a score of {winner.total_score:.3f} and {winner.confidence_label} confidence."
            ),
            threshold_score=MONTHLY_SELECTION_THRESHOLD,
            threshold_confidence=MONTHLY_MIN_CONFIDENCE_THRESHOLD,
            pick=winner,
            best_candidate=winner,
        )

    best_candidate = ranked[0] if ranked else None
    policy_reason = price_failure_no_pick_reason("monthly", stats)
    if policy_reason:
        reason = policy_reason
    elif best_candidate is None:
        reason = "No monthly candidate had enough clean data to evaluate."
    elif best_candidate.total_score < MONTHLY_SELECTION_THRESHOLD:
        reason = (
            f"No monthly candidate cleared the minimum score threshold of {MONTHLY_SELECTION_THRESHOLD:.2f}. "
            f"The best candidate was {best_candidate.symbol} at {best_candidate.total_score:.3f}."
        )
    else:
        reason = (
            f"No monthly candidate cleared the minimum confidence threshold of "
            f"{MONTHLY_MIN_CONFIDENCE_THRESHOLD:.2f}. The best score was {best_candidate.symbol} at "
            f"{best_candidate.total_score:.3f} with {best_candidate.confidence_label} confidence."
        )

    return MonthlySelectionDecision(
        status="no_pick",
        status_reason=reason,
        threshold_score=MONTHLY_SELECTION_THRESHOLD,
        threshold_confidence=MONTHLY_MIN_CONFIDENCE_THRESHOLD,
        pick=None,
        best_candidate=best_candidate,
    )


def serialize_monthly_candidate(candidate: MonthlyCandidate) -> Dict[str, Any]:
    return {
        "symbol": candidate.symbol,
        "company_name": candidate.company_name,
        "sector": candidate.sector,
        "risk": candidate.risk_level,
        "model_score": round(candidate.total_score, 4),
        "confidence_score": round(candidate.confidence_score, 2),
        "confidence_label": candidate.confidence_label,
        "price_as_of": candidate.price_as_of,
        "news_as_of": candidate.news_as_of,
        "macro_as_of": candidate.macro_as_of,
        "sector_as_of": candidate.sector_as_of,
        "article_count": candidate.article_count,
        "metrics": {
            "momentum_20d": round(candidate.momentum_20d, 4),
            "momentum_60d": round(candidate.momentum_60d, 4),
            "daily_volatility": round(candidate.volatility_20d, 4),
            "downside_volatility": round(candidate.downside_volatility_20d, 4),
            "max_drawdown_60d": round(candidate.max_drawdown_60d, 4),
            "price_vs_50d_average": round(candidate.trend_gap_50d, 4),
            "positive_day_ratio_20d": round(candidate.positive_day_ratio_20d, 4),
            "volume_trend_10d": round(candidate.volume_trend_10d, 4),
            "market_relative_20d": round(candidate.market_relative_20d, 4),
            "market_relative_60d": round(candidate.market_relative_60d, 4),
            "sector_relative_20d": round(candidate.sector_relative_20d, 4),
            "sector_relative_60d": round(candidate.sector_relative_60d, 4),
            "news_sentiment": round(candidate.news_score, 2),
            "raw_news_sentiment": round(candidate.raw_sentiment, 4),
            "calibrated_news_sentiment": round(candidate.calibrated_sentiment, 4),
            "news_confidence": round(candidate.news_confidence, 2),
            "macro_sentiment": round(candidate.macro_score, 2),
            "macro_confidence": round(candidate.macro_confidence, 2),
            "sector_sentiment": round(candidate.sector_score, 2),
            "sector_confidence": round(candidate.sector_confidence, 2),
            "effective_article_count": round(candidate.effective_article_count, 2),
            "source_count": candidate.source_count,
            "average_relevance": round(candidate.average_relevance, 2),
            "technical_training_ic": round(float(candidate.score_breakdown.get("technical_training_ic", 0.0)), 4),
            "technical_training_rows": int(candidate.score_breakdown.get("technical_training_rows", 0)),
            "block_training_rows": int(candidate.score_breakdown.get("block_training_rows", 0)),
        },
        "score_breakdown": {
            "trend_strength": round(candidate.score_breakdown["trend_strength"], 4),
            "relative_strength": round(candidate.score_breakdown["relative_strength"], 4),
            "participation": round(candidate.score_breakdown["participation"], 4),
            "risk_control": round(candidate.score_breakdown["risk_control"], 4),
            "technical_total": round(candidate.score_breakdown["technical_total"], 4),
            "news_adjustment": round(candidate.score_breakdown["weighted_news"], 4),
            "macro_adjustment": round(candidate.score_breakdown["weighted_macro"], 4),
            "sector_adjustment": round(candidate.score_breakdown["weighted_sector"], 4),
            "signal_alignment": round(candidate.score_breakdown["weighted_signal_alignment"], 4),
            "block_weight_news": round(candidate.score_breakdown.get("block_weight_news", 0.0), 4),
            "block_weight_macro": round(candidate.score_breakdown.get("block_weight_macro", 0.0), 4),
            "block_weight_sector": round(candidate.score_breakdown.get("block_weight_sector", 0.0), 4),
            "macro_dedup_penalty": round(candidate.score_breakdown.get("macro_dedup_penalty", 1.0), 4),
            "sector_dedup_penalty": round(candidate.score_breakdown.get("sector_dedup_penalty", 1.0), 4),
            "total": round(candidate.score_breakdown["total"], 4),
        },
        "reasons": candidate.reasons,
        "news_evidence": candidate.news_evidence,
        "macro_evidence": candidate.macro_evidence,
    }


def serialize_monthly_selection(selection: MonthlySelectionDecision) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": selection.status,
        "status_reason": selection.status_reason,
        "threshold_score": round(selection.threshold_score, 2),
        "threshold_confidence": round(selection.threshold_confidence, 2),
        "pick": serialize_monthly_candidate(selection.pick) if selection.pick else None,
    }
    if selection.status == "no_pick" and selection.best_candidate is not None:
        payload["best_candidate"] = {
            "symbol": selection.best_candidate.symbol,
            "company_name": selection.best_candidate.company_name,
            "risk": selection.best_candidate.risk_level,
            "model_score": round(selection.best_candidate.total_score, 4),
            "confidence_score": round(selection.best_candidate.confidence_score, 2),
            "confidence_label": selection.best_candidate.confidence_label,
        }
    return payload


def derive_monthly_data_as_of(candidates: List[MonthlyCandidate]) -> str:
    candidate_dates = [
        value
        for candidate in candidates
        for value in [
            candidate.price_as_of,
            candidate.news_as_of,
            candidate.macro_as_of,
            candidate.sector_as_of,
        ]
        if value
    ]
    return max(candidate_dates) if candidate_dates else date.today().isoformat()


def monthly_common_payload(
    *,
    generated_at: str,
    data_as_of: str,
    expected_next_refresh_at: str,
    stale_after: str,
    market_month: MarketMonth,
    stats: GenerationStats,
    data_quality: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MONTHLY_MODEL_VERSION,
        "generated_at": generated_at,
        "data_as_of": data_as_of,
        "expected_next_refresh_at": expected_next_refresh_at,
        "stale_after": stale_after,
        "period_context": {
            "timezone": MARKET_TIMEZONE,
            "month_id": market_month.month_id,
            "month_label": market_month.month_label,
            "month_start": market_month.month_start.isoformat(),
            "month_end": market_month.month_end.isoformat(),
            "rebalance_date": market_month.rebalance_date.isoformat(),
            "horizon_trading_days": market_month.horizon_trading_days,
        },
        "generation_summary": {
            "universe_size": stats.universe_size,
            "evaluated_candidates": stats.evaluated_candidates,
            "skipped_symbols": stats.skipped_symbols,
            "price_provider_failures": stats.price_provider_failures,
            "skipped_details": stats.skipped_details,
            "top_candidates": stats.top_candidates,
        },
        "selection_thresholds": {
            "overall_score": MONTHLY_SELECTION_THRESHOLD,
            "minimum_confidence": MONTHLY_MIN_CONFIDENCE_THRESHOLD,
        },
        "data_quality": data_quality,
    }


def build_monthly_pick_payload(
    *,
    selection: MonthlySelectionDecision,
    generated_at: str,
    data_as_of: str,
    expected_next_refresh_at: str,
    stale_after: str,
    market_month: MarketMonth,
    stats: GenerationStats,
    data_quality: Dict[str, Any],
) -> Dict[str, Any]:
    payload = monthly_common_payload(
        generated_at=generated_at,
        data_as_of=data_as_of,
        expected_next_refresh_at=expected_next_refresh_at,
        stale_after=stale_after,
        market_month=market_month,
        stats=stats,
        data_quality=data_quality,
    )
    payload["selection"] = serialize_monthly_selection(selection)
    return payload


def load_existing_monthly_history_entries(history_path: Path) -> List[Dict[str, Any]]:
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return []

    if isinstance(raw, dict):
        entries = raw.get("entries", [])
        return entries if isinstance(entries, list) else []
    if isinstance(raw, list):
        return raw
    return []


def normalize_monthly_history_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    month_start = entry.get("month_start") or date.today().replace(day=1).isoformat()
    month_id = entry.get("month_id") or month_start[:7]
    return {
        "month_id": month_id,
        "month_start": month_start,
        "month_end": entry.get("month_end") or month_start,
        "rebalance_date": entry.get("rebalance_date") or month_start,
        "month_label": entry.get("month_label") or month_id,
        "logged_at": entry.get("logged_at") or month_start,
        "status": entry.get("status") or ("picked" if entry.get("symbol") else "no_pick"),
        "status_reason": entry.get("status_reason") or "",
        "symbol": entry.get("symbol"),
        "company_name": entry.get("company_name"),
        "sector": entry.get("sector"),
        "risk": entry.get("risk"),
        "model_score": entry.get("model_score"),
        "confidence_score": entry.get("confidence_score"),
        "confidence_label": entry.get("confidence_label"),
        "data_as_of": entry.get("data_as_of"),
        "model_version": entry.get("model_version", MONTHLY_MODEL_VERSION),
        "realized_20d_return": entry.get("realized_20d_return"),
        "realized_20d_excess_return": entry.get("realized_20d_excess_return"),
        "top_candidates": entry.get("top_candidates") if isinstance(entry.get("top_candidates"), list) else [],
        "diagnostics": entry.get("diagnostics") if isinstance(entry.get("diagnostics"), dict) else None,
    }


def build_monthly_history_entry(
    market_month: MarketMonth,
    selection: MonthlySelectionDecision,
    generated_at: str,
    data_as_of: str,
    top_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pick = selection.pick
    return {
        "month_id": market_month.month_id,
        "month_start": market_month.month_start.isoformat(),
        "month_end": market_month.month_end.isoformat(),
        "rebalance_date": market_month.rebalance_date.isoformat(),
        "month_label": market_month.month_label,
        "logged_at": generated_at,
        "status": selection.status,
        "status_reason": selection.status_reason,
        "symbol": pick.symbol if pick else None,
        "company_name": pick.company_name if pick else None,
        "sector": pick.sector if pick else None,
        "risk": pick.risk_level if pick else None,
        "model_score": round(pick.total_score, 4) if pick else None,
        "confidence_score": round(pick.confidence_score, 2) if pick else None,
        "confidence_label": pick.confidence_label if pick else None,
        "data_as_of": data_as_of,
        "model_version": MONTHLY_MODEL_VERSION,
        "top_candidates": top_candidates or [],
        "diagnostics": (
            {
                "technical_total": round(pick.score_breakdown["technical_total"], 4),
                "news_adjustment": round(pick.score_breakdown["weighted_news"], 4),
                "macro_adjustment": round(pick.score_breakdown["weighted_macro"], 4),
                "sector_adjustment": round(pick.score_breakdown["weighted_sector"], 4),
                "news_signal_input": round(float(pick.score_breakdown.get("centered_news", 0.0)) * pick.news_confidence, 4),
                "macro_signal_input": round(
                    float(pick.score_breakdown.get("centered_macro", 0.0))
                    * pick.macro_confidence
                    * pick.score_breakdown.get("macro_dedup_penalty", 1.0),
                    4,
                ),
                "sector_signal_input": round(
                    float(pick.score_breakdown.get("centered_sector", 0.0))
                    * pick.sector_confidence
                    * pick.score_breakdown.get("sector_dedup_penalty", 1.0),
                    4,
                ),
                "technical_training_rows": int(pick.score_breakdown.get("technical_training_rows", 0)),
                "technical_training_ic": round(float(pick.score_breakdown.get("technical_training_ic", 0.0)), 4),
                "block_training_rows": int(pick.score_breakdown.get("block_training_rows", 0)),
                "news_macro_overlap": round(float(pick.score_breakdown.get("news_macro_overlap", 0.0)), 4),
                "news_sector_overlap": round(float(pick.score_breakdown.get("news_sector_overlap", 0.0)), 4),
                "macro_sector_overlap": round(float(pick.score_breakdown.get("macro_sector_overlap", 0.0)), 4),
            }
            if pick
            else None
        ),
    }


def realized_forward_return_monthly(
    symbol: str,
    anchor_date: str,
    benchmark_symbol: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    try:
        price_frame = fetch_price_frame_cached(symbol)
    except Exception:
        return None, None

    anchor_ts = pd.Timestamp(anchor_date)
    eligible = price_frame[price_frame["Date"] <= anchor_ts]
    if eligible.empty:
        return None, None
    anchor_index = int(eligible.index[-1])
    future_index = anchor_index + MONTHLY_HORIZON_TRADING_DAYS
    if future_index >= len(price_frame):
        return None, None

    start_price = float(price_frame.iloc[anchor_index]["Close"])
    end_price = float(price_frame.iloc[future_index]["Close"])
    stock_return = safe_divide(end_price, start_price, 1.0) - 1.0

    if not benchmark_symbol:
        return stock_return, None

    try:
        benchmark_frame = fetch_price_frame_cached(benchmark_symbol)
    except Exception:
        return stock_return, None

    benchmark_eligible = benchmark_frame[benchmark_frame["Date"] <= anchor_ts]
    if benchmark_eligible.empty:
        return stock_return, None
    benchmark_index = int(benchmark_eligible.index[-1])
    benchmark_future_index = benchmark_index + MONTHLY_HORIZON_TRADING_DAYS
    if benchmark_future_index >= len(benchmark_frame):
        return stock_return, None

    benchmark_start = float(benchmark_frame.iloc[benchmark_index]["Close"])
    benchmark_end = float(benchmark_frame.iloc[benchmark_future_index]["Close"])
    benchmark_return = safe_divide(benchmark_end, benchmark_start, 1.0) - 1.0
    return stock_return, relative_return(stock_return, benchmark_return)


def enrich_monthly_history_realized_returns(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not monthly_history_realized_enrichment_enabled():
        return entries

    enriched: List[Dict[str, Any]] = []
    for entry in entries:
        normalized = dict(entry)
        symbol = normalized.get("symbol")
        rebalance_date = normalized.get("rebalance_date") or normalized.get("data_as_of")
        if isinstance(symbol, str) and isinstance(rebalance_date, str):
            realized_return, realized_excess = realized_forward_return_monthly(
                symbol,
                rebalance_date,
                MARKET_BENCHMARK_SYMBOL,
            )
            if realized_return is not None:
                normalized["realized_20d_return"] = round(realized_return, 4)
            if realized_excess is not None:
                normalized["realized_20d_excess_return"] = round(realized_excess, 4)
        enriched.append(normalized)
    return enriched


def update_monthly_history(
    market_month: MarketMonth,
    selection: MonthlySelectionDecision,
    generated_at: str,
    data_as_of: str,
    data_quality: Dict[str, Any],
    top_candidates: Optional[List[Dict[str, Any]]] = None,
    history_path: Path = MONTHLY_HISTORY_PATH,
) -> Dict[str, Any]:
    entries = [
        normalize_monthly_history_entry(entry)
        for entry in load_existing_monthly_history_entries(history_path)
    ]
    new_entry = build_monthly_history_entry(
        market_month,
        selection,
        generated_at,
        data_as_of,
        top_candidates=top_candidates,
    )

    filtered_entries = [entry for entry in entries if entry.get("month_id") != market_month.month_id]
    filtered_entries.append(new_entry)
    filtered_entries.sort(key=lambda item: item.get("month_start", ""))
    filtered_entries = filtered_entries[-60:]
    filtered_entries = enrich_monthly_history_realized_returns(filtered_entries)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MONTHLY_MODEL_VERSION,
        "generated_at": generated_at,
        "entries": filtered_entries,
        "data_quality": data_quality,
    }
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    set_pipeline_scope("monthly_pick")
    print("[INFO] Generating monthly stock pick...")
    PRICE_SERIES_CACHE.clear()
    PRICE_FRAME_CACHE.clear()
    generated_at = iso_utc(now_utc())
    market_month = build_market_month()

    expected_next_refresh_at, stale_after = build_monthly_refresh_window(market_month)
    candidates, stats = get_monthly_candidates()
    selection = select_monthly_candidate(candidates, stats)
    data_as_of = derive_monthly_data_as_of(candidates)
    data_quality = build_data_quality_block(
        scope="monthly_pick",
        extra_reasons=stats.degraded_reasons,
    )

    payload = build_monthly_pick_payload(
        selection=selection,
        generated_at=generated_at,
        data_as_of=data_as_of,
        expected_next_refresh_at=expected_next_refresh_at,
        stale_after=stale_after,
        market_month=market_month,
        stats=stats,
        data_quality=data_quality,
    )
    write_json(MONTHLY_PICK_PATH, payload)
    update_monthly_history(
        market_month=market_month,
        selection=selection,
        generated_at=generated_at,
        data_as_of=data_as_of,
        data_quality=data_quality,
        top_candidates=stats.top_candidates,
    )
    print("[INFO] monthly_pick.json and monthly_history.json updated.")


if __name__ == "__main__":
    main()
