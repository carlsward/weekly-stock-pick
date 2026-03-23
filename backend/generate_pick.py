import json
import math
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

MODEL_VERSION = "v3.1"
SCHEMA_VERSION = 2
MARKET_TIMEZONE = "America/New_York"
MARKET_TZ = ZoneInfo(MARKET_TIMEZONE)

UNIVERSE_CSV_PATH = Path("universe.csv")
NEWS_SCORES_PATH = Path("news_scores.json")
CURRENT_PICK_PATH = Path("current_pick.json")
RISK_PICKS_PATH = Path("risk_picks.json")
HISTORY_PATH = Path("history.json")

SHORT_MOMENTUM_SCALE = 0.08
MEDIUM_MOMENTUM_SCALE = 0.18
TREND_GAP_SCALE = 0.08
VOLUME_TREND_SCALE = 0.60
VOLATILITY_SCALE = 0.04
DOWNSIDE_VOLATILITY_SCALE = 0.025
MAX_DRAWDOWN_SCALE = 0.12

OVERALL_SELECTION_THRESHOLD = 0.10
RISK_SELECTION_THRESHOLDS = {
    "low": 0.08,
    "medium": 0.10,
    "high": 0.12,
}
MIN_CONFIDENCE_THRESHOLD = 0.55
EXPECTED_REFRESH_HOUR_UTC = 12
STALE_GRACE_HOURS = 12
NEWS_STALE_AFTER_DAYS = 2
PRICE_FETCH_ATTEMPTS = 3
PRICE_FETCH_RETRY_SECONDS = 1.5
PRICE_REQUEST_TIMEOUT_SECONDS = 20
SIGNAL_ALIGNMENT_BASE = 0.03
SIGNAL_ALIGNMENT_CONFIDENCE_SCALE = 0.04
STOOQ_DAILY_ENDPOINT = "https://stooq.com/q/d/l/"


@dataclass(frozen=True)
class MarketWeek:
    week_id: str
    week_start: date
    week_end: date
    week_label: str


@dataclass(frozen=True)
class PriceSeries:
    closes: List[float]
    volumes: List[float]
    latest_trading_date: str
    observations: int


@dataclass(frozen=True)
class NewsSnapshot:
    news_score: float
    reasons: List[str]
    raw_sentiment: float
    calibrated_sentiment: float
    article_count: int
    effective_article_count: float
    source_count: int
    average_relevance: float
    news_confidence: float
    dominant_signal: str
    last_updated: Optional[str]
    news_evidence: List[Dict[str, Any]]


@dataclass
class StockCandidate:
    symbol: str
    company_name: str
    reasons: List[str]
    risk_level: str
    total_score: float
    confidence_score: float
    confidence_label: str
    price_as_of: str
    news_as_of: Optional[str]
    article_count: int
    effective_article_count: float
    source_count: int
    average_relevance: float
    momentum_5d: float
    momentum_20d: float
    volatility: float
    downside_volatility: float
    max_drawdown: float
    trend_gap: float
    positive_day_ratio: float
    volume_trend: float
    news_score: float
    news_confidence: float
    raw_sentiment: float
    calibrated_sentiment: float
    dominant_signal: str
    score_breakdown: Dict[str, float]
    news_evidence: List[Dict[str, Any]]


@dataclass(frozen=True)
class SelectionDecision:
    status: str
    status_reason: str
    threshold_score: float
    threshold_confidence: float
    pick: Optional[StockCandidate]
    best_candidate: Optional[StockCandidate]


@dataclass(frozen=True)
class GenerationStats:
    universe_size: int
    evaluated_candidates: int
    skipped_symbols: int
    skipped_details: List[Dict[str, str]]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_score(value: float) -> str:
    return f"{value:+.3f}"


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def market_today(now: Optional[datetime] = None) -> date:
    return (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ).date()


def market_day_age(as_of: date, reference: date) -> int:
    if as_of >= reference:
        return 0

    age = 0
    cursor = as_of
    while cursor < reference:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            age += 1
    return age


def format_market_age(label: str, days: Optional[int]) -> str:
    if days is None:
        return f"{label} no fresh news"
    suffix = "trading day" if days == 1 else "trading days"
    return f"{label} {days} {suffix}"


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None

    try:
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(normalized)
        except ValueError:
            return None
        return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)


def build_market_week(now: Optional[datetime] = None) -> MarketWeek:
    market_now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    reference_date = market_now.date()
    if reference_date.weekday() >= 5:
        reference_date = reference_date + timedelta(days=7 - reference_date.weekday())

    week_start = reference_date - timedelta(days=reference_date.weekday())
    week_end = week_start + timedelta(days=4)
    iso = week_start.isocalendar()
    week_id = f"{iso.year}-W{iso.week:02d}"

    if week_start.month == week_end.month:
        week_label = f"{week_start.strftime('%b %d')} - {week_end.strftime('%d, %Y')}"
    else:
        week_label = f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"

    return MarketWeek(
        week_id=week_id,
        week_start=week_start,
        week_end=week_end,
        week_label=week_label,
    )


def build_refresh_window(market_week: MarketWeek) -> Tuple[str, str]:
    next_refresh = datetime.combine(
        market_week.week_start + timedelta(days=7),
        time(hour=EXPECTED_REFRESH_HOUR_UTC, tzinfo=timezone.utc),
    )
    stale_after = next_refresh + timedelta(hours=STALE_GRACE_HOURS)
    return iso_utc(next_refresh), iso_utc(stale_after)


def load_universe(path: Path = UNIVERSE_CSV_PATH) -> List[Tuple[str, str]]:
    df = pd.read_csv(path)
    df = df[df.get("active", 1) == 1]
    df = df.dropna(subset=["symbol", "company_name"])

    rows: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        company_name = str(row["company_name"]).strip()
        if symbol and company_name:
            rows.append((symbol, company_name))

    if not rows:
        raise RuntimeError("No active symbols found in universe.csv")

    return rows


def stooq_symbol(symbol: str) -> str:
    normalized = symbol.strip().replace(".", "-").lower()
    return normalized if normalized.endswith(".us") else f"{normalized}.us"


def build_price_series_from_frame(frame: pd.DataFrame, symbol: str, max_days: int = 45) -> PriceSeries:
    if frame is None or frame.empty:
        raise RuntimeError(f"Price provider returned an empty frame for {symbol}")

    working = frame.copy()
    lower_columns = {column.lower(): column for column in working.columns}
    rename_map = {}
    if "date" in lower_columns:
        rename_map[lower_columns["date"]] = "Date"
    if "close" in lower_columns:
        rename_map[lower_columns["close"]] = "Close"
    if "volume" in lower_columns:
        rename_map[lower_columns["volume"]] = "Volume"
    working = working.rename(columns=rename_map)

    if "Date" not in working.columns:
        if isinstance(working.index, pd.DatetimeIndex):
            working = working.reset_index().rename(columns={working.index.name or "index": "Date"})
        else:
            raise RuntimeError(f"Price data for {symbol} did not include trading dates")
    if "Close" not in working.columns:
        raise RuntimeError(f"Price data for {symbol} did not include Close prices")

    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working["Close"] = pd.to_numeric(working["Close"], errors="coerce")
    if "Volume" in working.columns:
        working["Volume"] = pd.to_numeric(working["Volume"], errors="coerce").fillna(0.0)
    else:
        working["Volume"] = 0.0

    working = (
        working.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(working) < 21:
        raise RuntimeError(f"Not enough price history for {symbol}")

    recent_frame = working.tail(max_days)
    latest_date = recent_frame["Date"].iloc[-1].date().isoformat()

    return PriceSeries(
        closes=list(recent_frame["Close"].tolist())[::-1],
        volumes=list(recent_frame["Volume"].tolist())[::-1],
        latest_trading_date=latest_date,
        observations=len(recent_frame),
    )


def fetch_stooq_price_frame(symbol: str) -> pd.DataFrame:
    params = {
        "s": stooq_symbol(symbol),
        "i": "d",
    }
    url = f"{STOOQ_DAILY_ENDPOINT}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "Accept": "text/csv",
            "User-Agent": "weekly-stock-pick/1.0",
        },
    )

    try:
        with urlopen(request, timeout=PRICE_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
    except HTTPError as exc:
        raise RuntimeError(f"Stooq HTTP {exc.code} for {symbol}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Stooq request failed for {symbol}: {exc}") from exc

    if not body or "No data" in body or "404 Not Found" in body:
        raise RuntimeError(f"Stooq returned no usable price data for {symbol}")

    frame = pd.read_csv(StringIO(body))
    if frame.empty:
        raise RuntimeError(f"Stooq returned an empty CSV for {symbol}")
    return frame


def fetch_yahoo_price_frame(symbol: str) -> pd.DataFrame:
    data = yf.download(
        tickers=symbol,
        period="3mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise RuntimeError("empty Yahoo price response")
    if "Close" not in data:
        raise RuntimeError(f"Yahoo data for {symbol} did not include Close prices")

    closes = data["Close"]
    volumes = data["Volume"] if "Volume" in data else None

    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.iloc[:, 0]

    frame = pd.DataFrame(
        {
            "Date": closes.index,
            "Close": closes.values,
            "Volume": volumes.values if volumes is not None else [0.0] * len(closes),
        }
    )
    return frame


def fetch_price_series(symbol: str, max_days: int = 45) -> PriceSeries:
    providers = [
        ("stooq", fetch_stooq_price_frame),
        ("yahoo", fetch_yahoo_price_frame),
    ]
    failures: List[str] = []

    for provider_name, provider_fetcher in providers:
        provider_error: Optional[Exception] = None
        for attempt in range(1, PRICE_FETCH_ATTEMPTS + 1):
            try:
                frame = provider_fetcher(symbol)
                return build_price_series_from_frame(frame, symbol, max_days)
            except Exception as exc:
                provider_error = exc
                if attempt < PRICE_FETCH_ATTEMPTS:
                    time_module.sleep(PRICE_FETCH_RETRY_SECONDS * attempt)

        if provider_error is not None:
            failures.append(f"{provider_name}: {provider_error}")

    raise RuntimeError(
        f"Unable to fetch usable price history for {symbol} after {PRICE_FETCH_ATTEMPTS} attempts per provider: "
        + "; ".join(failures)
    )


def load_news_scores(path: Path = NEWS_SCORES_PATH) -> Dict[str, dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                if "FB" in data and "META" not in data:
                    data["META"] = data["FB"]
                if "META" in data and "FB" not in data:
                    data["FB"] = data["META"]
                return data
    except FileNotFoundError:
        print(f"[WARN] Could not find {path}; news contribution will be neutral.")

    return {}


def normalize_news_evidence(news_info: dict) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    top_articles = news_info.get("top_articles")
    if isinstance(top_articles, list):
        for article in top_articles[:5]:
            if not isinstance(article, dict):
                continue
            title = str(article.get("title", "")).strip()
            if not title:
                continue
            normalized.append(
                {
                    "title": title,
                    "provider": str(article.get("provider", "")).strip() or None,
                    "url": str(article.get("url", "")).strip() or None,
                    "published_at": str(article.get("published_at", "")).strip() or None,
                    "relevance_score": float(article["relevance_score"]) if isinstance(article.get("relevance_score"), (int, float)) else None,
                    "sentiment": float(article["sentiment"]) if isinstance(article.get("sentiment"), (int, float)) else None,
                }
            )

    if normalized:
        return normalized

    top_headlines = news_info.get("top_headlines")
    if isinstance(top_headlines, list):
        for title in top_headlines[:5]:
            normalized_title = str(title).strip()
            if normalized_title:
                normalized.append(
                    {
                        "title": normalized_title,
                        "provider": None,
                        "url": None,
                        "published_at": None,
                        "relevance_score": None,
                        "sentiment": None,
                    }
                )

    return normalized


def build_news_snapshot(news_info: Optional[dict]) -> NewsSnapshot:
    if not news_info:
        return NewsSnapshot(
            news_score=0.5,
            reasons=[
                "No recent relevant news was available, so the news component stayed neutral."
            ],
            raw_sentiment=0.0,
            calibrated_sentiment=0.0,
            article_count=0,
            effective_article_count=0.0,
            source_count=0,
            average_relevance=0.0,
            news_confidence=0.20,
            dominant_signal="neutral",
            last_updated=None,
            news_evidence=[],
        )

    reasons = news_info.get("news_reasons")
    normalized_reasons = [str(item) for item in reasons] if isinstance(reasons, list) else []
    raw_sentiment = float(news_info.get("raw_sentiment", 0.0))
    base_calibrated_sentiment = float(news_info.get("calibrated_sentiment", raw_sentiment))
    article_count = int(news_info.get("article_count", 0))
    base_news_confidence = float(news_info.get("news_confidence", 0.25 if article_count else 0.20))
    base_news_score = float(news_info.get("news_score", 0.5))
    last_updated = news_info.get("last_updated")
    now = now_utc()
    freshness_multiplier = 1.0

    parsed_last_updated = parse_iso_datetime(last_updated)
    if parsed_last_updated and parsed_last_updated < now - timedelta(days=NEWS_STALE_AFTER_DAYS):
        age_days = max(1, (now - parsed_last_updated).days)
        freshness_multiplier = 0.55
        normalized_reasons.append(
            f"News evidence is {age_days} days old, so its sentiment impact and confidence were reduced."
        )

    calibrated_sentiment = base_calibrated_sentiment * freshness_multiplier
    adjusted_news_score = 0.5 + (base_news_score - 0.5) * freshness_multiplier
    adjusted_news_confidence = clamp(
        base_news_confidence * freshness_multiplier,
        0.15 if article_count else 0.20,
        0.95,
    )
    dominant_signal = str(news_info.get("dominant_signal", "neutral"))
    if freshness_multiplier < 1.0 and abs(calibrated_sentiment) < 0.18:
        dominant_signal = "neutral"

    news_evidence = normalize_news_evidence(news_info)

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
        news_evidence=news_evidence,
    )


def daily_returns(closes: List[float]) -> List[float]:
    if len(closes) < 2:
        return []
    return [safe_divide(closes[index], closes[index + 1], 1.0) - 1.0 for index in range(len(closes) - 1)]


def std_dev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((item - mean_value) ** 2 for item in values) / len(values)
    return math.sqrt(variance)


def compute_max_drawdown(closes: List[float]) -> float:
    if not closes:
        return 0.0

    chronological = list(reversed(closes))
    peak = chronological[0]
    worst_drawdown = 0.0
    for price in chronological[1:]:
        peak = max(peak, price)
        drawdown = safe_divide(price, peak, 1.0) - 1.0
        worst_drawdown = min(worst_drawdown, drawdown)

    return abs(worst_drawdown)


def compute_technical_metrics(price_series: PriceSeries) -> Dict[str, float]:
    closes = price_series.closes
    volumes = price_series.volumes
    returns = daily_returns(closes)

    if len(closes) < 21 or len(returns) < 20:
        raise ValueError("At least 21 daily closes are required")

    momentum_5d = safe_divide(closes[0], closes[5], 1.0) - 1.0
    momentum_20d = safe_divide(closes[0], closes[20], 1.0) - 1.0
    volatility = std_dev(returns[:20])

    downside_returns = [item for item in returns[:20] if item < 0]
    downside_volatility = math.sqrt(sum(item * item for item in downside_returns) / len(downside_returns)) if downside_returns else 0.0

    average_recent_20 = sum(closes[:20]) / 20
    trend_gap = safe_divide(closes[0], average_recent_20, 1.0) - 1.0
    positive_day_ratio = sum(1 for item in returns[:10] if item > 0) / 10
    max_drawdown = compute_max_drawdown(closes[:21])

    recent_volume = sum(volumes[:5]) / 5 if len(volumes) >= 5 else 0.0
    earlier_volume_window = volumes[5:20]
    baseline_volume = sum(earlier_volume_window) / len(earlier_volume_window) if earlier_volume_window else recent_volume
    volume_trend = safe_divide(recent_volume, baseline_volume, 1.0) - 1.0 if baseline_volume > 0 else 0.0

    return {
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "volatility": volatility,
        "downside_volatility": downside_volatility,
        "trend_gap": trend_gap,
        "positive_day_ratio": positive_day_ratio,
        "max_drawdown": max_drawdown,
        "volume_trend": volume_trend,
    }


def compute_metrics(closes: List[float]) -> Tuple[float, float]:
    metrics = compute_technical_metrics(
        PriceSeries(
            closes=closes,
            volumes=[0.0] * len(closes),
            latest_trading_date=date.today().isoformat(),
            observations=len(closes),
        )
    )
    return metrics["momentum_5d"], metrics["volatility"]


def classify_risk(volatility: float) -> str:
    if volatility < 0.01:
        return "low"
    if volatility < 0.02:
        return "medium"
    return "high"


def compute_score_breakdown(
    momentum: float,
    volatility: float,
    news_score: float,
    momentum_20d: Optional[float] = None,
    trend_gap: float = 0.0,
    positive_day_ratio: float = 0.5,
    volume_trend: float = 0.0,
    downside_volatility: Optional[float] = None,
    max_drawdown: float = 0.0,
    news_confidence: float = 0.5,
) -> Dict[str, float]:
    medium_momentum = momentum if momentum_20d is None else momentum_20d
    downside_risk = volatility if downside_volatility is None else downside_volatility

    normalized_short_momentum = clamp(momentum / SHORT_MOMENTUM_SCALE, -1.0, 1.0)
    normalized_medium_momentum = clamp(medium_momentum / MEDIUM_MOMENTUM_SCALE, -1.0, 1.0)
    normalized_trend_gap = clamp(trend_gap / TREND_GAP_SCALE, -1.0, 1.0)
    normalized_positive_ratio = clamp((positive_day_ratio - 0.5) * 2.0, -1.0, 1.0)
    normalized_volume = clamp(volume_trend / VOLUME_TREND_SCALE, -1.0, 1.0)
    normalized_volatility = clamp(volatility / VOLATILITY_SCALE, 0.0, 1.0)
    normalized_downside = clamp(downside_risk / DOWNSIDE_VOLATILITY_SCALE, 0.0, 1.0)
    normalized_drawdown = clamp(max_drawdown / MAX_DRAWDOWN_SCALE, 0.0, 1.0)
    centered_news = clamp((news_score - 0.5) * 2.0, -1.0, 1.0)

    trend_quality = clamp(
        normalized_positive_ratio * 0.60 + normalized_trend_gap * 0.40,
        -1.0,
        1.0,
    )
    news_multiplier = 0.08 + 0.12 * clamp(news_confidence, 0.0, 1.0)

    weighted_short_momentum = normalized_short_momentum * 0.22
    weighted_medium_momentum = normalized_medium_momentum * 0.18
    weighted_trend_quality = trend_quality * 0.14
    weighted_volume_confirmation = normalized_volume * 0.08
    weighted_volatility_penalty = -(normalized_volatility * 0.12)
    weighted_downside_penalty = -(normalized_downside * 0.10)
    weighted_drawdown_penalty = -(normalized_drawdown * 0.08)
    weighted_news = centered_news * news_multiplier

    momentum_total = (
        weighted_short_momentum
        + weighted_medium_momentum
        + weighted_trend_quality
        + weighted_volume_confirmation
    )
    volatility_penalty_total = (
        weighted_volatility_penalty
        + weighted_downside_penalty
        + weighted_drawdown_penalty
    )
    technical_total = momentum_total + volatility_penalty_total
    technical_direction = clamp(technical_total / 0.30, -1.0, 1.0)
    alignment_signal = technical_direction * centered_news if abs(centered_news) >= 0.10 else 0.0
    weighted_signal_alignment = alignment_signal * (
        SIGNAL_ALIGNMENT_BASE
        + SIGNAL_ALIGNMENT_CONFIDENCE_SCALE * clamp(news_confidence, 0.0, 1.0)
    )
    total_score = technical_total + weighted_news + weighted_signal_alignment

    return {
        "normalized_short_momentum": normalized_short_momentum,
        "normalized_medium_momentum": normalized_medium_momentum,
        "normalized_trend_gap": normalized_trend_gap,
        "normalized_positive_ratio": normalized_positive_ratio,
        "normalized_volume": normalized_volume,
        "normalized_volatility": normalized_volatility,
        "normalized_downside": normalized_downside,
        "normalized_drawdown": normalized_drawdown,
        "centered_news": centered_news,
        "weighted_short_momentum": weighted_short_momentum,
        "weighted_medium_momentum": weighted_medium_momentum,
        "weighted_trend_quality": weighted_trend_quality,
        "weighted_volume_confirmation": weighted_volume_confirmation,
        "weighted_volatility_penalty": weighted_volatility_penalty,
        "weighted_downside_penalty": weighted_downside_penalty,
        "weighted_drawdown_penalty": weighted_drawdown_penalty,
        "weighted_news": weighted_news,
        "alignment_signal": alignment_signal,
        "weighted_signal_alignment": weighted_signal_alignment,
        "momentum_total": momentum_total,
        "volatility_penalty_total": volatility_penalty_total,
        "technical_total": technical_total,
        "total": total_score,
    }


def compute_confidence_score(
    total_score: float,
    article_count: int,
    price_observations: int,
    news_score: float,
    positive_day_ratio: float = 0.5,
    max_drawdown: float = 0.0,
    news_confidence: float = 0.25,
    effective_article_count: float = 0.0,
    signal_alignment: float = 0.0,
) -> float:
    score_signal = clamp((total_score + 0.20) / 0.70, 0.0, 1.0) * 0.32
    history_signal = min(price_observations, 45) / 45 * 0.12
    trend_signal = clamp((positive_day_ratio - 0.40) / 0.45, 0.0, 1.0) * 0.10
    drawdown_signal = (1.0 - clamp(max_drawdown / 0.18, 0.0, 1.0)) * 0.12
    weighted_coverage_signal = min(effective_article_count, 4.0) / 4.0 * 0.10
    article_count_signal = min(article_count, 6) / 6 * 0.07
    news_confidence_signal = clamp(news_confidence, 0.0, 1.0) * 0.12
    conviction_signal = clamp(abs(news_score - 0.5) * 2.0, 0.0, 1.0) * 0.05
    alignment_signal = clamp(signal_alignment / 0.08, -1.0, 1.0) * 0.05
    no_news_penalty = 0.05 if article_count == 0 else 0.0

    return clamp(
        0.10
        + score_signal
        + history_signal
        + trend_signal
        + drawdown_signal
        + weighted_coverage_signal
        + article_count_signal
        + news_confidence_signal
        + conviction_signal
        + alignment_signal
        - no_news_penalty,
        0.05,
        0.99,
    )


def confidence_label(confidence_score: float) -> str:
    if confidence_score >= 0.75:
        return "high"
    if confidence_score >= 0.55:
        return "medium"
    return "low"


def signal_severity(state: str) -> int:
    return {
        "positive": 0,
        "watch": 1,
        "risk": 2,
    }.get(state, 0)


def build_thesis_monitor(
    candidate: StockCandidate,
    threshold_score: float,
    threshold_confidence: float,
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    reference = reference_date or market_today()
    price_as_of = date.fromisoformat(candidate.price_as_of)
    price_age_days = market_day_age(price_as_of, reference)
    news_age_days = None
    if candidate.news_as_of:
        news_age_days = market_day_age(date.fromisoformat(candidate.news_as_of), reference)

    score_margin = candidate.total_score - threshold_score
    confidence_margin = candidate.confidence_score - threshold_confidence

    if (
        candidate.momentum_5d < -0.01
        or candidate.trend_gap < -0.015
        or candidate.positive_day_ratio < 0.45
    ):
        momentum_state = "risk"
        momentum_detail = "Short-term momentum rolled over and the trend is no longer confirming the thesis."
    elif (
        candidate.momentum_5d < 0.01
        or candidate.trend_gap < 0.01
        or candidate.positive_day_ratio < 0.55
    ):
        momentum_state = "watch"
        momentum_detail = "Trend quality is still positive, but the tape is no longer decisive."
    else:
        momentum_state = "positive"
        momentum_detail = "Price action is still supporting the release thesis."

    if (
        candidate.volatility > 0.025
        or candidate.downside_volatility > 0.017
        or candidate.max_drawdown > 0.09
    ):
        volatility_state = "risk"
        volatility_detail = "Volatility expanded enough to weaken the setup."
    elif (
        candidate.volatility > 0.016
        or candidate.downside_volatility > 0.011
        or candidate.max_drawdown > 0.05
    ):
        volatility_state = "watch"
        volatility_detail = "Risk is still manageable, but volatility widened versus a clean setup."
    else:
        volatility_state = "positive"
        volatility_detail = "Volatility and drawdown remain controlled for this release."

    if candidate.article_count == 0:
        news_state = "watch"
        news_detail = "No recent company-specific articles passed the filters, so the thesis rests mostly on price action."
    elif candidate.news_score < 0.45 and candidate.news_confidence >= 0.55:
        news_state = "risk"
        news_detail = "Recent news tone leaned negative with enough confidence to matter."
    elif (
        candidate.news_confidence < 0.45
        or candidate.article_count < 2
        or candidate.effective_article_count < 1.0
    ):
        news_state = "watch"
        news_detail = "The news layer is still thin, so conviction should stay measured."
    elif candidate.news_score >= 0.58:
        news_state = "positive"
        news_detail = "Recent news flow is supportive and diversified enough to back the setup."
    else:
        news_state = "watch"
        news_detail = "News flow is mostly neutral, so the thesis needs technical confirmation to stay intact."

    if score_margin < 0.015 or confidence_margin < 0.03:
        margin_state = "risk"
        margin_detail = "The pick only barely cleared the release bar."
    elif score_margin < 0.04 or confidence_margin < 0.08:
        margin_state = "watch"
        margin_detail = "The pick still qualified, but the margin of safety is not wide."
    else:
        margin_state = "positive"
        margin_detail = "The pick cleared the release thresholds with room to spare."

    if price_age_days > 2 or (news_age_days is not None and news_age_days > 3):
        freshness_state = "risk"
        freshness_detail = "Some of the supporting data is getting old enough to weaken monitoring quality."
    elif price_age_days > 1 or (news_age_days is not None and news_age_days > 2):
        freshness_state = "watch"
        freshness_detail = "The data is still usable, but freshness should be checked before acting."
    else:
        freshness_state = "positive"
        freshness_detail = "Price and news timestamps are recent enough for a live monitoring read."

    signals = [
        {
            "label": "Momentum",
            "state": momentum_state,
            "value": f"{format_pct(candidate.momentum_5d)} 5D • {format_pct(candidate.momentum_20d)} 20D",
            "detail": momentum_detail,
        },
        {
            "label": "Volatility",
            "state": volatility_state,
            "value": f"{format_pct(candidate.volatility)} daily • {format_pct(candidate.max_drawdown)} drawdown",
            "detail": volatility_detail,
        },
        {
            "label": "News",
            "state": news_state,
            "value": f"{candidate.news_score:.2f} sentiment • {candidate.article_count} articles",
            "detail": news_detail,
        },
        {
            "label": "Margin",
            "state": margin_state,
            "value": f"{format_score(score_margin)} score • {format_score(confidence_margin)} conf",
            "detail": margin_detail,
        },
        {
            "label": "Freshness",
            "state": freshness_state,
            "value": (
                f"{format_market_age('price', price_age_days)} • {format_market_age('news', news_age_days)}"
                if news_age_days is not None
                else f"{format_market_age('price', price_age_days)} • no fresh news"
            ),
            "detail": freshness_detail,
        },
    ]

    severity = max(signal_severity(signal["state"]) for signal in signals)
    alert_details = [signal["detail"] for signal in signals if signal["state"] != "positive"]
    alerts = alert_details[:3]

    if severity == 0:
        headline = "Support is intact"
        summary = "Momentum, risk, and evidence are still aligned with the release thesis."
        status = "healthy"
    elif severity == 1:
        headline = "Support needs watching"
        summary = alerts[0] if alerts else "One or more support signals narrowed, even though the pick still qualifies."
        status = "watch"
    else:
        headline = "Support needs review"
        summary = alerts[0] if alerts else "A core support signal slipped into risk territory and needs re-checking."
        status = "risk"

    return {
        "status": status,
        "headline": headline,
        "summary": summary,
        "alerts": alerts,
        "signals": signals,
    }


def build_candidate(
    symbol: str,
    company_name: str,
    news_info: Optional[dict] = None,
) -> StockCandidate:
    price_series = fetch_price_series(symbol)
    technical_metrics = compute_technical_metrics(price_series)
    news_snapshot = build_news_snapshot(news_info)
    risk_level = classify_risk(technical_metrics["volatility"])

    score_breakdown = compute_score_breakdown(
        momentum=technical_metrics["momentum_5d"],
        momentum_20d=technical_metrics["momentum_20d"],
        volatility=technical_metrics["volatility"],
        trend_gap=technical_metrics["trend_gap"],
        positive_day_ratio=technical_metrics["positive_day_ratio"],
        volume_trend=technical_metrics["volume_trend"],
        downside_volatility=technical_metrics["downside_volatility"],
        max_drawdown=technical_metrics["max_drawdown"],
        news_score=news_snapshot.news_score,
        news_confidence=news_snapshot.news_confidence,
    )

    confidence_score = compute_confidence_score(
        total_score=score_breakdown["total"],
        article_count=news_snapshot.article_count,
        price_observations=price_series.observations,
        news_score=news_snapshot.news_score,
        positive_day_ratio=technical_metrics["positive_day_ratio"],
        max_drawdown=technical_metrics["max_drawdown"],
        news_confidence=news_snapshot.news_confidence,
        effective_article_count=news_snapshot.effective_article_count,
        signal_alignment=score_breakdown["weighted_signal_alignment"],
    )
    confidence = confidence_label(confidence_score)

    reasons = [
        (
            f"Price trend remains constructive: 5-day momentum is {format_pct(technical_metrics['momentum_5d'])} "
            f"and 20-day momentum is {format_pct(technical_metrics['momentum_20d'])}."
        ),
        (
            f"Trend quality is supported by {technical_metrics['positive_day_ratio'] * 100:.0f}% positive sessions "
            f"over the last 10 trading days and the price sitting {format_pct(technical_metrics['trend_gap'])} "
            f"relative to its 20-day average."
        ),
        (
            f"Risk profile shows daily volatility around {format_pct(technical_metrics['volatility'])}, "
            f"downside volatility near {format_pct(technical_metrics['downside_volatility'])}, "
            f"and a recent max drawdown of {format_pct(technical_metrics['max_drawdown'])}."
        ),
        (
            f"Volume confirmation is {format_pct(technical_metrics['volume_trend'])} versus the trailing baseline, "
            f"which helps separate steady breakouts from thin momentum."
        ),
    ]

    if news_snapshot.article_count > 0:
        reasons.append(
            (
                f"News signal is {news_snapshot.dominant_signal} with score {news_snapshot.news_score:.2f}, "
                f"confidence {news_snapshot.news_confidence:.2f}, and "
                f"{news_snapshot.effective_article_count:.1f} weighted articles across {news_snapshot.source_count} sources."
            )
        )
    else:
        reasons.append(
            "No recent relevant news passed the filter, so the news layer stayed neutral and contributed less confidence."
        )

    if news_snapshot.reasons:
        reasons.extend(news_snapshot.reasons[:2])

    if score_breakdown["weighted_signal_alignment"] > 0.02:
        reasons.append(
            "Price action and news are pointing in the same direction, which raises trust in the setup."
        )
    elif score_breakdown["weighted_signal_alignment"] < -0.02:
        reasons.append(
            "Price action and news are in conflict, so the model discounted the setup despite the raw signal."
        )

    reasons.append(
        f"Overall model score is {score_breakdown['total']:.3f} with {confidence} confidence."
    )

    return StockCandidate(
        symbol=symbol,
        company_name=company_name,
        reasons=reasons,
        risk_level=risk_level,
        total_score=score_breakdown["total"],
        confidence_score=confidence_score,
        confidence_label=confidence,
        price_as_of=price_series.latest_trading_date,
        news_as_of=news_snapshot.last_updated,
        article_count=news_snapshot.article_count,
        effective_article_count=news_snapshot.effective_article_count,
        source_count=news_snapshot.source_count,
        average_relevance=news_snapshot.average_relevance,
        momentum_5d=technical_metrics["momentum_5d"],
        momentum_20d=technical_metrics["momentum_20d"],
        volatility=technical_metrics["volatility"],
        downside_volatility=technical_metrics["downside_volatility"],
        max_drawdown=technical_metrics["max_drawdown"],
        trend_gap=technical_metrics["trend_gap"],
        positive_day_ratio=technical_metrics["positive_day_ratio"],
        volume_trend=technical_metrics["volume_trend"],
        news_score=news_snapshot.news_score,
        news_confidence=news_snapshot.news_confidence,
        raw_sentiment=news_snapshot.raw_sentiment,
        calibrated_sentiment=news_snapshot.calibrated_sentiment,
        dominant_signal=news_snapshot.dominant_signal,
        score_breakdown=score_breakdown,
        news_evidence=news_snapshot.news_evidence,
    )


def get_candidates() -> Tuple[List[StockCandidate], GenerationStats]:
    universe = load_universe()
    news_scores = load_news_scores()

    candidates: List[StockCandidate] = []
    skipped_details: List[Dict[str, str]] = []
    for symbol, company_name in universe:
        try:
            candidate = build_candidate(
                symbol=symbol,
                company_name=company_name,
                news_info=news_scores.get(symbol),
            )
            candidates.append(candidate)
        except Exception as exc:
            print(f"[WARN] Skipping {symbol}: {exc}")
            skipped_details.append({"symbol": symbol, "reason": str(exc)})

    if not candidates:
        raise RuntimeError("No candidates could be generated")

    stats = GenerationStats(
        universe_size=len(universe),
        evaluated_candidates=len(candidates),
        skipped_symbols=len(skipped_details),
        skipped_details=skipped_details[:10],
    )
    return candidates, stats


def qualifies(candidate: StockCandidate, threshold_score: float) -> bool:
    return (
        candidate.total_score >= threshold_score
        and candidate.confidence_score >= MIN_CONFIDENCE_THRESHOLD
    )


def build_no_pick_reason(
    label: str,
    best_candidate: Optional[StockCandidate],
    threshold_score: float,
) -> str:
    if best_candidate is None:
        return f"No {label} candidate had enough clean data to evaluate this week."

    if best_candidate.total_score < threshold_score:
        return (
            f"No {label} candidate cleared the minimum score threshold of {threshold_score:.2f}. "
            f"The best candidate was {best_candidate.symbol} at {best_candidate.total_score:.3f}."
        )

    return (
        f"No {label} candidate cleared the minimum confidence threshold of "
        f"{MIN_CONFIDENCE_THRESHOLD:.2f}. The best score was {best_candidate.symbol} at "
        f"{best_candidate.total_score:.3f} with {best_candidate.confidence_label} confidence."
    )


def select_best_candidate(candidates: List[StockCandidate]) -> SelectionDecision:
    ranked = sorted(candidates, key=lambda item: item.total_score, reverse=True)
    qualified = [item for item in ranked if qualifies(item, OVERALL_SELECTION_THRESHOLD)]
    if qualified:
        winner = qualified[0]
        return SelectionDecision(
            status="picked",
            status_reason=(
                f"{winner.symbol} cleared the commercial release thresholds for this market week "
                f"with a score of {winner.total_score:.3f} and {winner.confidence_label} confidence."
            ),
            threshold_score=OVERALL_SELECTION_THRESHOLD,
            threshold_confidence=MIN_CONFIDENCE_THRESHOLD,
            pick=winner,
            best_candidate=winner,
        )

    best_candidate = ranked[0] if ranked else None
    return SelectionDecision(
        status="no_pick",
        status_reason=build_no_pick_reason("overall", best_candidate, OVERALL_SELECTION_THRESHOLD),
        threshold_score=OVERALL_SELECTION_THRESHOLD,
        threshold_confidence=MIN_CONFIDENCE_THRESHOLD,
        pick=None,
        best_candidate=best_candidate,
    )


def select_best_per_risk(candidates: List[StockCandidate]) -> Dict[str, SelectionDecision]:
    decisions: Dict[str, SelectionDecision] = {}
    for risk_level, threshold in RISK_SELECTION_THRESHOLDS.items():
        bucket = [candidate for candidate in candidates if candidate.risk_level == risk_level]
        bucket.sort(key=lambda item: item.total_score, reverse=True)
        qualified = [item for item in bucket if qualifies(item, threshold)]

        if qualified:
            winner = qualified[0]
            decisions[risk_level] = SelectionDecision(
                status="picked",
                status_reason=(
                    f"{winner.symbol} is the strongest {risk_level}-risk candidate this week "
                    f"at {winner.total_score:.3f} with {winner.confidence_label} confidence."
                ),
                threshold_score=threshold,
                threshold_confidence=MIN_CONFIDENCE_THRESHOLD,
                pick=winner,
                best_candidate=winner,
            )
            continue

        best_candidate = bucket[0] if bucket else None
        decisions[risk_level] = SelectionDecision(
            status="no_pick",
            status_reason=build_no_pick_reason(f"{risk_level}-risk", best_candidate, threshold),
            threshold_score=threshold,
            threshold_confidence=MIN_CONFIDENCE_THRESHOLD,
            pick=None,
            best_candidate=best_candidate,
        )

    return decisions


def serialize_candidate(
    candidate: StockCandidate,
    threshold_score: float,
    threshold_confidence: float,
) -> Dict[str, Any]:
    return {
        "symbol": candidate.symbol,
        "company_name": candidate.company_name,
        "risk": candidate.risk_level,
        "model_score": round(candidate.total_score, 4),
        "confidence_score": round(candidate.confidence_score, 2),
        "confidence_label": candidate.confidence_label,
        "price_as_of": candidate.price_as_of,
        "news_as_of": candidate.news_as_of,
        "article_count": candidate.article_count,
        "metrics": {
            "momentum_5d": round(candidate.momentum_5d, 4),
            "momentum_20d": round(candidate.momentum_20d, 4),
            "daily_volatility": round(candidate.volatility, 4),
            "downside_volatility": round(candidate.downside_volatility, 4),
            "max_drawdown_20d": round(candidate.max_drawdown, 4),
            "positive_day_ratio_10d": round(candidate.positive_day_ratio, 4),
            "price_vs_20d_average": round(candidate.trend_gap, 4),
            "volume_trend_5d": round(candidate.volume_trend, 4),
            "news_sentiment": round(candidate.news_score, 2),
            "raw_news_sentiment": round(candidate.raw_sentiment, 4),
            "calibrated_news_sentiment": round(candidate.calibrated_sentiment, 4),
            "news_confidence": round(candidate.news_confidence, 2),
            "effective_article_count": round(candidate.effective_article_count, 2),
            "source_count": candidate.source_count,
            "average_relevance": round(candidate.average_relevance, 2),
        },
        "score_breakdown": {
            "momentum": round(candidate.score_breakdown["momentum_total"], 4),
            "short_momentum": round(candidate.score_breakdown["weighted_short_momentum"], 4),
            "medium_momentum": round(candidate.score_breakdown["weighted_medium_momentum"], 4),
            "trend_quality": round(candidate.score_breakdown["weighted_trend_quality"], 4),
            "volume_confirmation": round(candidate.score_breakdown["weighted_volume_confirmation"], 4),
            "volatility_penalty": round(candidate.score_breakdown["volatility_penalty_total"], 4),
            "daily_volatility_penalty": round(candidate.score_breakdown["weighted_volatility_penalty"], 4),
            "downside_penalty": round(candidate.score_breakdown["weighted_downside_penalty"], 4),
            "drawdown_penalty": round(candidate.score_breakdown["weighted_drawdown_penalty"], 4),
            "news_adjustment": round(candidate.score_breakdown["weighted_news"], 4),
            "signal_alignment": round(candidate.score_breakdown["weighted_signal_alignment"], 4),
            "technical_total": round(candidate.score_breakdown["technical_total"], 4),
            "total": round(candidate.score_breakdown["total"], 4),
        },
        "reasons": candidate.reasons,
        "news_evidence": candidate.news_evidence,
        "thesis_monitor": build_thesis_monitor(
            candidate=candidate,
            threshold_score=threshold_score,
            threshold_confidence=threshold_confidence,
        ),
    }


def serialize_selection(selection: SelectionDecision) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": selection.status,
        "status_reason": selection.status_reason,
        "threshold_score": round(selection.threshold_score, 2),
        "threshold_confidence": round(selection.threshold_confidence, 2),
        "pick": serialize_candidate(
            selection.pick,
            threshold_score=selection.threshold_score,
            threshold_confidence=selection.threshold_confidence,
        ) if selection.pick else None,
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


def derive_data_as_of(candidates: List[StockCandidate]) -> str:
    candidate_dates = [
        item
        for candidate in candidates
        for item in [candidate.price_as_of, candidate.news_as_of]
        if item
    ]
    return max(candidate_dates) if candidate_dates else date.today().isoformat()


def common_payload(
    generated_at: str,
    data_as_of: str,
    expected_next_refresh_at: str,
    stale_after: str,
    market_week: MarketWeek,
    stats: GenerationStats,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at,
        "data_as_of": data_as_of,
        "expected_next_refresh_at": expected_next_refresh_at,
        "stale_after": stale_after,
        "market_context": {
            "timezone": MARKET_TIMEZONE,
            "week_id": market_week.week_id,
            "week_label": market_week.week_label,
            "week_start": market_week.week_start.isoformat(),
            "week_end": market_week.week_end.isoformat(),
        },
        "generation_summary": {
            "universe_size": stats.universe_size,
            "evaluated_candidates": stats.evaluated_candidates,
            "skipped_symbols": stats.skipped_symbols,
            "skipped_details": stats.skipped_details,
        },
        "selection_thresholds": {
            "overall_score": OVERALL_SELECTION_THRESHOLD,
            "risk_scores": RISK_SELECTION_THRESHOLDS,
            "minimum_confidence": MIN_CONFIDENCE_THRESHOLD,
        },
    }


def build_current_pick_payload(
    selection: SelectionDecision,
    generated_at: str,
    data_as_of: str,
    expected_next_refresh_at: str,
    stale_after: str,
    market_week: MarketWeek,
    stats: GenerationStats,
) -> Dict[str, Any]:
    payload = common_payload(
        generated_at=generated_at,
        data_as_of=data_as_of,
        expected_next_refresh_at=expected_next_refresh_at,
        stale_after=stale_after,
        market_week=market_week,
        stats=stats,
    )
    payload["selection"] = serialize_selection(selection)
    return payload


def build_risk_picks_payload(
    overall_selection: SelectionDecision,
    risk_selections: Dict[str, SelectionDecision],
    generated_at: str,
    data_as_of: str,
    expected_next_refresh_at: str,
    stale_after: str,
    market_week: MarketWeek,
    stats: GenerationStats,
) -> Dict[str, Any]:
    payload = common_payload(
        generated_at=generated_at,
        data_as_of=data_as_of,
        expected_next_refresh_at=expected_next_refresh_at,
        stale_after=stale_after,
        market_week=market_week,
        stats=stats,
    )
    payload["overall_selection"] = serialize_selection(overall_selection)
    payload["risk_selections"] = {
        risk: serialize_selection(selection)
        for risk, selection in risk_selections.items()
    }
    return payload


def load_existing_history_entries(history_path: Path) -> List[Dict[str, Any]]:
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


def history_week_id(week_start: str) -> str:
    parsed = date.fromisoformat(week_start)
    iso = parsed.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def normalize_history_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    week_start = entry.get("week_start") or date.today().isoformat()
    week_end = entry.get("week_end") or week_start
    week_id = entry.get("week_id") or history_week_id(week_start)
    status = entry.get("status") or ("picked" if entry.get("symbol") else "no_pick")

    return {
        "week_id": week_id,
        "week_start": week_start,
        "week_end": week_end,
        "week_label": entry.get("week_label") or f"{week_start} - {week_end}",
        "logged_at": entry.get("logged_at") or week_start,
        "status": status,
        "status_reason": entry.get("status_reason") or "",
        "symbol": entry.get("symbol"),
        "company_name": entry.get("company_name"),
        "risk": entry.get("risk"),
        "model_score": entry.get("model_score", entry.get("score")),
        "confidence_score": entry.get("confidence_score"),
        "confidence_label": entry.get("confidence_label"),
        "data_as_of": entry.get("data_as_of"),
        "model_version": entry.get("model_version", MODEL_VERSION),
    }


def build_history_entry(
    market_week: MarketWeek,
    selection: SelectionDecision,
    generated_at: str,
    data_as_of: str,
) -> Dict[str, Any]:
    pick = selection.pick
    return {
        "week_id": market_week.week_id,
        "week_start": market_week.week_start.isoformat(),
        "week_end": market_week.week_end.isoformat(),
        "week_label": market_week.week_label,
        "logged_at": generated_at,
        "status": selection.status,
        "status_reason": selection.status_reason,
        "symbol": pick.symbol if pick else None,
        "company_name": pick.company_name if pick else None,
        "risk": pick.risk_level if pick else None,
        "model_score": round(pick.total_score, 4) if pick else None,
        "confidence_score": round(pick.confidence_score, 2) if pick else None,
        "confidence_label": pick.confidence_label if pick else None,
        "data_as_of": data_as_of,
        "model_version": MODEL_VERSION,
    }


def update_history(
    market_week: MarketWeek,
    selection: SelectionDecision,
    generated_at: str,
    data_as_of: str,
    history_path: Path = HISTORY_PATH,
) -> Dict[str, Any]:
    entries = [normalize_history_entry(entry) for entry in load_existing_history_entries(history_path)]
    new_entry = build_history_entry(market_week, selection, generated_at, data_as_of)

    filtered_entries = [entry for entry in entries if entry.get("week_id") != market_week.week_id]
    filtered_entries.append(new_entry)
    filtered_entries.sort(key=lambda item: item.get("week_start", ""))
    filtered_entries = filtered_entries[-104:]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at,
        "entries": filtered_entries,
    }

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    print("[INFO] Generating weekly stock picks...")
    generated_at = iso_utc(now_utc())
    market_week = build_market_week()
    expected_next_refresh_at, stale_after = build_refresh_window(market_week)

    candidates, stats = get_candidates()
    overall_selection = select_best_candidate(candidates)
    risk_selections = select_best_per_risk(candidates)
    data_as_of = derive_data_as_of(candidates)

    current_pick_payload = build_current_pick_payload(
        selection=overall_selection,
        generated_at=generated_at,
        data_as_of=data_as_of,
        expected_next_refresh_at=expected_next_refresh_at,
        stale_after=stale_after,
        market_week=market_week,
        stats=stats,
    )
    risk_picks_payload = build_risk_picks_payload(
        overall_selection=overall_selection,
        risk_selections=risk_selections,
        generated_at=generated_at,
        data_as_of=data_as_of,
        expected_next_refresh_at=expected_next_refresh_at,
        stale_after=stale_after,
        market_week=market_week,
        stats=stats,
    )

    write_json(CURRENT_PICK_PATH, current_pick_payload)
    write_json(RISK_PICKS_PATH, risk_picks_payload)
    update_history(
        market_week=market_week,
        selection=overall_selection,
        generated_at=generated_at,
        data_as_of=data_as_of,
    )

    print("[INFO] current_pick.json, risk_picks.json and history.json updated.")


if __name__ == "__main__":
    main()
