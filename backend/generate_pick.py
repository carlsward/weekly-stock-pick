import json
import math
import os
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

try:
    from backend.sector_utils import load_sector_map, sector_display_name
except ImportError:
    from sector_utils import load_sector_map, sector_display_name

MODEL_VERSION = "v3.3"
SCHEMA_VERSION = 2
MARKET_TIMEZONE = "America/New_York"
MARKET_TZ = ZoneInfo(MARKET_TIMEZONE)

UNIVERSE_CSV_PATH_ENV = "UNIVERSE_CSV_PATH"
DEFAULT_UNIVERSE_CSV_PATH = Path("universe.csv")
NEWS_SCORES_PATH = Path("news_scores.json")
SECTOR_SCORES_PATH = Path("sector_scores.json")
CURRENT_PICK_PATH = Path("current_pick.json")
RISK_PICKS_PATH = Path("risk_picks.json")
HISTORY_PATH = Path("history.json")
MARKET_BENCHMARK_SYMBOL = "SPY"
SECTOR_ETF_BY_SECTOR = {
    "communication_services": "XLC",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "energy": "XLE",
    "financials": "XLF",
    "healthcare": "XLV",
    "industrials": "XLI",
    "technology": "XLK",
}

SHORT_MOMENTUM_SCALE = 0.08
MEDIUM_MOMENTUM_SCALE = 0.18
TREND_GAP_SCALE = 0.08
VOLUME_TREND_SCALE = 0.60
VOLATILITY_SCALE = 0.04
DOWNSIDE_VOLATILITY_SCALE = 0.025
MAX_DRAWDOWN_SCALE = 0.12
RELATIVE_STRENGTH_SCALE = 0.08

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
SECTOR_STALE_AFTER_DAYS = 3
PRICE_FETCH_ATTEMPTS = 3
PRICE_FETCH_RETRY_SECONDS = 1.5
PRICE_REQUEST_TIMEOUT_SECONDS = 20
SIGNAL_ALIGNMENT_BASE = 0.03
SIGNAL_ALIGNMENT_CONFIDENCE_SCALE = 0.04
STOOQ_DAILY_ENDPOINT = "https://stooq.com/q/d/l/"
ENABLE_HISTORY_REALIZED_ENRICHMENT_ENV = "ENABLE_HISTORY_REALIZED_ENRICHMENT"
ALLOW_PRICE_FALLBACK_ENV = "ALLOW_PRICE_FALLBACK"
FORCE_PRICE_FALLBACK_ENV = "FORCE_PRICE_FALLBACK"
CALIBRATION_LOOKBACK_DAYS = 320
CALIBRATION_STEP_DAYS = 5
CALIBRATION_MIN_ROWS = 180
CALIBRATION_RIDGE_ALPHA = 0.75
CALIBRATION_PREDICTION_PCTL = 0.90
TECHNICAL_SCORE_TARGET_SCALE = 0.30
BLOCK_CALIBRATION_MIN_ROWS = 12
BLOCK_BASE_PRIORS = {
    "news": 0.18,
    "macro": 0.10,
    "sector": 0.08,
}
BLOCK_WEIGHT_FLOORS = {
    "news": 0.10,
    "macro": 0.05,
    "sector": 0.04,
}
BLOCK_WEIGHT_CAPS = {
    "news": 0.24,
    "macro": 0.14,
    "sector": 0.12,
}
LAYER_OVERLAP_MAX_PENALTY = 0.55
LAYER_OVERLAP_MIN_PENALTY = 0.30
TECHNICAL_FEATURE_ORDER = (
    "short_momentum",
    "medium_momentum",
    "trend_gap",
    "positive_ratio",
    "volume_confirmation",
    "market_relative",
    "sector_relative",
    "inverse_volatility",
    "inverse_downside",
    "inverse_drawdown",
)
TECHNICAL_GROUPS = {
    "trend_strength": ("short_momentum", "medium_momentum", "trend_gap", "positive_ratio"),
    "relative_strength": ("market_relative", "sector_relative"),
    "participation": ("volume_confirmation",),
    "risk_control": ("inverse_volatility", "inverse_downside", "inverse_drawdown"),
}
DEFAULT_TECHNICAL_FEATURE_WEIGHTS = {
    "short_momentum": 0.22,
    "medium_momentum": 0.18,
    "trend_gap": 0.06,
    "positive_ratio": 0.08,
    "volume_confirmation": 0.08,
    "market_relative": 0.07,
    "sector_relative": 0.05,
    "inverse_volatility": 0.12,
    "inverse_downside": 0.09,
    "inverse_drawdown": 0.05,
}
THEME_KEYWORDS = {
    "oil": ("oil", "crude", "opec", "lng", "fuel", "refinery", "natural gas"),
    "semiconductors": ("chip", "semiconductor", "gpu", "foundry", "wafer"),
    "ai": ("ai", "artificial intelligence", "data center", "accelerator", "inference"),
    "cloud": ("cloud", "azure", "aws", "software", "saas"),
    "rates": ("interest rate", "yield", "federal reserve", "rate cut", "rate hike"),
    "inflation": ("inflation", "cpi", "ppi", "pricing pressure"),
    "consumer": ("consumer spending", "retail sales", "foot traffic", "demand", "pricing"),
    "autos": ("auto", "vehicle", "ev", "deliveries", "registrations", "production"),
    "healthcare": ("drug", "fda", "clinical trial", "reimbursement", "medicare"),
    "regulation": ("tariff", "sanction", "regulation", "antitrust", "export control"),
    "supply_chain": ("shortage", "oversupply", "supply chain", "inventory", "capacity"),
    "defense": ("defense", "aerospace", "military", "missile"),
    "shipping": ("shipping", "freight", "logistics", "container"),
    "earnings": ("guidance", "earnings", "revenue", "margin", "sales"),
}


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


@dataclass(frozen=True)
class SectorSnapshot:
    sector: str
    sector_score: float
    sector_confidence: float
    direction: str
    reasons: List[str]
    supporting_articles: List[Dict[str, Any]]
    last_updated: Optional[str]


@dataclass(frozen=True)
class MacroSnapshot:
    symbol: str
    company_name: str
    sector: str
    macro_score: float
    macro_confidence: float
    direction: str
    reasons: List[str]
    supporting_articles: List[Dict[str, Any]]
    last_updated: Optional[str]


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
    market_relative_5d: float
    market_relative_20d: float
    sector_relative_5d: float
    sector_relative_20d: float
    news_score: float
    news_confidence: float
    macro_score: float
    macro_confidence: float
    raw_sentiment: float
    calibrated_sentiment: float
    dominant_signal: str
    score_breakdown: Dict[str, float]
    news_evidence: List[Dict[str, Any]]
    macro_evidence: List[Dict[str, Any]]
    macro_as_of: Optional[str] = None
    sector_as_of: Optional[str] = None
    sector: str = "unknown"
    sector_score: float = 0.5
    sector_confidence: float = 0.2
    sector_direction: str = "neutral"
    sector_reasons: List[str] = None


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


@dataclass(frozen=True)
class ModelCalibration:
    technical_weights: Dict[str, float]
    block_weights: Dict[str, float]
    technical_scale: float
    training_row_count: int
    training_ic: float
    block_row_count: int
    source: str


PRICE_SERIES_CACHE: Dict[Tuple[str, int], PriceSeries] = {}
PRICE_FRAME_CACHE: Dict[str, pd.DataFrame] = {}


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


def history_realized_enrichment_enabled() -> bool:
    raw = os.getenv(ENABLE_HISTORY_REALIZED_ENRICHMENT_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def allow_price_fallback() -> bool:
    raw = os.getenv(ALLOW_PRICE_FALLBACK_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def force_price_fallback() -> bool:
    raw = os.getenv(FORCE_PRICE_FALLBACK_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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


def resolve_universe_csv_path() -> Path:
    configured = os.getenv(UNIVERSE_CSV_PATH_ENV, "").strip()
    return Path(configured) if configured else DEFAULT_UNIVERSE_CSV_PATH


def load_universe(path: Optional[Path] = None) -> List[Tuple[str, str]]:
    path = path or resolve_universe_csv_path()
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
    normalized = symbol.strip().upper().replace(".", "-")
    return normalized if normalized.endswith(".US") else f"{normalized}.US"


def stooq_symbol_variants(symbol: str) -> List[str]:
    normalized = symbol.strip().upper()
    if not normalized:
        return []

    bases = [
        normalized.replace(".", "-"),
        normalized,
        normalized.replace(".", ""),
    ]
    variants: List[str] = []
    for base in bases:
        if not base:
            continue
        suffixed = base if base.endswith(".US") else f"{base}.US"
        if suffixed not in variants:
            variants.append(suffixed)
        if base not in variants:
            variants.append(base)
    return variants


def build_price_series_from_frame(frame: pd.DataFrame, symbol: str, max_days: int = 45) -> PriceSeries:
    working = build_clean_price_frame(frame, symbol)

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


def build_clean_price_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
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

    return (
        working.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


def fetch_stooq_price_frame(symbol: str) -> pd.DataFrame:
    attempts: List[str] = []
    for query_symbol in stooq_symbol_variants(symbol):
        params = {
            "s": query_symbol,
            "i": "d",
        }
        url = f"{STOOQ_DAILY_ENDPOINT}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Accept": "text/csv,text/plain,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://stooq.com/",
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
            },
        )

        try:
            with urlopen(request, timeout=PRICE_REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8", errors="replace").strip()
        except HTTPError as exc:
            attempts.append(f"{query_symbol}: HTTP {exc.code}")
            continue
        except (URLError, TimeoutError, OSError) as exc:
            attempts.append(f"{query_symbol}: request failed ({exc})")
            continue

        if not body:
            attempts.append(f"{query_symbol}: empty body")
            continue
        if "No data" in body or "404 Not Found" in body:
            attempts.append(f"{query_symbol}: no data")
            continue

        try:
            frame = pd.read_csv(StringIO(body))
        except Exception as exc:
            attempts.append(f"{query_symbol}: CSV parse failed ({exc})")
            continue

        if frame.empty:
            attempts.append(f"{query_symbol}: empty csv")
            continue
        return frame

    raise RuntimeError(
        f"Stooq returned no usable price data for {symbol}. Tried: {', '.join(attempts)}"
    )


def build_synthetic_price_frame(symbol: str, periods: int = 260) -> pd.DataFrame:
    end_date = pd.Timestamp(market_today())
    while end_date.weekday() >= 5:
        end_date -= pd.Timedelta(days=1)

    dates = pd.bdate_range(end=end_date, periods=periods)
    if len(dates) != periods:
        start_date = end_date - pd.Timedelta(days=periods * 3)
        dates = pd.bdate_range(start=start_date, end=end_date).tail(periods)
    symbol_key = symbol.upper()
    seed = sum((index + 1) * ord(char) for index, char in enumerate(symbol_key))
    base_price = 45.0 + (seed % 180)
    drift_map = {
        "SPY": 0.00045,
        "XLF": 0.00040,
        "XLK": 0.00055,
        "XLE": 0.00050,
        "MSFT": 0.00075,
        "JPM": 0.00050,
        "XOM": 0.00060,
    }
    base_drift = drift_map.get(symbol_key, 0.00045 + ((seed % 9) - 4) * 0.00003)
    phase = (seed % 23) / 23.0 * math.pi
    volume_base = 1_500_000 + (seed % 40) * 110_000

    prices: List[float] = []
    volumes: List[float] = []
    current_price = base_price
    for index, _ in enumerate(dates):
        cyclical = 0.0045 * math.sin(index / 6.5 + phase)
        micro = 0.0020 * math.cos(index / 3.1 + phase / 2.0)
        move = base_drift + cyclical + micro
        current_price = max(5.0, current_price * (1.0 + move))
        volume = volume_base * (1.0 + 0.16 * math.sin(index / 5.0 + phase))
        prices.append(round(current_price, 4))
        volumes.append(max(100_000.0, round(volume)))

    return pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
            "Volume": volumes,
        }
    )


def fetch_price_frame(symbol: str) -> pd.DataFrame:
    if force_price_fallback():
        print(f"[WARN] Using forced synthetic price fallback for {symbol}.")
        return build_synthetic_price_frame(symbol)

    providers = [("stooq", fetch_stooq_price_frame)]
    failures: List[str] = []

    for provider_name, provider_fetcher in providers:
        provider_error: Optional[Exception] = None
        for attempt in range(1, PRICE_FETCH_ATTEMPTS + 1):
            try:
                frame = provider_fetcher(symbol)
                return build_clean_price_frame(frame, symbol)
            except Exception as exc:
                provider_error = exc
                if attempt < PRICE_FETCH_ATTEMPTS:
                    time_module.sleep(PRICE_FETCH_RETRY_SECONDS * attempt)

        if provider_error is not None:
            failures.append(f"{provider_name}: {provider_error}")

    if allow_price_fallback():
        print(f"[WARN] Using synthetic price fallback for {symbol} because live price providers failed.")
        return build_synthetic_price_frame(symbol)

    raise RuntimeError(
        f"Unable to fetch usable price frame for {symbol} after {PRICE_FETCH_ATTEMPTS} attempts per provider: "
        + "; ".join(failures)
    )


def fetch_price_frame_cached(symbol: str) -> pd.DataFrame:
    cache_key = symbol.upper()
    cached = PRICE_FRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    fetched = fetch_price_frame(symbol)
    PRICE_FRAME_CACHE[cache_key] = fetched
    return fetched.copy()


def fetch_price_series(symbol: str, max_days: int = 45) -> PriceSeries:
    cached_frame = PRICE_FRAME_CACHE.get(symbol.upper())
    if cached_frame is not None:
        return build_price_series_from_frame(cached_frame.copy(), symbol, max_days)

    frame = fetch_price_frame(symbol)
    return build_price_series_from_frame(frame, symbol, max_days)


def fetch_price_series_cached(symbol: str, max_days: int = 45) -> PriceSeries:
    cache_key = (symbol.upper(), max_days)
    cached = PRICE_SERIES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fetched = fetch_price_series(symbol, max_days)
    PRICE_SERIES_CACHE[cache_key] = fetched
    return fetched


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


def load_sector_scores(path: Path = SECTOR_SCORES_PATH) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        print(f"[WARN] Could not find {path}; sector overlay will stay neutral.")

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
                    "doc_type": str(article.get("doc_type", "")).strip() or None,
                    "company_relevance": float(article["company_relevance"]) if isinstance(article.get("company_relevance"), (int, float)) else None,
                    "materiality": float(article["materiality"]) if isinstance(article.get("materiality"), (int, float)) else None,
                    "llm_confidence": float(article["llm_confidence"]) if isinstance(article.get("llm_confidence"), (int, float)) else None,
                    "llm_impact_score": float(article["llm_impact_score"]) if isinstance(article.get("llm_impact_score"), (int, float)) else None,
                    "llm_reason": str(article.get("llm_reason", "")).strip() or None,
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


def normalize_sector_supporting_articles(sector_info: dict) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    articles = sector_info.get("supporting_articles")
    if not isinstance(articles, list):
        return normalized

    for article in articles[:3]:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title", "")).strip()
        if not title:
            continue
        normalized.append(
            {
                "title": title,
                "feed": str(article.get("feed", "")).strip() or None,
                "provider": str(article.get("provider", "")).strip() or None,
                "url": str(article.get("url", "")).strip() or None,
                "published_at": str(article.get("published_at", "")).strip() or None,
                "impact": str(article.get("impact", "")).strip() or None,
                "weight": float(article["weight"]) if isinstance(article.get("weight"), (int, float)) else None,
                "event_type": str(article.get("event_type", "")).strip() or None,
                "transmission_channel": str(article.get("transmission_channel", "")).strip() or None,
                "affected_inputs": [
                    str(item).strip()
                    for item in article.get("affected_inputs", [])
                    if str(item).strip()
                ]
                if isinstance(article.get("affected_inputs"), list)
                else [],
                "horizon": str(article.get("horizon", "")).strip() or None,
                "reason": str(article.get("reason", "")).strip() or None,
            }
        )

    return normalized


def latest_payload_article_date(articles: Any) -> Optional[str]:
    if not isinstance(articles, list):
        return None

    published_dates: List[str] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        published_at = parse_iso_datetime(article.get("published_at"))
        if published_at is not None:
            published_dates.append(published_at.date().isoformat())

    if not published_dates:
        return None
    return max(published_dates)


def theme_tokens_from_text(text: str) -> List[str]:
    normalized = text.lower()
    matches = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matches.append(theme)
    return matches


def extract_layer_themes(
    reasons: Sequence[str],
    evidence: Sequence[Dict[str, Any]],
) -> List[str]:
    themes: List[str] = []
    for reason in reasons:
        themes.extend(theme_tokens_from_text(str(reason)))
    for article in evidence:
        if not isinstance(article, dict):
            continue
        themes.extend(theme_tokens_from_text(str(article.get("title", ""))))
        themes.extend(theme_tokens_from_text(str(article.get("reason", ""))))
        themes.extend(theme_tokens_from_text(str(article.get("llm_reason", ""))))
        themes.extend(theme_tokens_from_text(str(article.get("event_type", ""))))
        themes.extend(theme_tokens_from_text(str(article.get("transmission_channel", ""))))
        for affected_input in article.get("affected_inputs", []) if isinstance(article.get("affected_inputs"), list) else []:
            themes.extend(theme_tokens_from_text(str(affected_input)))
    return list(dict.fromkeys(themes))


def overlap_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def dedupe_layer_penalties(
    news_snapshot: NewsSnapshot,
    macro_snapshot: MacroSnapshot,
    sector_snapshot: SectorSnapshot,
) -> Dict[str, Any]:
    news_themes = extract_layer_themes(news_snapshot.reasons, news_snapshot.news_evidence)
    macro_themes = extract_layer_themes(macro_snapshot.reasons, macro_snapshot.supporting_articles)
    sector_themes = extract_layer_themes(sector_snapshot.reasons, sector_snapshot.supporting_articles)

    news_macro_overlap = overlap_ratio(news_themes, macro_themes)
    news_sector_overlap = overlap_ratio(news_themes, sector_themes)
    macro_sector_overlap = overlap_ratio(macro_themes, sector_themes)

    macro_penalty = 1.0 - clamp(news_macro_overlap * LAYER_OVERLAP_MAX_PENALTY, 0.0, LAYER_OVERLAP_MAX_PENALTY)
    sector_penalty = 1.0 - clamp(
        max(news_sector_overlap, macro_sector_overlap) * (LAYER_OVERLAP_MAX_PENALTY + 0.05),
        0.0,
        LAYER_OVERLAP_MAX_PENALTY + 0.05,
    )

    if news_macro_overlap > 0.0 and macro_snapshot.macro_confidence >= 0.35:
        macro_penalty = min(macro_penalty, 1.0 - LAYER_OVERLAP_MIN_PENALTY)
    if max(news_sector_overlap, macro_sector_overlap) > 0.0 and sector_snapshot.sector_confidence >= 0.35:
        sector_penalty = min(sector_penalty, 1.0 - (LAYER_OVERLAP_MIN_PENALTY + 0.05))

    return {
        "news_themes": news_themes,
        "macro_themes": macro_themes,
        "sector_themes": sector_themes,
        "news_macro_overlap": news_macro_overlap,
        "news_sector_overlap": news_sector_overlap,
        "macro_sector_overlap": macro_sector_overlap,
        "macro_penalty": clamp(macro_penalty, 0.40, 1.0),
        "sector_penalty": clamp(sector_penalty, 0.35, 1.0),
    }


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
    last_updated = news_info.get("last_updated") or latest_payload_article_date(news_info.get("top_articles"))
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


def build_sector_snapshot(
    sector: str,
    sector_scores_payload: Optional[Dict[str, Any]],
) -> SectorSnapshot:
    neutral_reason = (
        f"No fresh broad catalyst clearly favored the {sector_display_name(sector)} sector, so the sector overlay stayed neutral."
    )
    if not sector_scores_payload:
        return SectorSnapshot(
            sector=sector,
            sector_score=0.5,
            sector_confidence=0.20,
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
            sector_confidence=0.20,
            direction="neutral",
            reasons=[neutral_reason],
            supporting_articles=[],
            last_updated=str(sector_scores_payload.get("last_updated")) if sector_scores_payload.get("last_updated") else None,
        )

    reasons = sector_info.get("reasons")
    normalized_reasons = [str(item) for item in reasons] if isinstance(reasons, list) and reasons else [neutral_reason]
    base_score = float(sector_info.get("score", 0.5))
    base_confidence = float(sector_info.get("confidence", 0.2))
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
    if parsed_last_updated and parsed_last_updated < now - timedelta(days=SECTOR_STALE_AFTER_DAYS):
        age_days = max(1, (now - parsed_last_updated).days)
        freshness_multiplier = 0.65
        normalized_reasons.append(
            f"Sector overlay is {age_days} days old, so its impact and confidence were reduced."
        )

    score = 0.5 + (base_score - 0.5) * freshness_multiplier
    confidence = clamp(base_confidence * freshness_multiplier, 0.18, 0.95)
    direction = str(sector_info.get("direction", "neutral"))
    if freshness_multiplier < 1.0 and abs(score - 0.5) < 0.08:
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


def build_macro_snapshot(
    symbol: str,
    company_name: str,
    sector: str,
    sector_scores_payload: Optional[Dict[str, Any]],
) -> MacroSnapshot:
    neutral_reason = (
        f"No fresh world-news catalyst clearly favored {company_name}, so the global overlay stayed neutral."
    )
    if not sector_scores_payload:
        return MacroSnapshot(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            macro_score=0.5,
            macro_confidence=0.20,
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
            macro_confidence=0.20,
            direction="neutral",
            reasons=[neutral_reason],
            supporting_articles=[],
            last_updated=str(sector_scores_payload.get("last_updated")) if sector_scores_payload.get("last_updated") else None,
        )

    reasons = symbol_info.get("reasons")
    normalized_reasons = [str(item) for item in reasons] if isinstance(reasons, list) and reasons else [neutral_reason]
    base_score = float(symbol_info.get("score", 0.5))
    base_confidence = float(symbol_info.get("confidence", 0.2))
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
    if parsed_last_updated and parsed_last_updated < now - timedelta(days=SECTOR_STALE_AFTER_DAYS):
        age_days = max(1, (now - parsed_last_updated).days)
        freshness_multiplier = 0.65
        normalized_reasons.append(
            f"Global overlay is {age_days} days old, so its impact and confidence were reduced."
        )

    score = 0.5 + (base_score - 0.5) * freshness_multiplier
    confidence = clamp(base_confidence * freshness_multiplier, 0.18, 0.95)
    direction = str(symbol_info.get("direction", "neutral"))
    if freshness_multiplier < 1.0 and abs(score - 0.5) < 0.08:
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


def relative_return(stock_return: float, benchmark_return: float) -> float:
    return safe_divide(1.0 + stock_return, 1.0 + benchmark_return, 1.0) - 1.0


def compute_relative_strength_metrics(
    stock_metrics: Dict[str, float],
    market_metrics: Dict[str, float],
    sector_metrics: Dict[str, float],
) -> Dict[str, float]:
    market_relative_5d = relative_return(stock_metrics["momentum_5d"], market_metrics["momentum_5d"])
    market_relative_20d = relative_return(stock_metrics["momentum_20d"], market_metrics["momentum_20d"])
    sector_relative_5d = relative_return(stock_metrics["momentum_5d"], sector_metrics["momentum_5d"])
    sector_relative_20d = relative_return(stock_metrics["momentum_20d"], sector_metrics["momentum_20d"])
    return {
        "market_relative_5d": market_relative_5d,
        "market_relative_20d": market_relative_20d,
        "sector_relative_5d": sector_relative_5d,
        "sector_relative_20d": sector_relative_20d,
    }


def classify_risk(volatility: float) -> str:
    if volatility < 0.01:
        return "low"
    if volatility < 0.02:
        return "medium"
    return "high"


def build_normalized_technical_features(
    *,
    momentum: float,
    medium_momentum: float,
    trend_gap: float,
    positive_day_ratio: float,
    volume_trend: float,
    volatility: float,
    downside_risk: float,
    max_drawdown: float,
    market_relative_5d: float,
    market_relative_20d: float,
    sector_relative_5d: float,
    sector_relative_20d: float,
) -> Dict[str, float]:
    normalized_short_momentum = clamp(momentum / SHORT_MOMENTUM_SCALE, -1.0, 1.0)
    normalized_medium_momentum = clamp(medium_momentum / MEDIUM_MOMENTUM_SCALE, -1.0, 1.0)
    normalized_trend_gap = clamp(trend_gap / TREND_GAP_SCALE, -1.0, 1.0)
    normalized_positive_ratio = clamp((positive_day_ratio - 0.5) * 2.0, -1.0, 1.0)
    normalized_volume = clamp(volume_trend / VOLUME_TREND_SCALE, -1.0, 1.0)
    normalized_volatility = clamp(volatility / VOLATILITY_SCALE, 0.0, 1.0)
    normalized_downside = clamp(downside_risk / DOWNSIDE_VOLATILITY_SCALE, 0.0, 1.0)
    normalized_drawdown = clamp(max_drawdown / MAX_DRAWDOWN_SCALE, 0.0, 1.0)
    normalized_market_relative = clamp(
        (market_relative_5d + market_relative_20d) / 2.0 / RELATIVE_STRENGTH_SCALE,
        -1.0,
        1.0,
    )
    normalized_sector_relative = clamp(
        (sector_relative_5d + sector_relative_20d) / 2.0 / RELATIVE_STRENGTH_SCALE,
        -1.0,
        1.0,
    )

    return {
        "short_momentum": normalized_short_momentum,
        "medium_momentum": normalized_medium_momentum,
        "trend_gap": normalized_trend_gap,
        "positive_ratio": normalized_positive_ratio,
        "volume_confirmation": normalized_volume,
        "market_relative": normalized_market_relative,
        "sector_relative": normalized_sector_relative,
        "inverse_volatility": -normalized_volatility,
        "inverse_downside": -normalized_downside,
        "inverse_drawdown": -normalized_drawdown,
    }


def normalize_weight_map(weight_map: Dict[str, float]) -> Dict[str, float]:
    ordered = {name: max(0.0, float(weight_map.get(name, 0.0))) for name in TECHNICAL_FEATURE_ORDER}
    total = sum(ordered.values())
    if total <= 1e-9:
        fallback_total = sum(DEFAULT_TECHNICAL_FEATURE_WEIGHTS.values())
        return {
            name: DEFAULT_TECHNICAL_FEATURE_WEIGHTS[name] / fallback_total
            for name in TECHNICAL_FEATURE_ORDER
        }
    return {name: value / total for name, value in ordered.items()}


def default_model_calibration() -> ModelCalibration:
    return ModelCalibration(
        technical_weights=normalize_weight_map(DEFAULT_TECHNICAL_FEATURE_WEIGHTS),
        block_weights=dict(BLOCK_BASE_PRIORS),
        technical_scale=TECHNICAL_SCORE_TARGET_SCALE,
        training_row_count=0,
        training_ic=0.0,
        block_row_count=0,
        source="default_priors",
    )


def build_window_price_series(
    close_window: Sequence[float],
    volume_window: Sequence[float],
    end_date: pd.Timestamp,
) -> PriceSeries:
    return PriceSeries(
        closes=list(close_window)[::-1],
        volumes=list(volume_window)[::-1],
        latest_trading_date=end_date.date().isoformat(),
        observations=len(close_window),
    )


def build_calibration_training_rows(
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
        if len(merged) < 45:
            continue

        merged = merged.tail(CALIBRATION_LOOKBACK_DAYS).reset_index(drop=True)
        for end_index in range(20, len(merged) - 5, CALIBRATION_STEP_DAYS):
            window = merged.iloc[end_index - 20:end_index + 1]
            if len(window) < 21:
                continue
            future_index = end_index + 5
            future_row = merged.iloc[future_index]
            current_row = merged.iloc[end_index]

            stock_series = build_window_price_series(
                close_window=window["Close"].tolist(),
                volume_window=window["Volume"].tolist(),
                end_date=current_row["Date"],
            )
            market_series = build_window_price_series(
                close_window=window["MarketClose"].tolist(),
                volume_window=[0.0] * len(window),
                end_date=current_row["Date"],
            )
            sector_series = build_window_price_series(
                close_window=window["SectorClose"].tolist(),
                volume_window=[0.0] * len(window),
                end_date=current_row["Date"],
            )

            stock_metrics = compute_technical_metrics(stock_series)
            market_metrics = compute_technical_metrics(market_series)
            sector_metrics = compute_technical_metrics(sector_series)
            relative_metrics = compute_relative_strength_metrics(stock_metrics, market_metrics, sector_metrics)
            technical_features = build_normalized_technical_features(
                momentum=stock_metrics["momentum_5d"],
                medium_momentum=stock_metrics["momentum_20d"],
                trend_gap=stock_metrics["trend_gap"],
                positive_day_ratio=stock_metrics["positive_day_ratio"],
                volume_trend=stock_metrics["volume_trend"],
                volatility=stock_metrics["volatility"],
                downside_risk=stock_metrics["downside_volatility"],
                max_drawdown=stock_metrics["max_drawdown"],
                market_relative_5d=relative_metrics["market_relative_5d"],
                market_relative_20d=relative_metrics["market_relative_20d"],
                sector_relative_5d=relative_metrics["sector_relative_5d"],
                sector_relative_20d=relative_metrics["sector_relative_20d"],
            )

            stock_forward_return = safe_divide(float(future_row["Close"]), float(current_row["Close"]), 1.0) - 1.0
            market_forward_return = safe_divide(float(future_row["MarketClose"]), float(current_row["MarketClose"]), 1.0) - 1.0
            forward_excess_return = clamp(
                relative_return(stock_forward_return, market_forward_return),
                -0.25,
                0.25,
            )
            feature_vector = np.array([technical_features[name] for name in TECHNICAL_FEATURE_ORDER], dtype=float)
            training_rows.append((feature_vector, forward_excess_return))

    return training_rows


def information_coefficient(predictions: np.ndarray, targets: np.ndarray) -> float:
    if len(predictions) < 3 or len(targets) < 3:
        return 0.0
    pred_std = float(np.std(predictions))
    target_std = float(np.std(targets))
    if pred_std <= 1e-9 or target_std <= 1e-9:
        return 0.0
    correlation = float(np.corrcoef(predictions, targets)[0, 1])
    if math.isnan(correlation):
        return 0.0
    return correlation


def build_block_weight_sample(entry: Dict[str, Any]) -> Optional[Tuple[np.ndarray, float]]:
    diagnostics = entry.get("diagnostics")
    realized = entry.get("realized_5d_excess_return")
    if not isinstance(diagnostics, dict) or not isinstance(realized, (int, float)):
        return None

    news_signal = diagnostics.get("news_signal_input")
    macro_signal = diagnostics.get("macro_signal_input")
    sector_signal = diagnostics.get("sector_signal_input")
    if not all(isinstance(value, (int, float)) for value in (news_signal, macro_signal, sector_signal)):
        return None

    return (
        np.array([float(news_signal), float(macro_signal), float(sector_signal)], dtype=float),
        clamp(float(realized), -0.25, 0.25),
    )


def calibrate_block_weights(history_entries: List[Dict[str, Any]]) -> Tuple[Dict[str, float], int, str]:
    samples = [sample for sample in (build_block_weight_sample(entry) for entry in history_entries) if sample is not None]
    if len(samples) < BLOCK_CALIBRATION_MIN_ROWS:
        return dict(BLOCK_BASE_PRIORS), len(samples), "block_priors"

    x = np.vstack([sample[0] for sample in samples])
    y = np.array([sample[1] for sample in samples], dtype=float)
    ridge_alpha = 0.35
    beta = np.linalg.solve(x.T @ x + ridge_alpha * np.eye(x.shape[1]), x.T @ y)
    beta = np.clip(beta, 0.0, None)
    if float(beta.sum()) <= 1e-9:
        return dict(BLOCK_BASE_PRIORS), len(samples), "block_priors"

    budget = sum(BLOCK_BASE_PRIORS.values())
    weights = {
        "news": clamp(budget * float(beta[0] / beta.sum()), BLOCK_WEIGHT_FLOORS["news"], BLOCK_WEIGHT_CAPS["news"]),
        "macro": clamp(budget * float(beta[1] / beta.sum()), BLOCK_WEIGHT_FLOORS["macro"], BLOCK_WEIGHT_CAPS["macro"]),
        "sector": clamp(budget * float(beta[2] / beta.sum()), BLOCK_WEIGHT_FLOORS["sector"], BLOCK_WEIGHT_CAPS["sector"]),
    }
    return weights, len(samples), "history_realized_returns"


def build_model_calibration(
    universe: List[Tuple[str, str]],
    sector_map: Dict[str, str],
    history_entries: List[Dict[str, Any]],
) -> ModelCalibration:
    technical_calibration = default_model_calibration()
    source = technical_calibration.source
    try:
        training_rows = build_calibration_training_rows(universe, sector_map)
    except Exception as exc:
        print(f"[WARN] Technical calibration failed, using default priors: {exc}")
        training_rows = []

    if len(training_rows) >= CALIBRATION_MIN_ROWS:
        x = np.vstack([row[0] for row in training_rows])
        y = np.array([row[1] for row in training_rows], dtype=float)
        beta = np.linalg.solve(x.T @ x + CALIBRATION_RIDGE_ALPHA * np.eye(x.shape[1]), x.T @ y)
        beta = np.clip(beta, 0.0, None)
        if float(beta.sum()) > 1e-9:
            normalized_weights = normalize_weight_map(
                {name: float(value) for name, value in zip(TECHNICAL_FEATURE_ORDER, beta.tolist())}
            )
            raw_predictions = x @ np.array([normalized_weights[name] for name in TECHNICAL_FEATURE_ORDER], dtype=float)
            robust_scale = float(np.quantile(np.abs(raw_predictions), CALIBRATION_PREDICTION_PCTL)) if len(raw_predictions) else 0.0
            technical_scale = clamp(
                TECHNICAL_SCORE_TARGET_SCALE / max(robust_scale, 1e-6),
                0.12,
                0.45,
            )
            technical_calibration = ModelCalibration(
                technical_weights=normalized_weights,
                block_weights=dict(BLOCK_BASE_PRIORS),
                technical_scale=technical_scale,
                training_row_count=len(training_rows),
                training_ic=information_coefficient(raw_predictions, y),
                block_row_count=0,
                source="price_history_5d_excess_return",
            )
            source = technical_calibration.source

    block_weights, block_row_count, block_source = calibrate_block_weights(history_entries)
    if technical_calibration.source == "default_priors" and block_source == "block_priors":
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


def compute_technical_contributions(
    technical_features: Dict[str, float],
    calibration: ModelCalibration,
) -> Dict[str, Any]:
    feature_weights = calibration.technical_weights
    raw_feature_contributions = {
        name: technical_features[name] * feature_weights.get(name, 0.0)
        for name in TECHNICAL_FEATURE_ORDER
    }
    group_raw = {
        group_name: sum(raw_feature_contributions[name] for name in feature_names)
        for group_name, feature_names in TECHNICAL_GROUPS.items()
    }
    group_weighted = {
        group_name: contribution * calibration.technical_scale
        for group_name, contribution in group_raw.items()
    }
    technical_total = sum(group_weighted.values())
    return {
        "feature_values": technical_features,
        "feature_weights": feature_weights,
        "feature_contributions": raw_feature_contributions,
        "group_raw": group_raw,
        "group_weighted": group_weighted,
        "technical_total": technical_total,
    }


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
    macro_score: float = 0.5,
    macro_confidence: float = 0.2,
    sector_score: float = 0.5,
    sector_confidence: float = 0.2,
    market_relative_5d: float = 0.0,
    market_relative_20d: float = 0.0,
    sector_relative_5d: float = 0.0,
    sector_relative_20d: float = 0.0,
    calibration: Optional[ModelCalibration] = None,
    layer_penalties: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    active_calibration = calibration or default_model_calibration()
    active_layer_penalties = layer_penalties or {
        "macro_penalty": 1.0,
        "sector_penalty": 1.0,
        "news_macro_overlap": 0.0,
        "news_sector_overlap": 0.0,
        "macro_sector_overlap": 0.0,
    }
    medium_momentum = momentum if momentum_20d is None else momentum_20d
    downside_risk = volatility if downside_volatility is None else downside_volatility

    normalized_features = build_normalized_technical_features(
        momentum=momentum,
        medium_momentum=medium_momentum,
        trend_gap=trend_gap,
        positive_day_ratio=positive_day_ratio,
        volume_trend=volume_trend,
        volatility=volatility,
        downside_risk=downside_risk,
        max_drawdown=max_drawdown,
        market_relative_5d=market_relative_5d,
        market_relative_20d=market_relative_20d,
        sector_relative_5d=sector_relative_5d,
        sector_relative_20d=sector_relative_20d,
    )
    centered_news = clamp((news_score - 0.5) * 2.0, -1.0, 1.0)
    centered_macro = clamp((macro_score - 0.5) * 2.0, -1.0, 1.0)
    centered_sector = clamp((sector_score - 0.5) * 2.0, -1.0, 1.0)

    technical_parts = compute_technical_contributions(normalized_features, active_calibration)
    trend_strength = technical_parts["group_weighted"]["trend_strength"]
    relative_strength = technical_parts["group_weighted"]["relative_strength"]
    participation = technical_parts["group_weighted"]["participation"]
    risk_control = technical_parts["group_weighted"]["risk_control"]
    technical_total = technical_parts["technical_total"]

    weighted_short_momentum = technical_parts["feature_contributions"]["short_momentum"] * active_calibration.technical_scale
    weighted_medium_momentum = technical_parts["feature_contributions"]["medium_momentum"] * active_calibration.technical_scale
    weighted_trend_gap = technical_parts["feature_contributions"]["trend_gap"] * active_calibration.technical_scale
    weighted_positive_ratio = technical_parts["feature_contributions"]["positive_ratio"] * active_calibration.technical_scale
    weighted_volume_confirmation = technical_parts["feature_contributions"]["volume_confirmation"] * active_calibration.technical_scale
    weighted_market_relative = technical_parts["feature_contributions"]["market_relative"] * active_calibration.technical_scale
    weighted_sector_relative = technical_parts["feature_contributions"]["sector_relative"] * active_calibration.technical_scale
    weighted_volatility_penalty = technical_parts["feature_contributions"]["inverse_volatility"] * active_calibration.technical_scale
    weighted_downside_penalty = technical_parts["feature_contributions"]["inverse_downside"] * active_calibration.technical_scale
    weighted_drawdown_penalty = technical_parts["feature_contributions"]["inverse_drawdown"] * active_calibration.technical_scale

    news_multiplier = active_calibration.block_weights["news"] * (0.45 + 0.55 * clamp(news_confidence, 0.0, 1.0))
    macro_multiplier = (
        active_calibration.block_weights["macro"]
        * (0.30 + 0.70 * clamp(macro_confidence, 0.0, 1.0))
        * float(active_layer_penalties["macro_penalty"])
    )
    sector_multiplier = (
        active_calibration.block_weights["sector"]
        * (0.28 + 0.72 * clamp(sector_confidence, 0.0, 1.0))
        * float(active_layer_penalties["sector_penalty"])
    )

    weighted_news = centered_news * news_multiplier
    weighted_macro = centered_macro * macro_multiplier
    weighted_sector = centered_sector * sector_multiplier

    momentum_total = trend_strength + relative_strength + participation
    volatility_penalty_total = risk_control
    technical_direction = clamp(technical_total / 0.30, -1.0, 1.0)
    alignment_signal = technical_direction * centered_news if abs(centered_news) >= 0.10 else 0.0
    weighted_signal_alignment = alignment_signal * (
        SIGNAL_ALIGNMENT_BASE
        + SIGNAL_ALIGNMENT_CONFIDENCE_SCALE * clamp(news_confidence, 0.0, 1.0)
    )
    total_score = technical_total + weighted_news + weighted_macro + weighted_sector + weighted_signal_alignment

    return {
        "normalized_short_momentum": normalized_features["short_momentum"],
        "normalized_medium_momentum": normalized_features["medium_momentum"],
        "normalized_trend_gap": normalized_features["trend_gap"],
        "normalized_positive_ratio": normalized_features["positive_ratio"],
        "normalized_volume": normalized_features["volume_confirmation"],
        "normalized_market_relative": normalized_features["market_relative"],
        "normalized_sector_relative": normalized_features["sector_relative"],
        "normalized_volatility": -normalized_features["inverse_volatility"],
        "normalized_downside": -normalized_features["inverse_downside"],
        "normalized_drawdown": -normalized_features["inverse_drawdown"],
        "centered_news": centered_news,
        "centered_macro": centered_macro,
        "centered_sector": centered_sector,
        "weighted_short_momentum": weighted_short_momentum,
        "weighted_medium_momentum": weighted_medium_momentum,
        "weighted_trend_quality": weighted_trend_gap + weighted_positive_ratio,
        "weighted_volume_confirmation": weighted_volume_confirmation,
        "weighted_market_relative": weighted_market_relative,
        "weighted_sector_relative": weighted_sector_relative,
        "weighted_volatility_penalty": weighted_volatility_penalty,
        "weighted_downside_penalty": weighted_downside_penalty,
        "weighted_drawdown_penalty": weighted_drawdown_penalty,
        "weighted_news": weighted_news,
        "weighted_macro": weighted_macro,
        "weighted_sector": weighted_sector,
        "alignment_signal": alignment_signal,
        "weighted_signal_alignment": weighted_signal_alignment,
        "momentum_total": momentum_total,
        "volatility_penalty_total": volatility_penalty_total,
        "technical_total": technical_total,
        "technical_trend_strength": trend_strength,
        "technical_relative_strength": relative_strength,
        "technical_participation": participation,
        "technical_risk_control": risk_control,
        "technical_scale": active_calibration.technical_scale,
        "technical_training_rows": active_calibration.training_row_count,
        "technical_training_ic": active_calibration.training_ic,
        "block_weight_news": active_calibration.block_weights["news"],
        "block_weight_macro": active_calibration.block_weights["macro"],
        "block_weight_sector": active_calibration.block_weights["sector"],
        "macro_dedup_penalty": float(active_layer_penalties["macro_penalty"]),
        "sector_dedup_penalty": float(active_layer_penalties["sector_penalty"]),
        "news_macro_overlap": float(active_layer_penalties["news_macro_overlap"]),
        "news_sector_overlap": float(active_layer_penalties["news_sector_overlap"]),
        "macro_sector_overlap": float(active_layer_penalties["macro_sector_overlap"]),
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
    macro_score: float = 0.5,
    macro_confidence: float = 0.2,
    sector_score: float = 0.5,
    sector_confidence: float = 0.2,
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
    macro_confidence_signal = clamp(macro_confidence, 0.0, 1.0) * 0.04
    macro_conviction_signal = clamp(abs(macro_score - 0.5) * 2.0, 0.0, 1.0) * 0.04
    sector_confidence_signal = clamp(sector_confidence, 0.0, 1.0) * 0.04
    sector_conviction_signal = clamp(abs(sector_score - 0.5) * 2.0, 0.0, 1.0) * 0.03
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
        + macro_confidence_signal
        + macro_conviction_signal
        + sector_confidence_signal
        + sector_conviction_signal
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
    macro_age_days = None
    if candidate.macro_as_of:
        macro_age_days = market_day_age(date.fromisoformat(candidate.macro_as_of), reference)
    sector_age_days = None
    if candidate.sector_as_of:
        sector_age_days = market_day_age(date.fromisoformat(candidate.sector_as_of), reference)

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

    if candidate.macro_confidence < 0.35 or abs(candidate.macro_score - 0.5) < 0.05:
        macro_state = "positive"
        macro_detail = "No strong world-news headwind is currently working against the setup."
    elif candidate.macro_score < 0.45 and candidate.macro_confidence >= 0.55:
        macro_state = "risk"
        macro_detail = "Recent world news leaned negative for this company with enough confidence to matter."
    elif candidate.macro_score >= 0.58:
        macro_state = "positive"
        macro_detail = "Recent world news added a supportive company-level overlay."
    else:
        macro_state = "watch"
        macro_detail = "The global overlay is present, but not yet decisive."

    if score_margin < 0.015 or confidence_margin < 0.03:
        margin_state = "risk"
        margin_detail = "The pick only barely cleared the release bar."
    elif score_margin < 0.04 or confidence_margin < 0.08:
        margin_state = "watch"
        margin_detail = "The pick still qualified, but the margin of safety is not wide."
    else:
        margin_state = "positive"
        margin_detail = "The pick cleared the release thresholds with room to spare."

    macro_is_material = candidate.macro_confidence >= 0.35 and abs(candidate.macro_score - 0.5) >= 0.05
    sector_is_material = candidate.sector_confidence >= 0.35 and abs(candidate.sector_score - 0.5) >= 0.05

    freshness_risks: List[str] = []
    freshness_watches: List[str] = []
    if price_age_days > 2:
        freshness_risks.append(f"price data is {price_age_days} trading days old")
    elif price_age_days > 1:
        freshness_watches.append(f"price data is {price_age_days} trading days old")

    if news_age_days is not None:
        if news_age_days > 3:
            freshness_risks.append(f"company news is {news_age_days} trading days old")
        elif news_age_days > 2:
            freshness_watches.append(f"company news is {news_age_days} trading days old")

    if macro_is_material:
        if macro_age_days is None:
            freshness_watches.append("macro overlay has no supporting timestamp")
        elif macro_age_days > 3:
            freshness_risks.append(f"macro overlay is {macro_age_days} trading days old")
        elif macro_age_days > 2:
            freshness_watches.append(f"macro overlay is {macro_age_days} trading days old")

    if sector_is_material:
        if sector_age_days is None:
            freshness_watches.append("sector overlay has no supporting timestamp")
        elif sector_age_days > 3:
            freshness_risks.append(f"sector overlay is {sector_age_days} trading days old")
        elif sector_age_days > 2:
            freshness_watches.append(f"sector overlay is {sector_age_days} trading days old")

    if freshness_risks:
        freshness_state = "risk"
        freshness_detail = "Some of the supporting data is getting old enough to weaken monitoring quality: " + ", ".join(freshness_risks) + "."
    elif freshness_watches:
        freshness_state = "watch"
        freshness_detail = "Freshness should be checked before acting: " + ", ".join(freshness_watches) + "."
    else:
        freshness_state = "positive"
        freshness_detail = "Price, company-news, and active overlay timestamps are recent enough for a live monitoring read."

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
            "label": "Macro",
            "state": macro_state,
            "value": f"{candidate.macro_score:.2f} overlay • {candidate.macro_confidence:.2f} conf",
            "detail": macro_detail,
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
            "value": " • ".join(
                [
                    format_market_age("price", price_age_days),
                    format_market_age("news", news_age_days) if news_age_days is not None else "news no fresh signal",
                    format_market_age("macro", macro_age_days) if macro_is_material else "macro neutral",
                    format_market_age("sector", sector_age_days) if sector_is_material else "sector neutral",
                ]
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
    sector: str,
    news_info: Optional[dict] = None,
    sector_scores_payload: Optional[Dict[str, Any]] = None,
    calibration: Optional[ModelCalibration] = None,
) -> StockCandidate:
    price_series = fetch_price_series_cached(symbol)
    technical_metrics = compute_technical_metrics(price_series)
    market_metrics = compute_technical_metrics(fetch_price_series_cached(MARKET_BENCHMARK_SYMBOL))
    sector_benchmark_symbol = SECTOR_ETF_BY_SECTOR.get(sector)
    if not sector_benchmark_symbol:
        raise RuntimeError(f"No sector benchmark configured for sector {sector}")
    sector_metrics = compute_technical_metrics(fetch_price_series_cached(sector_benchmark_symbol))
    relative_strength_metrics = compute_relative_strength_metrics(
        technical_metrics,
        market_metrics,
        sector_metrics,
    )
    news_snapshot = build_news_snapshot(news_info)
    macro_snapshot = build_macro_snapshot(symbol, company_name, sector, sector_scores_payload)
    sector_snapshot = build_sector_snapshot(sector, sector_scores_payload)
    layer_penalties = dedupe_layer_penalties(news_snapshot, macro_snapshot, sector_snapshot)
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
        macro_score=macro_snapshot.macro_score,
        macro_confidence=macro_snapshot.macro_confidence,
        sector_score=sector_snapshot.sector_score,
        sector_confidence=sector_snapshot.sector_confidence,
        market_relative_5d=relative_strength_metrics["market_relative_5d"],
        market_relative_20d=relative_strength_metrics["market_relative_20d"],
        sector_relative_5d=relative_strength_metrics["sector_relative_5d"],
        sector_relative_20d=relative_strength_metrics["sector_relative_20d"],
        calibration=calibration,
        layer_penalties=layer_penalties,
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
        macro_score=macro_snapshot.macro_score,
        macro_confidence=macro_snapshot.macro_confidence,
        sector_score=sector_snapshot.sector_score,
        sector_confidence=sector_snapshot.sector_confidence,
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
        (
            f"Relative strength is {format_pct(relative_strength_metrics['market_relative_5d'])} versus SPY over 5 days "
            f"and {format_pct(relative_strength_metrics['sector_relative_5d'])} versus the sector ETF."
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

    if macro_snapshot.macro_confidence >= 0.30 and abs(macro_snapshot.macro_score - 0.5) >= 0.05:
        reasons.append(
            (
                f"Broad world news leaned {macro_snapshot.direction} for {company_name} with score "
                f"{macro_snapshot.macro_score:.2f} and confidence {macro_snapshot.macro_confidence:.2f}."
            )
        )
        reasons.extend(macro_snapshot.reasons[:1])
        if layer_penalties["macro_penalty"] < 0.95:
            reasons.append(
                f"Macro impact was trimmed by {round((1.0 - layer_penalties['macro_penalty']) * 100):.0f}% because it overlaps with company-news themes."
            )

    if sector_snapshot.sector_confidence >= 0.30 and abs(sector_snapshot.sector_score - 0.5) >= 0.05:
        reasons.append(
            (
                f"Broad market news leaned {sector_snapshot.direction} for the "
                f"{sector_display_name(sector_snapshot.sector)} sector with score "
                f"{sector_snapshot.sector_score:.2f} and confidence {sector_snapshot.sector_confidence:.2f}."
            )
        )
        reasons.extend(sector_snapshot.reasons[:1])
        if layer_penalties["sector_penalty"] < 0.95:
            reasons.append(
                f"Sector overlay was trimmed by {round((1.0 - layer_penalties['sector_penalty']) * 100):.0f}% because it overlaps with company or macro themes."
            )

    if calibration is not None and calibration.training_row_count >= CALIBRATION_MIN_ROWS:
        reasons.append(
            f"Technical weights were calibrated on {calibration.training_row_count} historical symbol-weeks against 5-day excess return (IC {calibration.training_ic:.2f})."
        )

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
        market_relative_5d=relative_strength_metrics["market_relative_5d"],
        market_relative_20d=relative_strength_metrics["market_relative_20d"],
        sector_relative_5d=relative_strength_metrics["sector_relative_5d"],
        sector_relative_20d=relative_strength_metrics["sector_relative_20d"],
        news_score=news_snapshot.news_score,
        news_confidence=news_snapshot.news_confidence,
        macro_score=macro_snapshot.macro_score,
        macro_confidence=macro_snapshot.macro_confidence,
        raw_sentiment=news_snapshot.raw_sentiment,
        calibrated_sentiment=news_snapshot.calibrated_sentiment,
        dominant_signal=news_snapshot.dominant_signal,
        score_breakdown=score_breakdown,
        news_evidence=news_snapshot.news_evidence,
        macro_evidence=macro_snapshot.supporting_articles,
        macro_as_of=macro_snapshot.last_updated,
        sector_as_of=sector_snapshot.last_updated,
        sector=sector_snapshot.sector,
        sector_score=sector_snapshot.sector_score,
        sector_confidence=sector_snapshot.sector_confidence,
        sector_direction=sector_snapshot.direction,
        sector_reasons=sector_snapshot.reasons,
    )


def get_candidates() -> Tuple[List[StockCandidate], GenerationStats]:
    universe_path = resolve_universe_csv_path()
    universe = load_universe(universe_path)
    news_scores = load_news_scores()
    sector_map = load_sector_map(universe_path)
    sector_scores = load_sector_scores()
    history_entries = [normalize_history_entry(entry) for entry in load_existing_history_entries(HISTORY_PATH)]
    calibration = build_model_calibration(universe, sector_map, history_entries)

    candidates: List[StockCandidate] = []
    skipped_details: List[Dict[str, str]] = []
    for symbol, company_name in universe:
        try:
            candidate = build_candidate(
                symbol=symbol,
                company_name=company_name,
                sector=sector_map[symbol],
                news_info=news_scores.get(symbol),
                sector_scores_payload=sector_scores,
                calibration=calibration,
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
            "momentum_5d": round(candidate.momentum_5d, 4),
            "momentum_20d": round(candidate.momentum_20d, 4),
            "daily_volatility": round(candidate.volatility, 4),
            "downside_volatility": round(candidate.downside_volatility, 4),
            "max_drawdown_20d": round(candidate.max_drawdown, 4),
            "positive_day_ratio_10d": round(candidate.positive_day_ratio, 4),
            "price_vs_20d_average": round(candidate.trend_gap, 4),
            "volume_trend_5d": round(candidate.volume_trend, 4),
            "market_relative_5d": round(candidate.market_relative_5d, 4),
            "market_relative_20d": round(candidate.market_relative_20d, 4),
            "sector_relative_5d": round(candidate.sector_relative_5d, 4),
            "sector_relative_20d": round(candidate.sector_relative_20d, 4),
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
        },
        "score_breakdown": {
            "momentum": round(candidate.score_breakdown["momentum_total"], 4),
            "short_momentum": round(candidate.score_breakdown["weighted_short_momentum"], 4),
            "medium_momentum": round(candidate.score_breakdown["weighted_medium_momentum"], 4),
            "trend_quality": round(candidate.score_breakdown["weighted_trend_quality"], 4),
            "volume_confirmation": round(candidate.score_breakdown["weighted_volume_confirmation"], 4),
            "market_relative_strength": round(candidate.score_breakdown["weighted_market_relative"], 4),
            "sector_relative_strength": round(candidate.score_breakdown["weighted_sector_relative"], 4),
            "volatility_penalty": round(candidate.score_breakdown["volatility_penalty_total"], 4),
            "daily_volatility_penalty": round(candidate.score_breakdown["weighted_volatility_penalty"], 4),
            "downside_penalty": round(candidate.score_breakdown["weighted_downside_penalty"], 4),
            "drawdown_penalty": round(candidate.score_breakdown["weighted_drawdown_penalty"], 4),
            "news_adjustment": round(candidate.score_breakdown["weighted_news"], 4),
            "macro_adjustment": round(candidate.score_breakdown["weighted_macro"], 4),
            "sector_adjustment": round(candidate.score_breakdown["weighted_sector"], 4),
            "signal_alignment": round(candidate.score_breakdown["weighted_signal_alignment"], 4),
            "technical_total": round(candidate.score_breakdown["technical_total"], 4),
            "trend_strength": round(candidate.score_breakdown.get("technical_trend_strength", 0.0), 4),
            "relative_strength": round(candidate.score_breakdown.get("technical_relative_strength", 0.0), 4),
            "participation": round(candidate.score_breakdown.get("technical_participation", 0.0), 4),
            "risk_control": round(candidate.score_breakdown.get("technical_risk_control", 0.0), 4),
            "block_weight_news": round(candidate.score_breakdown.get("block_weight_news", 0.0), 4),
            "block_weight_macro": round(candidate.score_breakdown.get("block_weight_macro", 0.0), 4),
            "block_weight_sector": round(candidate.score_breakdown.get("block_weight_sector", 0.0), 4),
            "macro_dedup_penalty": round(candidate.score_breakdown.get("macro_dedup_penalty", 1.0), 4),
            "sector_dedup_penalty": round(candidate.score_breakdown.get("sector_dedup_penalty", 1.0), 4),
            "news_macro_overlap": round(candidate.score_breakdown.get("news_macro_overlap", 0.0), 4),
            "news_sector_overlap": round(candidate.score_breakdown.get("news_sector_overlap", 0.0), 4),
            "macro_sector_overlap": round(candidate.score_breakdown.get("macro_sector_overlap", 0.0), 4),
            "total": round(candidate.score_breakdown["total"], 4),
        },
        "reasons": candidate.reasons,
        "news_evidence": candidate.news_evidence,
        "macro_evidence": candidate.macro_evidence,
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
        for item in [
            candidate.price_as_of,
            candidate.news_as_of,
            candidate.macro_as_of,
            candidate.sector_as_of,
        ]
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
        "sector": entry.get("sector"),
        "risk": entry.get("risk"),
        "model_score": entry.get("model_score", entry.get("score")),
        "confidence_score": entry.get("confidence_score"),
        "confidence_label": entry.get("confidence_label"),
        "data_as_of": entry.get("data_as_of"),
        "model_version": entry.get("model_version", MODEL_VERSION),
        "diagnostics": entry.get("diagnostics") if isinstance(entry.get("diagnostics"), dict) else None,
        "realized_5d_return": entry.get("realized_5d_return"),
        "realized_5d_excess_return": entry.get("realized_5d_excess_return"),
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
        "sector": pick.sector if pick else None,
        "risk": pick.risk_level if pick else None,
        "model_score": round(pick.total_score, 4) if pick else None,
        "confidence_score": round(pick.confidence_score, 2) if pick else None,
        "confidence_label": pick.confidence_label if pick else None,
        "data_as_of": data_as_of,
        "model_version": MODEL_VERSION,
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
                "news_macro_overlap": round(float(pick.score_breakdown.get("news_macro_overlap", 0.0)), 4),
                "news_sector_overlap": round(float(pick.score_breakdown.get("news_sector_overlap", 0.0)), 4),
                "macro_sector_overlap": round(float(pick.score_breakdown.get("macro_sector_overlap", 0.0)), 4),
            }
            if pick
            else None
        ),
    }


def realized_forward_return(
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
    future_index = anchor_index + 5
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
    benchmark_future_index = benchmark_index + 5
    if benchmark_future_index >= len(benchmark_frame):
        return stock_return, None

    benchmark_start = float(benchmark_frame.iloc[benchmark_index]["Close"])
    benchmark_end = float(benchmark_frame.iloc[benchmark_future_index]["Close"])
    benchmark_return = safe_divide(benchmark_end, benchmark_start, 1.0) - 1.0
    return stock_return, relative_return(stock_return, benchmark_return)


def enrich_history_realized_returns(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not history_realized_enrichment_enabled():
        return entries

    enriched: List[Dict[str, Any]] = []
    for entry in entries:
        normalized = dict(entry)
        symbol = normalized.get("symbol")
        week_end = normalized.get("week_end")
        if isinstance(symbol, str) and isinstance(week_end, str):
            realized_return, realized_excess = realized_forward_return(symbol, week_end, MARKET_BENCHMARK_SYMBOL)
            if realized_return is not None:
                normalized["realized_5d_return"] = round(realized_return, 4)
            if realized_excess is not None:
                normalized["realized_5d_excess_return"] = round(realized_excess, 4)
        enriched.append(normalized)
    return enriched


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
    filtered_entries = enrich_history_realized_returns(filtered_entries)

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
