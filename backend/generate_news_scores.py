import json
import math
import os
import re
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

try:
    from backend.generate_pick import (
        HISTORY_PATH,
        MARKET_BENCHMARK_SYMBOL,
        RISK_PICKS_PATH,
        SECTOR_ETF_BY_SECTOR,
        compute_technical_metrics,
        compute_relative_strength_metrics,
        fetch_price_series_cached,
        iso_utc,
        load_existing_history_entries,
        normalize_history_entry,
    )
    from backend.llm_utils import (
        allow_llm_fallback,
        openai_api_key,
        openai_model,
        request_structured_response,
    )
    from backend.pipeline_runtime import (
        attach_data_quality,
        consume_provider_budget,
        record_runtime_event,
        set_pipeline_scope,
    )
    from backend.sector_utils import load_sector_map
except ImportError:
    from generate_pick import (
        HISTORY_PATH,
        MARKET_BENCHMARK_SYMBOL,
        RISK_PICKS_PATH,
        SECTOR_ETF_BY_SECTOR,
        compute_technical_metrics,
        compute_relative_strength_metrics,
        fetch_price_series_cached,
        iso_utc,
        load_existing_history_entries,
        normalize_history_entry,
    )
    from llm_utils import (
        allow_llm_fallback,
        openai_api_key,
        openai_model,
        request_structured_response,
    )
    from pipeline_runtime import (
        attach_data_quality,
        consume_provider_budget,
        record_runtime_event,
        set_pipeline_scope,
    )
    from sector_utils import load_sector_map

NEWS_SCORES_PATH = "news_scores.json"
UNIVERSE_CSV_PATH_ENV = "UNIVERSE_CSV_PATH"
DEFAULT_UNIVERSE_CSV_PATH = "universe.csv"
MARKETAUX_NEWS_ENDPOINT = "https://api.marketaux.com/v1/news/all"
ALPHA_VANTAGE_NEWS_ENDPOINT = "https://www.alphavantage.co/query"
GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

MARKETAUX_API_TOKEN_ENV = "MARKETAUX_API_TOKEN"
ALPHA_VANTAGE_API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"
ALLOW_MARKETAUX_FALLBACK_ENV = "ALLOW_MARKETAUX_FALLBACK"
MARKETAUX_FIXTURE_PATH_ENV = "MARKETAUX_FIXTURE_PATH"
MARKETAUX_NEWS_LIMIT_ENV = "MARKETAUX_NEWS_LIMIT"
MARKETAUX_NEWS_LOOKBACK_DAYS_ENV = "MARKETAUX_NEWS_LOOKBACK_DAYS"
MARKETAUX_REQUEST_INTERVAL_SECONDS_ENV = "MARKETAUX_REQUEST_INTERVAL_SECONDS"
ALPHA_VANTAGE_COMPANY_NEWS_LIMIT_ENV = "ALPHA_VANTAGE_COMPANY_NEWS_LIMIT"
ALPHA_VANTAGE_COMPANY_REQUEST_INTERVAL_SECONDS_ENV = "ALPHA_VANTAGE_COMPANY_REQUEST_INTERVAL_SECONDS"
ALPHA_VANTAGE_MAX_COMPANY_REQUESTS_ENV = "ALPHA_VANTAGE_MAX_COMPANY_REQUESTS"
GDELT_COMPANY_NEWS_LIMIT_ENV = "GDELT_COMPANY_NEWS_LIMIT"
COMPANY_NEWS_PROVIDER_MODE_ENV = "COMPANY_NEWS_PROVIDER_MODE"
ENABLE_COMPANY_LLM_REVIEW_ENV = "ENABLE_COMPANY_LLM_REVIEW"
COMPANY_NEWS_SHORTLIST_SIZE_ENV = "COMPANY_NEWS_SHORTLIST_SIZE"
MARKETAUX_DAILY_REQUEST_LIMIT = 100
ALPHA_VANTAGE_DAILY_REQUEST_LIMIT = 25
MARKETAUX_DEFAULT_NEWS_LIMIT = 3
MARKETAUX_MAX_NEWS_LIMIT = 3
MARKETAUX_DEFAULT_REQUEST_INTERVAL_SECONDS = 0.5
ALPHA_VANTAGE_DEFAULT_COMPANY_NEWS_LIMIT = 6
ALPHA_VANTAGE_MAX_COMPANY_NEWS_LIMIT = 10
ALPHA_VANTAGE_DEFAULT_COMPANY_REQUEST_INTERVAL_SECONDS = 12.0
ALPHA_VANTAGE_DEFAULT_MAX_COMPANY_REQUESTS = 8
GDELT_DEFAULT_COMPANY_NEWS_LIMIT = 8
GDELT_MAX_COMPANY_NEWS_LIMIT = 10
MARKETAUX_REQUEST_TIMEOUT_SECONDS = 20
COMPANY_LLM_ARTICLE_LIMIT_ENV = "COMPANY_LLM_ARTICLE_LIMIT"
COMPANY_PRIMARY_MARKETAUX_TARGET = 2
COMPANY_ALPHA_VANTAGE_ESCALATION_FLOOR = 1
COMPANY_LLM_LOW_QUALITY_SHARE_THRESHOLD = 0.67
COMPANY_LLM_MIN_DIRECTNESS = 0.45
COMPANY_HEURISTIC_LOW_QUALITY_SHARE_THRESHOLD = 0.75
COMPANY_HEURISTIC_LOW_QUALITY_RELEVANCE_CAP = 0.72
LOW_QUALITY_COMPANY_NEWS_HINTS = (
    "stocks to buy",
    "top stocks",
    "best stocks",
    "watchlist",
    "market recap",
    "market roundup",
    "wall street",
    "s&p 500",
    "dow jones",
    "nasdaq",
    "etf",
    "analyst picks",
    "price prediction",
    "these stocks",
    "stock market",
)

NEWS_LOOKBACK_DAYS = 5
MAX_ARTICLES_PER_SYMBOL = 10
MIN_RELEVANCE_SCORE = 0.45
RECENCY_HALFLIFE_HOURS = 36.0
NEWS_FETCH_ATTEMPTS = 3
NEWS_FETCH_RETRY_SECONDS = 1.5
DEFAULT_COMPANY_NEWS_SHORTLIST_SIZE = 15

POSITIVE_HINTS = (
    "beats earnings",
    "raised guidance",
    "raises guidance",
    "strong demand",
    "strong growth",
    "expands",
    "upgrade",
    "wins contract",
    "profit jumps",
    "revenue growth",
    "buyback",
    "record sales",
    "margin expansion",
    "subscriber growth",
)
NEGATIVE_HINTS = (
    "misses earnings",
    "cuts guidance",
    "guidance cut",
    "weak demand",
    "investigation",
    "lawsuit",
    "downgrade",
    "layoffs",
    "profit warning",
    "recall",
    "decline in sales",
    "regulatory pressure",
    "antitrust",
    "slows growth",
)

LLM_DEFAULT_NEWS_ARTICLE_LIMIT = 5
COMPANY_NEWS_TYPE_WEIGHTS = {
    "company_specific": 1.00,
    "sector_macro": 0.55,
    "market_roundup": 0.12,
    "opinion_or_listicle": 0.08,
    "irrelevant": 0.00,
}
SIGNAL_ARTICLE_COVERAGE_TARGET = 0.85
SIGNAL_ARTICLE_MIN_SHARE = 0.08
SIGNAL_ARTICLE_MIN_WEIGHT = 0.03

_marketaux_last_request_monotonic: Optional[float] = None
_marketaux_rate_limit_exhausted = False
_alpha_vantage_last_request_monotonic: Optional[float] = None
_alpha_vantage_company_request_count = 0
_alpha_vantage_company_budget_exhausted = False


def configured_news_lookback_days() -> int:
    raw_value = os.getenv(MARKETAUX_NEWS_LOOKBACK_DAYS_ENV, "").strip()
    if not raw_value:
        return NEWS_LOOKBACK_DAYS
    try:
        parsed = int(raw_value)
    except ValueError:
        return NEWS_LOOKBACK_DAYS
    return max(1, min(parsed, 30))


def configured_marketaux_request_interval_seconds() -> float:
    raw_value = os.getenv(
        MARKETAUX_REQUEST_INTERVAL_SECONDS_ENV,
        str(MARKETAUX_DEFAULT_REQUEST_INTERVAL_SECONDS),
    ).strip()
    if not raw_value:
        return MARKETAUX_DEFAULT_REQUEST_INTERVAL_SECONDS
    try:
        parsed = float(raw_value)
    except ValueError:
        return MARKETAUX_DEFAULT_REQUEST_INTERVAL_SECONDS
    return max(0.0, min(parsed, 5.0))


def configured_alpha_vantage_company_news_limit() -> int:
    raw_value = os.getenv(
        ALPHA_VANTAGE_COMPANY_NEWS_LIMIT_ENV,
        str(ALPHA_VANTAGE_DEFAULT_COMPANY_NEWS_LIMIT),
    ).strip()
    if not raw_value:
        return ALPHA_VANTAGE_DEFAULT_COMPANY_NEWS_LIMIT
    try:
        parsed = int(raw_value)
    except ValueError:
        return ALPHA_VANTAGE_DEFAULT_COMPANY_NEWS_LIMIT
    return max(1, min(parsed, ALPHA_VANTAGE_MAX_COMPANY_NEWS_LIMIT))


def configured_alpha_vantage_company_request_interval_seconds() -> float:
    raw_value = os.getenv(
        ALPHA_VANTAGE_COMPANY_REQUEST_INTERVAL_SECONDS_ENV,
        str(ALPHA_VANTAGE_DEFAULT_COMPANY_REQUEST_INTERVAL_SECONDS),
    ).strip()
    if not raw_value:
        return ALPHA_VANTAGE_DEFAULT_COMPANY_REQUEST_INTERVAL_SECONDS
    try:
        parsed = float(raw_value)
    except ValueError:
        return ALPHA_VANTAGE_DEFAULT_COMPANY_REQUEST_INTERVAL_SECONDS
    return max(0.0, min(parsed, 60.0))


def configured_alpha_vantage_max_company_requests() -> int:
    raw_value = os.getenv(
        ALPHA_VANTAGE_MAX_COMPANY_REQUESTS_ENV,
        str(ALPHA_VANTAGE_DEFAULT_MAX_COMPANY_REQUESTS),
    ).strip()
    if not raw_value:
        return ALPHA_VANTAGE_DEFAULT_MAX_COMPANY_REQUESTS
    try:
        parsed = int(raw_value)
    except ValueError:
        return ALPHA_VANTAGE_DEFAULT_MAX_COMPANY_REQUESTS
    return max(0, min(parsed, 25))


def configured_gdelt_company_news_limit() -> int:
    raw_value = os.getenv(
        GDELT_COMPANY_NEWS_LIMIT_ENV,
        str(GDELT_DEFAULT_COMPANY_NEWS_LIMIT),
    ).strip()
    if not raw_value:
        return GDELT_DEFAULT_COMPANY_NEWS_LIMIT
    try:
        parsed = int(raw_value)
    except ValueError:
        return GDELT_DEFAULT_COMPANY_NEWS_LIMIT
    return max(1, min(parsed, GDELT_MAX_COMPANY_NEWS_LIMIT))


def configured_company_news_provider_mode() -> str:
    raw_value = os.getenv(COMPANY_NEWS_PROVIDER_MODE_ENV, "blended").strip().lower()
    if raw_value in {"marketaux_only", "fallback_only", "blended"}:
        return raw_value
    return "blended"


def reset_marketaux_fetch_state() -> None:
    global _marketaux_last_request_monotonic, _marketaux_rate_limit_exhausted
    global _alpha_vantage_last_request_monotonic, _alpha_vantage_company_request_count
    global _alpha_vantage_company_budget_exhausted
    _marketaux_last_request_monotonic = None
    _marketaux_rate_limit_exhausted = False
    _alpha_vantage_last_request_monotonic = None
    _alpha_vantage_company_request_count = 0
    _alpha_vantage_company_budget_exhausted = False


def apply_marketaux_request_spacing() -> None:
    global _marketaux_last_request_monotonic
    minimum_interval = configured_marketaux_request_interval_seconds()
    if minimum_interval <= 0:
        return

    now = time_module.monotonic()
    if _marketaux_last_request_monotonic is not None:
        remaining = minimum_interval - (now - _marketaux_last_request_monotonic)
        if remaining > 0:
            time_module.sleep(remaining)

    _marketaux_last_request_monotonic = time_module.monotonic()


def mark_marketaux_rate_limit_exhausted() -> None:
    global _marketaux_rate_limit_exhausted
    _marketaux_rate_limit_exhausted = True


def marketaux_rate_limit_exhausted() -> bool:
    return _marketaux_rate_limit_exhausted


def alpha_vantage_company_budget_exhausted() -> bool:
    return _alpha_vantage_company_budget_exhausted


def mark_alpha_vantage_company_budget_exhausted() -> None:
    global _alpha_vantage_company_budget_exhausted
    _alpha_vantage_company_budget_exhausted = True


def try_consume_alpha_vantage_company_request() -> bool:
    global _alpha_vantage_company_request_count
    budget = configured_alpha_vantage_max_company_requests()
    if budget <= 0:
        mark_alpha_vantage_company_budget_exhausted()
        return False
    if _alpha_vantage_company_request_count >= budget:
        mark_alpha_vantage_company_budget_exhausted()
        return False
    _alpha_vantage_company_request_count += 1
    return True


def apply_alpha_vantage_company_request_spacing() -> None:
    global _alpha_vantage_last_request_monotonic
    minimum_interval = configured_alpha_vantage_company_request_interval_seconds()
    if minimum_interval <= 0:
        _alpha_vantage_last_request_monotonic = time_module.monotonic()
        return

    now = time_module.monotonic()
    if _alpha_vantage_last_request_monotonic is not None:
        remaining = minimum_interval - (now - _alpha_vantage_last_request_monotonic)
        if remaining > 0:
            time_module.sleep(remaining)

    _alpha_vantage_last_request_monotonic = time_module.monotonic()


def is_alpha_vantage_limit_message(message: str) -> bool:
    normalized = message.strip().lower()
    return any(
        fragment in normalized
        for fragment in (
            "call frequency",
            "rate limit",
            "too many requests",
            "premium endpoint",
            "higher api call volume",
        )
    )


@dataclass(frozen=True)
class NewsArticle:
    title: str
    text: str
    published_at: Optional[datetime]
    provider: str
    url: Optional[str]
    relevance_score: float
    recency_weight: float
    source_quality: float
    weight: float
    entity_sentiment: Optional[float] = None
    match_score: float = 0.0
    highlight_snippets: Tuple[str, ...] = ()
    cluster_size: int = 1
    feed: str = "marketaux"


@dataclass(frozen=True)
class ArticleReview:
    article_id: str
    doc_type: str
    company_relevance: float
    materiality: float
    impact_score: float
    confidence: float
    reason: str


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def resolve_universe_csv_path() -> str:
    configured = os.getenv(UNIVERSE_CSV_PATH_ENV, "").strip()
    return configured or DEFAULT_UNIVERSE_CSV_PATH


def load_universe(path: Optional[str] = None) -> List[Tuple[str, str]]:
    path = path or resolve_universe_csv_path()
    df = pd.read_csv(path)
    df = df[df.get("active", 1) == 1]
    df = df.dropna(subset=["symbol", "company_name"])
    rows: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        name = str(row["company_name"]).strip()
        if symbol and name:
            rows.append((symbol, name))
    if not rows:
        raise RuntimeError("No active symbols found in universe.csv")
    return rows


def configured_company_news_shortlist_size() -> int:
    raw_value = os.getenv(
        COMPANY_NEWS_SHORTLIST_SIZE_ENV,
        str(DEFAULT_COMPANY_NEWS_SHORTLIST_SIZE),
    ).strip()
    if not raw_value:
        return DEFAULT_COMPANY_NEWS_SHORTLIST_SIZE
    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_COMPANY_NEWS_SHORTLIST_SIZE
    return max(1, min(parsed, 100))


def shortlist_priority_symbols() -> List[str]:
    symbols: List[str] = []

    try:
        with RISK_PICKS_PATH.open("r", encoding="utf-8") as handle:
            risk_payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        risk_payload = None
    if isinstance(risk_payload, dict):
        overall_selection = risk_payload.get("overall_selection")
        pick = overall_selection.get("pick") if isinstance(overall_selection, dict) else None
        if isinstance(pick, dict):
            symbol = str(pick.get("symbol", "")).strip().upper()
            if symbol:
                symbols.append(symbol)

    try:
        history_entries = [
            normalize_history_entry(entry)
            for entry in load_existing_history_entries(HISTORY_PATH)
        ]
    except Exception:
        history_entries = []
    for entry in reversed(history_entries[-4:]):
        symbol = str(entry.get("symbol", "")).strip().upper()
        if symbol:
            symbols.append(symbol)

    seen = set()
    ordered: List[str] = []
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def technical_shortlist_score(
    symbol: str,
    sector: str,
    market_metrics: Dict[str, float],
    sector_metrics_by_sector: Dict[str, Dict[str, float]],
) -> float:
    stock_metrics = compute_technical_metrics(fetch_price_series_cached(symbol))
    sector_metrics = sector_metrics_by_sector[sector]
    relative_metrics = compute_relative_strength_metrics(stock_metrics, market_metrics, sector_metrics)

    short_component = clamp(stock_metrics["momentum_5d"] / 0.08, -1.0, 1.0)
    medium_component = clamp(stock_metrics["momentum_20d"] / 0.18, -1.0, 1.0)
    market_relative_component = clamp(relative_metrics["market_relative_20d"] / 0.08, -1.0, 1.0)
    sector_relative_component = clamp(relative_metrics["sector_relative_20d"] / 0.08, -1.0, 1.0)
    trend_component = clamp(stock_metrics["trend_gap"] / 0.08, -1.0, 1.0)
    participation_component = clamp((stock_metrics["positive_day_ratio"] - 0.5) * 2.0, -1.0, 1.0)
    volatility_penalty = clamp(stock_metrics["volatility"] / 0.04, 0.0, 1.0)

    return (
        0.24 * short_component
        + 0.26 * medium_component
        + 0.20 * market_relative_component
        + 0.12 * sector_relative_component
        + 0.08 * trend_component
        + 0.05 * participation_component
        - 0.05 * volatility_penalty
    )


def select_company_news_shortlist(universe: Sequence[Tuple[str, str]]) -> List[str]:
    shortlist_size = configured_company_news_shortlist_size()
    if shortlist_size >= len(universe):
        return [symbol for symbol, _ in universe]

    universe_path = Path(resolve_universe_csv_path())
    sector_map = load_sector_map(universe_path)
    market_metrics = compute_technical_metrics(fetch_price_series_cached(MARKET_BENCHMARK_SYMBOL))
    sector_metrics_by_sector = {
        sector: compute_technical_metrics(fetch_price_series_cached(etf))
        for sector, etf in SECTOR_ETF_BY_SECTOR.items()
    }

    ranked: List[Tuple[float, str]] = []
    for symbol, _company_name in universe:
        sector = sector_map.get(symbol)
        if not sector or sector not in sector_metrics_by_sector:
            continue
        try:
            score = technical_shortlist_score(symbol, sector, market_metrics, sector_metrics_by_sector)
        except Exception as exc:
            print(f"[WARN] [{symbol}] Technical shortlist scoring failed: {exc}")
            continue
        ranked.append((score, symbol))

    ranked.sort(reverse=True)
    selected = [symbol for _score, symbol in ranked[:shortlist_size]]
    selected.extend(shortlist_priority_symbols())

    deduped: List[str] = []
    seen = set()
    for symbol in selected:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def count_relevant_company_articles(
    items: Sequence[dict],
    symbol: str,
    company_name: str,
) -> int:
    now = datetime.now(timezone.utc)
    seen_titles = set()
    count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        article = build_marketaux_article(item, symbol, company_name, now)
        if article is None:
            continue
        title_key = normalize_text(article.title)
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        count += 1
    return count


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
        return None


def parse_alpha_vantage_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        parsed = datetime.strptime(normalized, "%Y%m%dT%H%M")
        return parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_gdelt_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    for pattern in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            parsed = datetime.strptime(normalized, pattern)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def iso_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def normalize_symbol_token(symbol: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", symbol.lower())


def company_keywords(company_name: str) -> List[str]:
    stop_words = {
        "inc",
        "corp",
        "corporation",
        "plc",
        "sa",
        "co",
        "ltd",
        "company",
        "incorporated",
        "holdings",
        "group",
    }
    return [
        part
        for part in normalize_text(company_name).split()
        if part not in stop_words and len(part) > 2
    ]


def company_base_phrase(company_name: str) -> str:
    stop_words = {
        "inc",
        "corp",
        "corporation",
        "plc",
        "sa",
        "co",
        "ltd",
        "company",
        "incorporated",
        "holdings",
        "group",
    }
    parts = [
        part
        for part in normalize_text(company_name).split()
        if part not in stop_words
    ]
    return " ".join(parts).strip()


def build_symbol_keywords(symbol: str, company_name: str) -> List[str]:
    normalized_company = normalize_text(company_name)
    base_phrase = company_base_phrase(company_name)
    significant_words = company_keywords(company_name)
    phrase_candidates = {normalized_company}
    if base_phrase:
        phrase_candidates.add(base_phrase)

    if len(significant_words) >= 2:
        phrase_candidates.add(" ".join(significant_words[:2]))

    manual: Dict[str, List[str]] = {
        "GOOGL": ["google", "alphabet"],
        "GOOG": ["google", "alphabet"],
        "META": ["meta", "facebook"],
        "FB": ["meta", "facebook"],
        "BRK.B": ["berkshire hathaway", "berkshire"],
        "BRK.A": ["berkshire hathaway", "berkshire"],
        "NVDA": ["nvidia"],
        "TSLA": ["tesla"],
        "MSFT": ["microsoft"],
        "AAPL": ["apple"],
        "AMZN": ["amazon"],
        "T": ["at&t", "att"],
        "V": ["visa"],
        "HD": ["home depot"],
        "MA": ["mastercard"],
        "DIS": ["disney"],
        "JPM": ["jpmorgan", "jp morgan"],
        "CSCO": ["cisco"],
    }

    for item in manual.get(symbol.upper(), []):
        phrase_candidates.add(normalize_text(item))

    cleaned_symbol = normalize_symbol_token(symbol)
    keyword_set = set()
    if len(cleaned_symbol) >= 2:
        keyword_set.add(cleaned_symbol)

    return [keyword for keyword in keyword_set.union(phrase_candidates) if keyword]


def source_quality_weight(provider: str, url: Optional[str]) -> float:
    provider_key = provider.lower()
    domain = urlparse(url).netloc.lower() if url else ""
    source_key = f"{provider_key} {domain}"

    premium_sources = {
        "reuters": 1.00,
        "associated press": 0.98,
        "apnews.com": 0.98,
        "bloomberg": 0.96,
        "wsj": 0.95,
        "wall street journal": 0.95,
        "financial times": 0.94,
        "ft.com": 0.94,
        "barrons": 0.92,
        "cnbc": 0.90,
        "marketwatch": 0.88,
        "investopedia": 0.85,
        "zacks": 0.82,
        "motley fool": 0.78,
        "seeking alpha": 0.78,
        "benzinga": 0.82,
    }

    for key, weight in premium_sources.items():
        if key in source_key:
            return weight

    return 0.75


def compute_recency_weight(
    published_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> float:
    if published_at is None:
        return 0.55

    current_time = now or datetime.now(timezone.utc)
    age_hours = max(0.0, (current_time - published_at).total_seconds() / 3600)
    decay = math.exp(-age_hours / RECENCY_HALFLIFE_HOURS)
    return clamp(0.30 + 0.70 * decay, 0.30, 1.00)


def extract_title_and_text(item: dict) -> Tuple[str, str]:
    content = item.get("content") or {}

    title_candidates = [
        content.get("title"),
        item.get("title"),
    ]
    summary_candidates = [
        content.get("summary"),
        content.get("description"),
        item.get("description"),
        item.get("snippet"),
    ]

    title = ""
    for candidate in title_candidates:
        if isinstance(candidate, str) and candidate.strip():
            title = candidate.strip()
            break

    parts = [title]
    for candidate in summary_candidates:
        if isinstance(candidate, str) and candidate.strip():
            parts.append(strip_html_tags(candidate.strip()))

    text = " ".join(part for part in parts if part).strip()
    return title, text


def compute_relevance_from_text(
    title: str,
    text: str,
    symbol: str,
    company_name: str,
) -> float:
    normalized_title = normalize_text(title)
    normalized_text = normalize_text(text)

    if not normalized_text:
        return 0.0

    keywords = build_symbol_keywords(symbol, company_name)
    company_phrase = normalize_text(company_name)
    company_base = company_base_phrase(company_name)
    title_tokens = set(normalized_title.split())
    text_tokens = set(normalized_text.split())
    significant_words = company_keywords(company_name)

    score = 0.0

    if company_phrase and company_phrase in normalized_title:
        score += 0.45
    elif company_phrase and company_phrase in normalized_text:
        score += 0.30

    if company_base and company_base in normalized_title:
        score += 0.28
    elif company_base and company_base in normalized_text:
        score += 0.18

    phrase_hits_title = sum(1 for keyword in keywords if " " in keyword and keyword in normalized_title)
    phrase_hits_text = sum(1 for keyword in keywords if " " in keyword and keyword in normalized_text)
    token_hits_title = sum(1 for keyword in keywords if " " not in keyword and keyword in title_tokens)
    token_hits_text = sum(1 for keyword in keywords if " " not in keyword and keyword in text_tokens)

    if phrase_hits_title > 0:
        score += 0.22
    elif phrase_hits_text > 0:
        score += 0.14

    if token_hits_title > 0:
        score += 0.14
    elif token_hits_text > 0:
        score += 0.08

    matched_words = sum(1 for word in significant_words if word in text_tokens)
    matched_title_words = sum(1 for word in significant_words if word in title_tokens)
    if matched_words >= 2:
        score += 0.16
    elif matched_title_words >= 1:
        score += 0.10

    if normalized_title and len(normalized_title.split()) <= 4 and matched_words == 0 and phrase_hits_title == 0:
        score *= 0.70

    return clamp(score, 0.0, 1.0)


def compute_relevance_score(
    item: dict,
    symbol: str,
    company_name: str,
) -> float:
    title, text = extract_title_and_text(item)
    return compute_relevance_from_text(title, text, symbol, company_name)


def heuristic_sentiment_value(text: str) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.0

    positive_hits = sum(1 for phrase in POSITIVE_HINTS if phrase in normalized)
    negative_hits = sum(1 for phrase in NEGATIVE_HINTS if phrase in normalized)

    if positive_hits == 0 and negative_hits == 0:
        return 0.0

    total_hits = positive_hits + negative_hits
    balance = (positive_hits - negative_hits) / total_hits
    return clamp(balance * 0.75, -0.75, 0.75)


def normalize_match_score(match_score: Optional[float]) -> float:
    if match_score is None:
        return 0.0
    return clamp(1.0 - math.exp(-max(0.0, float(match_score)) / 8.0), 0.0, 1.0)


def configured_marketaux_news_limit() -> int:
    raw_limit = os.getenv(MARKETAUX_NEWS_LIMIT_ENV, str(MARKETAUX_DEFAULT_NEWS_LIMIT))
    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        parsed_limit = MARKETAUX_DEFAULT_NEWS_LIMIT
    return max(1, min(parsed_limit, MARKETAUX_MAX_NEWS_LIMIT))


def configured_llm_news_limit() -> int:
    raw_limit = os.getenv(COMPANY_LLM_ARTICLE_LIMIT_ENV, str(LLM_DEFAULT_NEWS_ARTICLE_LIMIT))
    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        parsed_limit = LLM_DEFAULT_NEWS_ARTICLE_LIMIT
    return max(1, min(parsed_limit, MAX_ARTICLES_PER_SYMBOL))


def alpha_vantage_api_key(required: bool = False) -> str:
    key = os.getenv(ALPHA_VANTAGE_API_KEY_ENV, "").strip()
    if key or not required:
        return key
    raise RuntimeError(f"{ALPHA_VANTAGE_API_KEY_ENV} is required for company-news fallback enrichment.")


def marketaux_api_token() -> str:
    token = os.getenv(MARKETAUX_API_TOKEN_ENV, "").strip()
    if token:
        return token
    if allow_marketaux_fallback():
        return ""
    raise RuntimeError(
        f"{MARKETAUX_API_TOKEN_ENV} is required for news generation. "
        "Set a free Marketaux API token for weekly runs."
    )


def allow_marketaux_fallback() -> bool:
    raw = os.getenv(ALLOW_MARKETAUX_FALLBACK_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_marketaux_fixture(symbol: str) -> Optional[List[dict]]:
    configured = os.getenv(MARKETAUX_FIXTURE_PATH_ENV, "").strip()
    if not configured:
        return None

    fixture_path = Path(configured)
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        symbol_payload = payload.get(symbol.upper(), [])
        if isinstance(symbol_payload, list):
            print(f"[INFO] [{symbol}] Loaded {len(symbol_payload)} fixture news items from {fixture_path}.")
            return symbol_payload
        raise RuntimeError(f"Fixture payload for {symbol} must be a list in {fixture_path}")

    if isinstance(payload, list):
        print(f"[INFO] [{symbol}] Loaded shared fixture news list from {fixture_path}.")
        return payload

    raise RuntimeError(f"Unsupported Marketaux fixture shape in {fixture_path}")


def build_marketaux_query(symbol: str, published_after: datetime) -> str:
    params = {
        "symbols": symbol,
        "filter_entities": "true",
        "must_have_entities": "true",
        "group_similar": "true",
        "language": "en",
        "sort": "published_at",
        "sort_order": "desc",
        "published_after": published_after.strftime("%Y-%m-%dT%H:%M"),
        "limit": str(configured_marketaux_news_limit()),
        "api_token": marketaux_api_token(),
    }
    return f"{MARKETAUX_NEWS_ENDPOINT}?{urlencode(params)}"


def alpha_vantage_company_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if "." in normalized:
        return normalized.replace(".", "-")
    return normalized


def build_alpha_vantage_company_query(symbol: str, published_after: datetime) -> str:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": alpha_vantage_company_symbol(symbol),
        "time_from": published_after.strftime("%Y%m%dT%H%M"),
        "sort": "LATEST",
        "limit": str(configured_alpha_vantage_company_news_limit()),
        "apikey": alpha_vantage_api_key(required=True),
    }
    return f"{ALPHA_VANTAGE_NEWS_ENDPOINT}?{urlencode(params)}"


def company_gdelt_query(symbol: str, company_name: str) -> str:
    quoted_terms: List[str] = []
    seen = set()
    for keyword in build_symbol_keywords(symbol, company_name):
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        if len(keyword) <= 2:
            continue
        if " " in keyword:
            quoted_terms.append(f'"{keyword}"')
        elif len(keyword) >= 4:
            quoted_terms.append(keyword)
    if not quoted_terms:
        base = company_base_phrase(company_name) or company_name.strip()
        quoted_terms.append(f'"{base}"')
    return " OR ".join(quoted_terms[:4])


def build_gdelt_company_query(symbol: str, company_name: str) -> str:
    params = {
        "query": company_gdelt_query(symbol, company_name),
        "mode": "artlist",
        "maxrecords": str(configured_gdelt_company_news_limit()),
        "timespan": f"{configured_news_lookback_days()}days",
        "format": "json",
    }
    return f"{GDELT_DOC_ENDPOINT}?{urlencode(params)}"


def fetch_marketaux_payload(symbol: str, published_after: datetime) -> List[dict]:
    if allow_marketaux_fallback() and marketaux_rate_limit_exhausted():
        print(f"[WARN] [{symbol}] Marketaux rate limit already hit earlier in this run, returning neutral fallback data.")
        return []

    url = build_marketaux_query(symbol, published_after)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "weekly-stock-pick/1.0",
        },
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, NEWS_FETCH_ATTEMPTS + 1):
        try:
            if not consume_provider_budget(
                "marketaux",
                units=1,
                category="company_news",
                daily_limit=MARKETAUX_DAILY_REQUEST_LIMIT,
            ):
                mark_marketaux_rate_limit_exhausted()
                print(f"[WARN] [{symbol}] Marketaux workflow budget exhausted, returning fallback data.")
                return []
            apply_marketaux_request_spacing()
            with urlopen(request, timeout=MARKETAUX_REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise RuntimeError("Unexpected Marketaux response shape")
            return payload["data"]
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 402 and allow_marketaux_fallback():
                mark_marketaux_rate_limit_exhausted()
                print(f"[WARN] [{symbol}] Marketaux usage limit reached, returning neutral fallback data.")
                return []
            if exc.code in (401, 402, 403):
                raise RuntimeError(f"Marketaux authentication failed for {symbol}: HTTP {exc.code} {body[:160]}") from exc
            if exc.code == 429:
                if allow_marketaux_fallback():
                    mark_marketaux_rate_limit_exhausted()
                    print(f"[WARN] [{symbol}] Marketaux rate limit reached, returning neutral fallback data.")
                    return []
                last_error = RuntimeError(f"Marketaux rate limit hit for {symbol}: {body[:160]}")
            else:
                last_error = RuntimeError(f"Marketaux HTTP {exc.code} for {symbol}: {body[:160]}")
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc

        if attempt < NEWS_FETCH_ATTEMPTS:
            time_module.sleep(NEWS_FETCH_RETRY_SECONDS * attempt)

    raise RuntimeError(
        f"Unable to fetch Marketaux news for {symbol} after {NEWS_FETCH_ATTEMPTS} attempts: {last_error}"
    )


def fetch_alpha_vantage_company_payload(symbol: str, published_after: datetime) -> List[dict]:
    api_key = alpha_vantage_api_key(required=False)
    if not api_key:
        print(f"[WARN] [{symbol}] Alpha Vantage API key missing, skipping company-news fallback feed.")
        return []
    if alpha_vantage_company_budget_exhausted():
        print(f"[WARN] [{symbol}] Alpha Vantage company-news budget exhausted earlier in this run, skipping fallback feed.")
        return []

    url = build_alpha_vantage_company_query(symbol, published_after)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "weekly-stock-pick/1.0",
        },
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, NEWS_FETCH_ATTEMPTS + 1):
        try:
            if not try_consume_alpha_vantage_company_request():
                print(f"[WARN] [{symbol}] Alpha Vantage company-news budget exhausted for this run, skipping fallback feed.")
                return []
            if not consume_provider_budget(
                "alpha_vantage",
                units=1,
                category="company_news",
                daily_limit=ALPHA_VANTAGE_DAILY_REQUEST_LIMIT,
            ):
                mark_alpha_vantage_company_budget_exhausted()
                print(f"[WARN] [{symbol}] Alpha Vantage daily workflow budget exhausted, skipping fallback feed.")
                return []
            apply_alpha_vantage_company_request_spacing()
            with urlopen(request, timeout=MARKETAUX_REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("feed"), list):
                return payload["feed"][: configured_alpha_vantage_company_news_limit()]
            if isinstance(payload, dict):
                message = (
                    str(payload.get("Information", "")).strip()
                    or str(payload.get("Note", "")).strip()
                    or str(payload.get("Error Message", "")).strip()
                )
                if is_alpha_vantage_limit_message(message):
                    mark_alpha_vantage_company_budget_exhausted()
                    print(f"[WARN] [{symbol}] Alpha Vantage company-news limit reached, skipping fallback feed.")
                    return []
                raise RuntimeError(message or "Unexpected Alpha Vantage response shape")
            raise RuntimeError("Unexpected Alpha Vantage response shape")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                mark_alpha_vantage_company_budget_exhausted()
                print(f"[WARN] [{symbol}] Alpha Vantage company-news rate limit reached, skipping fallback feed.")
                return []
            last_error = RuntimeError(f"Alpha Vantage HTTP {exc.code} for {symbol}: {body[:160]}")
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        if attempt < NEWS_FETCH_ATTEMPTS:
            time_module.sleep(NEWS_FETCH_RETRY_SECONDS * attempt)

    raise RuntimeError(
        f"Unable to fetch Alpha Vantage company news for {symbol} after {NEWS_FETCH_ATTEMPTS} attempts: {last_error}"
    )


def fetch_gdelt_company_payload(symbol: str, company_name: str) -> List[dict]:
    consume_provider_budget("gdelt", units=1, category="company_news")
    url = build_gdelt_company_query(symbol, company_name)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "weekly-stock-pick/1.0",
        },
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, NEWS_FETCH_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=MARKETAUX_REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8", errors="replace").strip()
            if not body:
                raise RuntimeError("Empty GDELT response body")
            if body.startswith("<"):
                raise RuntimeError(f"Non-JSON GDELT response: {body[:120]}")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid GDELT JSON response: {body[:120]}") from exc
            if isinstance(payload, dict):
                if isinstance(payload.get("articles"), list):
                    return payload["articles"][: configured_gdelt_company_news_limit()]
                if isinstance(payload.get("data"), list):
                    return payload["data"][: configured_gdelt_company_news_limit()]
            raise RuntimeError("Unexpected GDELT response shape")
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
        if attempt < NEWS_FETCH_ATTEMPTS:
            time_module.sleep(NEWS_FETCH_RETRY_SECONDS * attempt)

    raise RuntimeError(
        f"Unable to fetch GDELT company news for {symbol} after {NEWS_FETCH_ATTEMPTS} attempts: {last_error}"
    )


def alpha_vantage_ticker_item(
    ticker_items: Any,
    symbol: str,
) -> Optional[dict]:
    target_token = normalize_symbol_token(symbol)
    if not isinstance(ticker_items, list):
        return None
    for item in ticker_items:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip()
        if normalize_symbol_token(ticker) == target_token:
            return item
    if len(ticker_items) == 1 and isinstance(ticker_items[0], dict):
        return ticker_items[0]
    return None


def normalize_alpha_vantage_company_item(item: dict, symbol: str) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title", "")).strip()
    summary = str(item.get("summary", "")).strip()
    url = str(item.get("url", "")).strip() or None
    provider = str(item.get("source", "")).strip() or "Alpha Vantage"
    published_at = parse_alpha_vantage_datetime(item.get("time_published"))
    if not title:
        return None

    topics = [
        str(topic.get("topic", "")).strip()
        for topic in item.get("topics", [])
        if isinstance(topic, dict) and str(topic.get("topic", "")).strip()
    ]
    sentiment_label = str(item.get("overall_sentiment_label", "")).strip()
    alpha_entity = alpha_vantage_ticker_item(item.get("ticker_sentiment"), symbol)

    entity_payload: List[dict] = []
    if alpha_entity is not None:
        sentiment_score = None
        relevance_score = 0.0
        raw_sentiment = alpha_entity.get("ticker_sentiment_score")
        if isinstance(raw_sentiment, str):
            try:
                sentiment_score = float(raw_sentiment)
            except ValueError:
                sentiment_score = None
        elif isinstance(raw_sentiment, (int, float)):
            sentiment_score = float(raw_sentiment)

        raw_relevance = alpha_entity.get("relevance_score")
        if isinstance(raw_relevance, str):
            try:
                relevance_score = float(raw_relevance)
            except ValueError:
                relevance_score = 0.0
        elif isinstance(raw_relevance, (int, float)):
            relevance_score = float(raw_relevance)

        entity_payload.append(
            {
                "symbol": symbol,
                "sentiment_score": clamp(sentiment_score or 0.0, -1.0, 1.0) if sentiment_score is not None else None,
                "match_score": max(0.0, relevance_score * 20.0),
                "highlights": [],
            }
        )

    topic_summary = f"Topics: {', '.join(topics[:4])}." if topics else ""
    sentiment_summary = f"Overall sentiment label: {sentiment_label}." if sentiment_label else ""

    return {
        "feed": "alpha_vantage",
        "title": title,
        "description": summary,
        "snippet": " ".join(part for part in (topic_summary, sentiment_summary) if part).strip(),
        "published_at": iso_utc(published_at),
        "source": provider,
        "url": url,
        "entities": entity_payload,
        "similar": [],
    }


def normalize_gdelt_company_item(item: dict) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title", "")).strip()
    url = str(item.get("url", "")).strip() or None
    provider = (
        str(item.get("domain", "")).strip()
        or str(item.get("source", "")).strip()
        or "GDELT"
    )
    published_at = parse_gdelt_datetime(item.get("seendate") or item.get("date"))
    description = str(item.get("snippet", "")).strip()
    if not description:
        source_country = str(item.get("sourcecountry", "")).strip()
        language = str(item.get("language", "")).strip()
        fragments = []
        if source_country:
            fragments.append(f"Source country: {source_country}.")
        if language:
            fragments.append(f"Language: {language}.")
        description = " ".join(fragments)
    if not title:
        return None

    return {
        "feed": "gdelt",
        "title": title,
        "description": description,
        "snippet": "",
        "published_at": iso_utc(published_at),
        "source": provider,
        "url": url,
        "entities": [],
        "similar": [],
    }


def fetch_company_raw_news_sources(symbol: str, company_name: str) -> Dict[str, List[dict]]:
    fixture_payload = load_marketaux_fixture(symbol)
    if fixture_payload is not None:
        return {"fixture": fixture_payload}

    published_after = datetime.now(timezone.utc) - timedelta(days=configured_news_lookback_days())
    raw_sources: Dict[str, List[dict]] = {}
    marketaux_unavailable = False
    provider_mode = configured_company_news_provider_mode()
    marketaux_relevant_count = 0

    if provider_mode == "fallback_only":
        raw_sources["marketaux"] = []
        marketaux_unavailable = True
    elif allow_marketaux_fallback() and marketaux_rate_limit_exhausted():
        print(f"[WARN] [{symbol}] Marketaux rate limit already exhausted earlier in this run, trying fallback providers.")
        raw_sources["marketaux"] = []
        marketaux_unavailable = True
    elif not os.getenv(MARKETAUX_API_TOKEN_ENV, "").strip() and allow_marketaux_fallback():
        print(f"[WARN] [{symbol}] Marketaux token missing, trying fallback providers.")
        raw_sources["marketaux"] = []
        marketaux_unavailable = True
    else:
        try:
            raw_sources["marketaux"] = fetch_marketaux_payload(symbol, published_after)
            marketaux_relevant_count = count_relevant_company_articles(
                raw_sources["marketaux"],
                symbol,
                company_name,
            )
            if allow_marketaux_fallback() and marketaux_rate_limit_exhausted() and not raw_sources["marketaux"]:
                marketaux_unavailable = True
        except Exception as exc:
            if not allow_marketaux_fallback():
                raise
            print(f"[WARN] [{symbol}] Marketaux company feed failed, trying fallback providers: {exc}")
            record_runtime_event(
                f"Marketaux company news failed for at least one symbol and fallback feeds were used.",
                provider="marketaux",
            )
            raw_sources["marketaux"] = []
            marketaux_unavailable = True

    fetch_companion_sources = provider_mode != "marketaux_only"
    should_fetch_gdelt = fetch_companion_sources and (
        provider_mode == "fallback_only"
        or marketaux_unavailable
        or marketaux_relevant_count < COMPANY_PRIMARY_MARKETAUX_TARGET
    )

    if should_fetch_gdelt:
        try:
            raw_sources["gdelt"] = [
                item
                for item in (
                    normalize_gdelt_company_item(raw_item)
                    for raw_item in fetch_gdelt_company_payload(symbol, company_name)
                )
                if item is not None
            ]
        except Exception as exc:
            print(f"[WARN] [{symbol}] GDELT company-news fallback failed: {exc}")
            record_runtime_event(
                f"GDELT company news failed for at least one symbol.",
                provider="gdelt",
            )
            raw_sources["gdelt"] = []
    elif fetch_companion_sources:
        raw_sources["gdelt"] = []

    if fetch_companion_sources:
        primary_relevant_count = marketaux_relevant_count + count_relevant_company_articles(
            raw_sources.get("gdelt", []),
            symbol,
            company_name,
        )
        if primary_relevant_count < COMPANY_ALPHA_VANTAGE_ESCALATION_FLOOR:
            try:
                raw_sources["alpha_vantage"] = [
                    item
                    for item in (
                        normalize_alpha_vantage_company_item(raw_item, symbol)
                        for raw_item in fetch_alpha_vantage_company_payload(symbol, published_after)
                    )
                    if item is not None
                ]
            except Exception as exc:
                print(f"[WARN] [{symbol}] Alpha Vantage company-news fallback failed: {exc}")
                record_runtime_event(
                    f"Alpha Vantage company-news fallback failed for at least one symbol.",
                    provider="alpha_vantage",
                )
                raw_sources["alpha_vantage"] = []
        else:
            raw_sources["alpha_vantage"] = []

    return raw_sources


def extract_marketaux_entity(entities: List[dict], symbol: str) -> Optional[dict]:
    target_symbol = symbol.upper()
    for entity in entities:
        if str(entity.get("symbol", "")).upper() == target_symbol:
            return entity
    if len(entities) == 1:
        return entities[0]
    return None


def extract_entity_highlights(entity: Optional[dict]) -> Tuple[str, ...]:
    if not entity:
        return ()

    snippets: List[str] = []
    for highlight in entity.get("highlights", []) or []:
        raw_snippet = strip_html_tags(str(highlight.get("highlight", "")).strip())
        if raw_snippet:
            snippets.append(raw_snippet)
    return tuple(snippets[:2])


def derive_entity_sentiment(entity: Optional[dict]) -> Optional[float]:
    if not entity:
        return None

    direct = entity.get("sentiment_score")
    direct_value: Optional[float] = None
    if isinstance(direct, (int, float)):
        direct_value = clamp(float(direct), -1.0, 1.0)

    highlight_scores = [
        float(item["sentiment"])
        for item in entity.get("highlights", []) or []
        if isinstance(item, dict) and isinstance(item.get("sentiment"), (int, float))
    ]
    if highlight_scores:
        highlight_average = sum(highlight_scores) / len(highlight_scores)
        highlight_value = clamp(highlight_average, -1.0, 1.0)
        if direct_value is None:
            return highlight_value
        return clamp(direct_value * 0.70 + highlight_value * 0.30, -1.0, 1.0)

    return direct_value


def build_marketaux_article(
    item: dict,
    symbol: str,
    company_name: str,
    now: datetime,
) -> Optional[NewsArticle]:
    published_at = parse_iso_datetime(item.get("published_at"))
    lookback_days = configured_news_lookback_days()
    if published_at is not None and published_at < now - timedelta(days=lookback_days):
        return None

    title, text = extract_title_and_text(item)
    if not text:
        return None

    entity = extract_marketaux_entity(item.get("entities", []) or [], symbol)
    keyword_relevance = compute_relevance_from_text(title, text, symbol, company_name)
    provider_match_score = float(entity.get("match_score", 0.0)) if entity else 0.0
    provider_relevance = normalize_match_score(provider_match_score)

    if entity is None:
        relevance_score = keyword_relevance
    else:
        relevance_score = clamp(
            max(keyword_relevance, provider_relevance * 0.65 + keyword_relevance * 0.35),
            0.0,
            1.0,
        )

    if relevance_score < MIN_RELEVANCE_SCORE:
        return None

    provider = str(item.get("source", "") or "unknown source").strip() or "unknown source"
    url = item.get("url")
    recency_weight = compute_recency_weight(published_at, now)
    source_quality = source_quality_weight(provider, url)
    text_length_weight = 1.0 if len(text) >= 160 else 0.90
    cluster_size = 1 + len(item.get("similar", []) or [])
    cluster_support = min(1.0 + 0.08 * max(0, cluster_size - 1), 1.35)
    weight = relevance_score * recency_weight * source_quality * text_length_weight * cluster_support

    return NewsArticle(
        title=title or symbol,
        text=text,
        published_at=published_at,
        provider=provider,
        url=url if isinstance(url, str) else None,
        relevance_score=relevance_score,
        recency_weight=recency_weight,
        source_quality=source_quality,
        weight=weight,
        entity_sentiment=derive_entity_sentiment(entity),
        match_score=provider_match_score,
        highlight_snippets=extract_entity_highlights(entity),
        cluster_size=cluster_size,
        feed=str(item.get("feed", "marketaux")).strip() or "marketaux",
    )


def fetch_raw_news(symbol: str) -> List[dict]:
    fixture_payload = load_marketaux_fixture(symbol)
    if fixture_payload is not None:
        return fixture_payload
    if allow_marketaux_fallback() and marketaux_rate_limit_exhausted():
        print(f"[WARN] [{symbol}] Skipping Marketaux fetch because the rate limit was already reached earlier in this run.")
        return []
    if not os.getenv(MARKETAUX_API_TOKEN_ENV, "").strip() and allow_marketaux_fallback():
        print(f"[WARN] [{symbol}] Marketaux token missing, returning neutral fallback data.")
        return []
    published_after = datetime.now(timezone.utc) - timedelta(days=configured_news_lookback_days())
    return fetch_marketaux_payload(symbol, published_after)


def fetch_symbol_news(symbol: str, company_name: str) -> List[NewsArticle]:
    raw_sources = fetch_company_raw_news_sources(symbol, company_name)
    raw_news = [
        item
        for items in raw_sources.values()
        for item in items
        if isinstance(item, dict)
    ]

    print(f"\n=== Fetching news for {symbol} ===")
    print(f"[{symbol}] raw news count: {len(raw_news)}")
    for source_name, items in raw_sources.items():
        print(f"[{symbol}] {source_name} items: {len(items)}")

    now = datetime.now(timezone.utc)
    articles: List[NewsArticle] = []
    seen_titles = set()

    for idx, item in enumerate(raw_news):
        article = build_marketaux_article(item, symbol, company_name, now)
        feed = str(item.get("feed", "marketaux")).strip() or "marketaux"
        print(
            f"[{symbol}] {feed} item {idx}: time={item.get('published_at')} title='{str(item.get('title', ''))[:120]}'"
        )

        if article is None:
            continue

        title_key = normalize_text(article.title)
        if title_key in seen_titles:
            continue

        articles.append(article)
        seen_titles.add(title_key)

    articles.sort(
        key=lambda article: (
            article.weight,
            article.relevance_score,
            article.source_quality,
            article.recency_weight,
            article.published_at or datetime.fromtimestamp(0, tz=timezone.utc),
        ),
        reverse=True,
    )
    selected = articles[:MAX_ARTICLES_PER_SYMBOL]

    print(f"[{symbol}] articles used: {len(selected)}")
    return selected


def init_models():
    return None, None


def article_sentiment_value(article: NewsArticle) -> float:
    heuristic_value = heuristic_sentiment_value(article.text)
    if article.entity_sentiment is None:
        return heuristic_value
    if heuristic_value == 0.0:
        return clamp(article.entity_sentiment, -1.0, 1.0)
    return clamp(article.entity_sentiment * 0.75 + heuristic_value * 0.25, -1.0, 1.0)


def classify_dominant_signal(calibrated_sentiment: float, dispersion: float) -> str:
    if dispersion > 0.55 and abs(calibrated_sentiment) < 0.15:
        return "mixed"
    if calibrated_sentiment >= 0.20:
        return "bullish"
    if calibrated_sentiment <= -0.20:
        return "bearish"
    return "neutral"


def aggregate_news_signal(
    articles: List[NewsArticle],
    sentiment_values: List[float],
) -> Dict[str, float]:
    if not articles or not sentiment_values:
        return {
            "raw_sentiment": 0.0,
            "calibrated_sentiment": 0.0,
            "news_score": 0.5,
            "news_confidence": 0.20,
            "effective_article_count": 0.0,
            "source_count": 0,
            "average_relevance": 0.0,
            "average_recency_weight": 0.0,
            "average_source_quality": 0.0,
            "provider_sentiment_coverage": 0.0,
            "dominant_weight_share": 0.0,
            "concentration_score": 0.0,
            "sentiment_dispersion": 0.0,
            "dominant_signal": "neutral",
        }

    if len(articles) != len(sentiment_values):
        raise ValueError("articles and sentiment_values must have the same length")

    total_weight = sum(article.weight for article in articles)
    if total_weight <= 0:
        return {
            "raw_sentiment": 0.0,
            "calibrated_sentiment": 0.0,
            "news_score": 0.5,
            "news_confidence": 0.20,
            "effective_article_count": 0.0,
            "source_count": 0,
            "average_relevance": 0.0,
            "average_recency_weight": 0.0,
            "average_source_quality": 0.0,
            "provider_sentiment_coverage": 0.0,
            "dominant_weight_share": 0.0,
            "concentration_score": 0.0,
            "sentiment_dispersion": 0.0,
            "dominant_signal": "neutral",
        }

    raw_sentiment = sum(value * article.weight for article, value in zip(articles, sentiment_values)) / total_weight
    average_relevance = sum(article.relevance_score for article in articles) / len(articles)
    average_recency_weight = sum(article.recency_weight for article in articles) / len(articles)
    average_source_quality = sum(article.source_quality for article in articles) / len(articles)
    source_count = len({article.provider.lower() for article in articles})
    dispersion = sum(abs(value - raw_sentiment) * article.weight for article, value in zip(articles, sentiment_values)) / total_weight
    dominant_weight_share = max(article.weight for article in articles) / total_weight
    provider_sentiment_coverage = sum(
        article.weight for article in articles if article.entity_sentiment is not None
    ) / total_weight

    coverage_score = clamp(total_weight / 2.5, 0.0, 1.0)
    diversity_score = clamp(source_count / min(max(len(articles), 1), 4), 0.0, 1.0)
    stability_score = 1.0 - clamp(dispersion / 1.0, 0.0, 1.0)
    concentration_score = 1.0 - clamp((dominant_weight_share - 0.45) / 0.35, 0.0, 1.0)
    evidence_score = clamp(
        0.25 * coverage_score
        + 0.16 * diversity_score
        + 0.16 * average_relevance
        + 0.10 * average_recency_weight
        + 0.10 * stability_score
        + 0.08 * average_source_quality
        + 0.08 * provider_sentiment_coverage
        + 0.07 * concentration_score,
        0.0,
        1.0,
    )

    calibrated_sentiment = raw_sentiment * evidence_score
    news_score = sentiment_to_news_score(calibrated_sentiment)
    news_confidence = clamp(0.20 + evidence_score * 0.75, 0.20, 0.95)

    return {
        "raw_sentiment": raw_sentiment,
        "calibrated_sentiment": calibrated_sentiment,
        "news_score": news_score,
        "news_confidence": news_confidence,
        "effective_article_count": total_weight,
        "source_count": source_count,
        "average_relevance": average_relevance,
        "average_recency_weight": average_recency_weight,
        "average_source_quality": average_source_quality,
        "provider_sentiment_coverage": provider_sentiment_coverage,
        "dominant_weight_share": dominant_weight_share,
        "concentration_score": concentration_score,
        "sentiment_dispersion": dispersion,
        "dominant_signal": classify_dominant_signal(calibrated_sentiment, dispersion),
    }


def compute_finbert_sentiment(pipe, articles: List[NewsArticle]) -> Dict[str, float]:
    del pipe
    if not articles:
        return aggregate_news_signal([], [])
    sentiment_values = [article_sentiment_value(article) for article in articles]
    return aggregate_news_signal(articles, sentiment_values)


def summarize_texts(pipe, articles: List[NewsArticle]) -> str:
    del pipe
    if not articles:
        return ""

    highlight_fragments = [
        snippet
        for article in articles[:3]
        for snippet in article.highlight_snippets
        if snippet
    ]
    if highlight_fragments:
        return " ".join(highlight_fragments[:2])

    top_titles = [article.title[:140] for article in articles[:3]]
    return "; ".join(top_titles)


def llm_news_enabled() -> bool:
    raw = os.getenv(ENABLE_COMPANY_LLM_REVIEW_ENV, "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return False
    return bool(openai_api_key(required=False))


def build_company_news_prompt(
    symbol: str,
    company_name: str,
    articles: Sequence[NewsArticle],
) -> str:
    lines = [
        f"Company: {company_name} ({symbol})",
        "Horizon: next 1-10 trading days.",
        "Classify each article carefully.",
        "Use company_specific only when the article materially affects this exact company.",
        "Use sector_macro when the article is mainly macro/sector news but still likely relevant for the company.",
        "Use market_roundup for broad market recaps, ETF commentary, or weakly-related mentions.",
        "Use opinion_or_listicle for tips, rankings, comparisons, or editorial pieces with weak new information.",
        "Use irrelevant when the company mention is incidental.",
        "",
        "Articles:",
    ]

    for index, article in enumerate(articles, start=1):
        article_id = f"article_{index}"
        compact_text = article.text.replace("\n", " ").strip()[:700]
        lines.extend(
            [
                f"- article_id: {article_id}",
                f"  title: {article.title}",
                f"  provider: {article.provider}",
                f"  published_at: {article.published_at.isoformat().replace('+00:00', 'Z') if article.published_at else 'unknown'}",
                f"  text: {compact_text}",
            ]
        )

    return "\n".join(lines)


def build_company_news_review_schema() -> Dict[str, Any]:
    article_schema = {
        "type": "object",
        "properties": {
            "article_id": {"type": "string"},
            "doc_type": {
                "type": "string",
                "enum": list(COMPANY_NEWS_TYPE_WEIGHTS.keys()),
            },
            "company_relevance": {"type": "number"},
            "materiality": {"type": "number"},
            "impact_score": {"type": "number"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": [
            "article_id",
            "doc_type",
            "company_relevance",
            "materiality",
            "impact_score",
            "confidence",
            "reason",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "overall_signal": {
                "type": "string",
                "enum": ["bullish", "bearish", "neutral", "mixed"],
            },
            "overall_impact_score": {"type": "number"},
            "overall_confidence": {"type": "number"},
            "articles": {
                "type": "array",
                "items": article_schema,
            },
        },
        "required": [
            "summary",
            "overall_signal",
            "overall_impact_score",
            "overall_confidence",
            "articles",
        ],
        "additionalProperties": False,
    }


def request_company_news_review(
    symbol: str,
    company_name: str,
    articles: Sequence[NewsArticle],
) -> Dict[str, Any]:
    return request_structured_response(
        system_prompt=(
            "You are a financial news analyst. Return JSON only. "
            "Judge whether each article is actually useful for predicting this company's performance over the next 1-10 trading days. "
            "Down-weight broad market roundups, generic listicles, and incidental mentions."
        ),
        user_prompt=build_company_news_prompt(symbol, company_name, articles),
        schema_name="company_news_review",
        schema=build_company_news_review_schema(),
    )


def normalize_company_news_review(
    review: Dict[str, Any],
    articles: Sequence[NewsArticle],
) -> Tuple[List[ArticleReview], Dict[str, Any]]:
    valid_ids = {f"article_{index}": article for index, article in enumerate(articles, start=1)}
    normalized: Dict[str, ArticleReview] = {
        article_id: ArticleReview(
            article_id=article_id,
            doc_type="irrelevant",
            company_relevance=0.0,
            materiality=0.0,
            impact_score=0.0,
            confidence=0.0,
            reason="No usable company-specific impact extracted.",
        )
        for article_id in valid_ids
    }

    for item in review.get("articles", []) if isinstance(review.get("articles"), list) else []:
        if not isinstance(item, dict):
            continue
        article_id = str(item.get("article_id", "")).strip()
        if article_id not in valid_ids:
            continue
        doc_type = str(item.get("doc_type", "")).strip()
        if doc_type not in COMPANY_NEWS_TYPE_WEIGHTS:
            doc_type = "irrelevant"
        normalized[article_id] = ArticleReview(
            article_id=article_id,
            doc_type=doc_type,
            company_relevance=clamp(float(item.get("company_relevance", 0.0)), 0.0, 1.0),
            materiality=clamp(float(item.get("materiality", 0.0)), 0.0, 1.0),
            impact_score=clamp(float(item.get("impact_score", 0.0)), -1.0, 1.0),
            confidence=clamp(float(item.get("confidence", 0.0)), 0.0, 1.0),
            reason=str(item.get("reason", "")).strip() or "No explicit reason returned.",
        )

    normalized_meta = {
        "summary": str(review.get("summary", "")).strip(),
        "overall_signal": str(review.get("overall_signal", "neutral")).strip() or "neutral",
        "overall_impact_score": clamp(float(review.get("overall_impact_score", 0.0)), -1.0, 1.0),
        "overall_confidence": clamp(float(review.get("overall_confidence", 0.0)), 0.0, 1.0),
    }
    ordered = [normalized[f"article_{index}"] for index, _ in enumerate(articles, start=1)]
    return ordered, normalized_meta


def aggregate_llm_news_signal(
    articles: Sequence[NewsArticle],
    reviews: Sequence[ArticleReview],
    review_meta: Dict[str, Any],
) -> Dict[str, float]:
    if not articles or not reviews:
        return aggregate_news_signal([], [])

    if len(articles) != len(reviews):
        raise ValueError("articles and reviews must have the same length")

    effective_weights: List[float] = []
    llm_impacts: List[float] = []
    directness_values: List[float] = []
    materiality_values: List[float] = []
    llm_confidences: List[float] = []
    low_quality_count = 0

    for article, review in zip(articles, reviews):
        effective_weight = review_effective_weight(article, review)
        type_weight = COMPANY_NEWS_TYPE_WEIGHTS.get(review.doc_type, 0.0)
        if effective_weight <= 0.0 or type_weight <= 0.0:
            low_quality_count += 1
            effective_weights.append(0.0)
            llm_impacts.append(0.0)
            directness_values.append(0.0)
            materiality_values.append(0.0)
            llm_confidences.append(0.0)
            continue

        effective_weights.append(effective_weight)
        llm_impacts.append(review.impact_score)
        directness_values.append(review.company_relevance * type_weight)
        materiality_values.append(review.materiality)
        llm_confidences.append(review.confidence)
        if review.doc_type in {"market_roundup", "opinion_or_listicle"}:
            low_quality_count += 1

    total_effective_weight = sum(effective_weights)
    if total_effective_weight <= 1e-9:
        return {
            **aggregate_news_signal([], []),
            "llm_average_directness": 0.0,
            "llm_average_materiality": 0.0,
            "llm_average_confidence": 0.0,
            "llm_low_quality_share": 1.0 if reviews else 0.0,
        }

    article_level_sentiment = sum(
        impact * weight for impact, weight in zip(llm_impacts, effective_weights)
    ) / total_effective_weight
    overall_impact = float(review_meta.get("overall_impact_score", 0.0))
    overall_confidence = float(review_meta.get("overall_confidence", 0.0))
    raw_sentiment = clamp(
        article_level_sentiment * 0.80 + overall_impact * 0.20,
        -1.0,
        1.0,
    )

    average_relevance = sum(article.relevance_score for article in articles) / len(articles)
    average_recency_weight = sum(article.recency_weight for article in articles) / len(articles)
    average_source_quality = sum(article.source_quality for article in articles) / len(articles)
    source_count = len({article.provider.lower() for article in articles})
    provider_sentiment_coverage = sum(
        article.weight for article in articles if article.entity_sentiment is not None
    ) / max(sum(article.weight for article in articles), 1e-9)
    dominant_weight_share = max(effective_weights) / total_effective_weight
    dispersion = sum(
        abs(impact - raw_sentiment) * weight
        for impact, weight in zip(llm_impacts, effective_weights)
    ) / total_effective_weight

    llm_average_directness = sum(
        value * weight for value, weight in zip(directness_values, effective_weights)
    ) / total_effective_weight
    llm_average_materiality = sum(
        value * weight for value, weight in zip(materiality_values, effective_weights)
    ) / total_effective_weight
    llm_average_confidence = sum(
        value * weight for value, weight in zip(llm_confidences, effective_weights)
    ) / total_effective_weight
    llm_low_quality_share = low_quality_count / len(reviews)

    coverage_score = clamp(total_effective_weight / 1.8, 0.0, 1.0)
    diversity_score = clamp(source_count / min(max(len(articles), 1), 4), 0.0, 1.0)
    stability_score = 1.0 - clamp(dispersion / 1.0, 0.0, 1.0)
    concentration_score = 1.0 - clamp((dominant_weight_share - 0.45) / 0.35, 0.0, 1.0)
    evidence_score = clamp(
        0.22 * coverage_score
        + 0.15 * diversity_score
        + 0.15 * llm_average_directness
        + 0.12 * llm_average_materiality
        + 0.12 * llm_average_confidence
        + 0.08 * average_relevance
        + 0.06 * average_recency_weight
        + 0.05 * average_source_quality
        + 0.03 * provider_sentiment_coverage
        + 0.02 * concentration_score
        - 0.12 * llm_low_quality_share,
        0.0,
        1.0,
    )
    evidence_score = clamp(evidence_score * (0.85 + 0.15 * overall_confidence), 0.0, 1.0)

    calibrated_sentiment = raw_sentiment * evidence_score
    news_score = sentiment_to_news_score(calibrated_sentiment)
    news_confidence = clamp(
        0.20
        + evidence_score * 0.65
        + llm_average_confidence * 0.10
        + overall_confidence * 0.05,
        0.20,
        0.95,
    )

    return {
        "raw_sentiment": raw_sentiment,
        "calibrated_sentiment": calibrated_sentiment,
        "news_score": news_score,
        "news_confidence": news_confidence,
        "effective_article_count": total_effective_weight,
        "source_count": source_count,
        "average_relevance": average_relevance,
        "average_recency_weight": average_recency_weight,
        "average_source_quality": average_source_quality,
        "provider_sentiment_coverage": provider_sentiment_coverage,
        "dominant_weight_share": dominant_weight_share,
        "concentration_score": concentration_score,
        "sentiment_dispersion": dispersion,
        "dominant_signal": classify_dominant_signal(calibrated_sentiment, dispersion),
        "llm_average_directness": llm_average_directness,
        "llm_average_materiality": llm_average_materiality,
        "llm_average_confidence": llm_average_confidence,
        "llm_low_quality_share": llm_low_quality_share,
    }


def sentiment_to_news_score(raw_sent: float) -> float:
    x = clamp(raw_sent, -1.0, 1.0)
    return 0.5 + 0.5 * x


def review_effective_weight(article: NewsArticle, review: ArticleReview) -> float:
    type_weight = COMPANY_NEWS_TYPE_WEIGHTS.get(review.doc_type, 0.0)
    if type_weight <= 0.0:
        return 0.0
    return (
        article.weight
        * type_weight
        * max(0.15, review.company_relevance)
        * max(0.10, review.materiality)
        * max(0.20, review.confidence)
    )


def heuristic_signal_weight(article: NewsArticle, scale: float = 1.0) -> float:
    conviction = max(0.15, abs(article_sentiment_value(article)))
    return article.weight * conviction * max(scale, 0.0)


def build_signal_article_entries(
    primary_articles: Sequence[NewsArticle],
    reviews: Optional[Sequence[ArticleReview]] = None,
    overflow_articles: Optional[Sequence[NewsArticle]] = None,
    overflow_scale: float = 0.10,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if reviews is None:
        for article in primary_articles:
            signal_weight = heuristic_signal_weight(article)
            if signal_weight <= 0.0:
                continue
            entries.append({"article": article, "review": None, "signal_weight": signal_weight})
    else:
        for article, review in zip(primary_articles, reviews):
            signal_weight = review_effective_weight(article, review)
            if signal_weight <= 0.0:
                continue
            entries.append({"article": article, "review": review, "signal_weight": signal_weight})

    for article in overflow_articles or []:
        signal_weight = heuristic_signal_weight(article, scale=overflow_scale)
        if signal_weight <= 0.0:
            continue
        entries.append({"article": article, "review": None, "signal_weight": signal_weight})

    return entries


def select_supporting_article_entries(
    entries: Sequence[Dict[str, Any]],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and float(entry.get("signal_weight", 0.0)) > 0.0
        ),
        key=lambda entry: (
            float(entry["signal_weight"]),
            entry["article"].published_at or datetime.fromtimestamp(0, tz=timezone.utc),
        ),
        reverse=True,
    )
    if not ranked:
        return []

    total_weight = sum(float(entry["signal_weight"]) for entry in ranked)
    min_weight = max(
        SIGNAL_ARTICLE_MIN_WEIGHT,
        total_weight * SIGNAL_ARTICLE_MIN_SHARE,
        float(ranked[0]["signal_weight"]) * 0.20,
    )

    selected: List[Dict[str, Any]] = []
    cumulative_weight = 0.0
    for entry in ranked:
        if len(selected) >= limit:
            break
        if float(entry["signal_weight"]) < min_weight and len(selected) >= 2:
            break
        selected.append(entry)
        cumulative_weight += float(entry["signal_weight"])
        if cumulative_weight >= total_weight * SIGNAL_ARTICLE_COVERAGE_TARGET:
            break

    return selected


def build_news_reasons(
    symbol: str,
    company_name: str,
    summary: str,
    signal: Dict[str, float],
    articles: List[NewsArticle],
    reviews: Optional[Sequence[ArticleReview]] = None,
    llm_used: bool = False,
) -> List[str]:
    feeds_used = sorted({article.feed for article in articles if article.feed})
    engine_label = (
        "GPT-5 article classification and company-news provider coverage"
        if llm_used
        else "Company-news provider coverage"
    )
    reasons = [
        (
            f"{engine_label} rated the news flow {signal['dominant_signal']} at {signal['news_score']:.2f}, "
            f"after adjusting for relevance, recency, source quality, and story concentration."
        ),
        (
            f"Evidence quality: {len(articles)} articles across {signal['source_count']} sources, "
            f"{signal['effective_article_count']:.1f} effective weighted articles, "
            f"average relevance {signal['average_relevance']:.2f}, average source quality "
            f"{signal['average_source_quality']:.2f}, confidence {signal['news_confidence']:.2f}."
        ),
    ]

    if llm_used:
        reasons.append(
            (
                f"GPT-5 kept the weighted set focused on direct and material coverage: "
                f"average directness {signal.get('llm_average_directness', 0.0):.2f}, "
                f"materiality {signal.get('llm_average_materiality', 0.0):.2f}, "
                f"LLM confidence {signal.get('llm_average_confidence', 0.0):.2f}."
            )
        )
        if signal.get("llm_low_quality_share", 0.0) >= 0.34:
            reasons.append(
                "Several fetched articles looked like market roundups or listicles, so GPT-5 discounted them heavily."
            )

    if signal["provider_sentiment_coverage"] >= 0.60:
        reasons.append(
            "Most of the weighted coverage included entity-level sentiment metadata, so the signal depended less on generic headline parsing."
        )

    if signal["dominant_signal"] == "mixed":
        reasons.append(
            "Headline sentiment is mixed, so the model kept the net news effect closer to neutral."
        )

    if signal["dominant_weight_share"] > 0.60:
        reasons.append(
            "One story cluster dominated the evidence set, so confidence was capped until more independent coverage appears."
        )

    if feeds_used and feeds_used != ["marketaux"]:
        reasons.append(
            "Fallback company-news feeds contributed to this symbol's evidence set: "
            + ", ".join(feeds_used)
            + "."
        )

    if summary:
        reasons.append(f"News summary for {symbol} ({company_name}): {summary}")

    return reasons


def serialize_article_evidence(
    article: NewsArticle,
    review: Optional[ArticleReview] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "feed": article.feed,
        "title": article.title,
        "provider": article.provider,
        "url": article.url,
        "published_at": article.published_at.isoformat().replace("+00:00", "Z") if article.published_at else None,
        "relevance_score": round(article.relevance_score, 2),
        "sentiment": round(article_sentiment_value(article), 2),
    }
    if review is not None:
        payload["doc_type"] = review.doc_type
        payload["company_relevance"] = round(review.company_relevance, 2)
        payload["materiality"] = round(review.materiality, 2)
        payload["llm_confidence"] = round(review.confidence, 2)
        payload["llm_impact_score"] = round(review.impact_score, 2)
        payload["llm_reason"] = review.reason
    return payload


def heuristic_low_quality_share(
    articles: Sequence[NewsArticle],
    symbol: str,
    company_name: str,
) -> float:
    if not articles:
        return 0.0

    normalized_symbol = normalize_text(symbol)
    company_tokens = [
        token
        for token in normalize_text(company_name).split()
        if len(token) >= 4 and token not in {"corp", "inc", "group", "company", "corporation", "holdings"}
    ]
    low_quality_count = 0

    for article in articles:
        combined = normalize_text(f"{article.title} {article.text}")
        has_hint = any(hint in combined for hint in LOW_QUALITY_COMPANY_NEWS_HINTS)
        has_direct_mention = normalized_symbol in combined or any(token in combined for token in company_tokens[:3])
        if has_hint and (not has_direct_mention or article.relevance_score < COMPANY_HEURISTIC_LOW_QUALITY_RELEVANCE_CAP):
            low_quality_count += 1

    return low_quality_count / len(articles)


def latest_article_date(articles: Sequence[NewsArticle]) -> Optional[str]:
    published_dates = [
        article.published_at.astimezone(timezone.utc).date().isoformat()
        for article in articles
        if article.published_at is not None
    ]
    if not published_dates:
        return None
    return max(published_dates)


def neutral_news_payload(
    symbol: str,
    model_name: Optional[str],
    reason: Optional[str] = None,
    analysis_method: str = "neutral_no_articles",
    feeds_used: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "news_score": 0.5,
        "news_confidence": 0.20,
        "raw_sentiment": 0.0,
        "calibrated_sentiment": 0.0,
        "article_count": 0,
        "effective_article_count": 0.0,
        "source_count": 0,
        "average_relevance": 0.0,
        "average_recency_weight": 0.0,
        "average_source_quality": 0.0,
        "provider_sentiment_coverage": 0.0,
        "dominant_weight_share": 0.0,
        "dominant_signal": "neutral",
        "news_reasons": [
            reason
            or "No recently identified relevant company-news articles passed the relevance and freshness filters, so the news contribution is neutral."
        ],
        "top_headlines": [],
        "top_articles": [],
        "last_updated": None,
        "analysis_method": analysis_method,
        "llm_model": model_name,
        "feeds_used": list(feeds_used or []),
    }


def score_symbol_news(
    symbol: str,
    company_name: str,
    summarizer_pipe=None,
    *,
    llm_enabled: Optional[bool] = None,
    model_name: Optional[str] = None,
    company_llm_limit: Optional[int] = None,
) -> Dict[str, Any]:
    llm_enabled = llm_news_enabled() if llm_enabled is None else llm_enabled
    model_name = model_name if model_name is not None else (openai_model() if llm_enabled else None)
    company_llm_limit = company_llm_limit or configured_llm_news_limit()
    articles = fetch_symbol_news(symbol, company_name)

    if not articles:
        print(f"[{symbol}] No relevant recent company news from any configured provider - setting neutral 0.50")
        return neutral_news_payload(symbol, model_name)

    reviews: Optional[List[ArticleReview]] = None
    llm_used = False
    try:
        if llm_enabled:
            llm_used = True
            llm_review = request_company_news_review(
                symbol=symbol,
                company_name=company_name,
                articles=articles[:company_llm_limit],
            )
            reviews, review_meta = normalize_company_news_review(
                llm_review,
                articles[:company_llm_limit],
            )
            llm_signal = aggregate_llm_news_signal(
                articles[:company_llm_limit],
                reviews,
                review_meta,
            )

            if len(articles) > company_llm_limit:
                fallback_signal = aggregate_news_signal(
                    articles[company_llm_limit:],
                    [article_sentiment_value(article) for article in articles[company_llm_limit:]],
                )
                signal = {
                    **llm_signal,
                    "raw_sentiment": clamp(
                        llm_signal["raw_sentiment"] * 0.88 + fallback_signal["raw_sentiment"] * 0.12,
                        -1.0,
                        1.0,
                    ),
                    "calibrated_sentiment": clamp(
                        llm_signal["calibrated_sentiment"] * 0.90 + fallback_signal["calibrated_sentiment"] * 0.10,
                        -1.0,
                        1.0,
                    ),
                    "news_score": clamp(
                        llm_signal["news_score"] * 0.90 + fallback_signal["news_score"] * 0.10,
                        0.0,
                        1.0,
                    ),
                    "news_confidence": clamp(
                        llm_signal["news_confidence"] * 0.90 + fallback_signal["news_confidence"] * 0.10,
                        0.20,
                        0.95,
                    ),
                    "effective_article_count": llm_signal["effective_article_count"] + fallback_signal["effective_article_count"] * 0.10,
                }
            else:
                signal = llm_signal

            summary = review_meta["summary"] or summarize_texts(summarizer_pipe, articles)
        else:
            signal = compute_finbert_sentiment(None, articles)
            summary = summarize_texts(summarizer_pipe, articles)
    except Exception as exc:
        if llm_enabled and not allow_llm_fallback():
            raise
        llm_used = False
        reviews = None
        print(f"[WARN] [{symbol}] GPT company news review failed, using heuristic fallback: {exc}")
        record_runtime_event(
            "OpenAI company-news review failed and heuristic scoring was used instead.",
            provider="openai",
        )
        signal = compute_finbert_sentiment(None, articles)
        summary = summarize_texts(summarizer_pipe, articles)

    primary_articles = articles[:company_llm_limit] if llm_used else articles
    overflow_articles = articles[company_llm_limit:] if llm_used else []
    feeds_used = sorted({article.feed for article in articles if article.feed})
    heuristic_low_quality = heuristic_low_quality_share(articles, symbol, company_name)
    if (
        llm_used
        and signal.get("llm_low_quality_share", 0.0) >= COMPANY_LLM_LOW_QUALITY_SHARE_THRESHOLD
        and signal.get("llm_average_directness", 1.0) < COMPANY_LLM_MIN_DIRECTNESS
    ) or (
        not llm_used
        and heuristic_low_quality >= COMPANY_HEURISTIC_LOW_QUALITY_SHARE_THRESHOLD
        and signal.get("average_relevance", 0.0) < COMPANY_HEURISTIC_LOW_QUALITY_RELEVANCE_CAP
    ):
        reason = (
            f"Recent {symbol} company-news coverage was mostly broad market roundups or weakly direct mentions, "
            "so the company-news contribution was forced back to neutral."
        )
        return neutral_news_payload(
            symbol,
            model_name if llm_used else None,
            reason=reason,
            analysis_method="neutral_low_quality_company_news",
            feeds_used=feeds_used,
        )

    signal_article_entries = build_signal_article_entries(
        primary_articles=primary_articles,
        reviews=reviews if llm_used else None,
        overflow_articles=overflow_articles,
    )
    supporting_entries = select_supporting_article_entries(signal_article_entries)
    supporting_articles = [
        serialize_article_evidence(entry["article"], entry.get("review"))
        for entry in supporting_entries
    ]
    signal_last_updated = latest_article_date([entry["article"] for entry in supporting_entries])

    reasons = build_news_reasons(
        symbol,
        company_name,
        summary,
        signal,
        articles,
        reviews=reviews,
        llm_used=llm_used,
    )
    heuristic_method = "marketaux_entity_scoring" if feeds_used == ["marketaux"] else "company_provider_fallback_scoring"

    return {
        "news_score": round(signal["news_score"], 2),
        "news_confidence": round(signal["news_confidence"], 2),
        "raw_sentiment": round(signal["raw_sentiment"], 4),
        "calibrated_sentiment": round(signal["calibrated_sentiment"], 4),
        "article_count": len(articles),
        "effective_article_count": round(signal["effective_article_count"], 2),
        "source_count": signal["source_count"],
        "average_relevance": round(signal["average_relevance"], 2),
        "average_recency_weight": round(signal["average_recency_weight"], 2),
        "average_source_quality": round(signal["average_source_quality"], 2),
        "provider_sentiment_coverage": round(signal["provider_sentiment_coverage"], 2),
        "dominant_weight_share": round(signal["dominant_weight_share"], 2),
        "dominant_signal": signal["dominant_signal"],
        "news_reasons": reasons,
        "top_headlines": [article.title for article in articles[:3]],
        "top_articles": supporting_articles,
        "last_updated": signal_last_updated,
        "analysis_method": "gpt_company_review" if llm_used else heuristic_method,
        "llm_model": model_name if llm_used else None,
        "feeds_used": feeds_used,
    }


def main():
    set_pipeline_scope("news_scores")
    reset_marketaux_fetch_state()
    _, summarizer_pipe = init_models()
    universe = load_universe()
    llm_enabled = llm_news_enabled()
    model_name = openai_model() if llm_enabled else None
    company_llm_limit = configured_llm_news_limit()
    try:
        shortlist_symbols = set(select_company_news_shortlist(universe))
    except Exception as exc:
        record_runtime_event(
            f"Technical company-news shortlist failed, so the full universe was evaluated. Error: {exc}",
            provider="company_news_shortlist",
        )
        shortlist_symbols = {symbol for symbol, _ in universe}
    if not shortlist_symbols:
        shortlist_symbols = {symbol for symbol, _ in universe}

    out: Dict[str, dict] = {}

    for symbol, company_name in universe:
        if symbol not in shortlist_symbols:
            out[symbol] = neutral_news_payload(
                symbol,
                model_name,
                reason=(
                    f"Company news was only fetched for the weekly technical shortlist. "
                    f"{symbol} was outside the current shortlist, so its company-news contribution stayed neutral."
                ),
                analysis_method="neutral_shortlist_pruned",
            )
            continue
        out[symbol] = score_symbol_news(
            symbol=symbol,
            company_name=company_name,
            summarizer_pipe=summarizer_pipe,
            llm_enabled=llm_enabled,
            model_name=model_name,
            company_llm_limit=company_llm_limit,
        )

    payload = attach_data_quality(out, scope="news_scores")
    with open(NEWS_SCORES_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\nnews_scores.json updated.")


if __name__ == "__main__":
    main()
