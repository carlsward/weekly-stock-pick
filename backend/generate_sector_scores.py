import json
import math
import os
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from backend.generate_news_scores import (
        allow_marketaux_fallback,
        MARKETAUX_NEWS_ENDPOINT,
        MARKETAUX_REQUEST_TIMEOUT_SECONDS,
        NEWS_FETCH_ATTEMPTS,
        NEWS_FETCH_RETRY_SECONDS,
        clamp,
        compute_recency_weight,
        extract_title_and_text,
        marketaux_api_token,
        normalize_text,
        parse_iso_datetime,
        source_quality_weight,
    )
    from backend.sector_utils import (
        load_active_sectors,
        load_symbol_metadata,
        sector_display_name,
    )
    from backend.llm_utils import (
        allow_llm_fallback,
        openai_model,
        request_structured_response,
    )
    from backend.pipeline_runtime import (
        attach_data_quality,
        consume_provider_budget,
        record_runtime_event,
        set_pipeline_scope,
    )
except ImportError:
    from generate_news_scores import (
        allow_marketaux_fallback,
        MARKETAUX_NEWS_ENDPOINT,
        MARKETAUX_REQUEST_TIMEOUT_SECONDS,
        NEWS_FETCH_ATTEMPTS,
        NEWS_FETCH_RETRY_SECONDS,
        clamp,
        compute_recency_weight,
        extract_title_and_text,
        marketaux_api_token,
        normalize_text,
        parse_iso_datetime,
        source_quality_weight,
    )
    from sector_utils import (
        load_active_sectors,
        load_symbol_metadata,
        sector_display_name,
    )
    from llm_utils import (
        allow_llm_fallback,
        openai_model,
        request_structured_response,
    )
    from pipeline_runtime import (
        attach_data_quality,
        consume_provider_budget,
        record_runtime_event,
        set_pipeline_scope,
    )

SECTOR_SCORES_PATH = Path("sector_scores.json")
UNIVERSE_CSV_PATH_ENV = "UNIVERSE_CSV_PATH"
DEFAULT_UNIVERSE_CSV_PATH = Path("universe.csv")

GLOBAL_NEWS_LOOKBACK_DAYS = 3
MARKETAUX_GLOBAL_FIXTURE_PATH_ENV = "MARKETAUX_GLOBAL_FIXTURE_PATH"
GLOBAL_NEWS_LOOKBACK_DAYS_ENV = "GLOBAL_NEWS_LOOKBACK_DAYS"
GLOBAL_NEWS_LIMIT_ENV = "MARKETAUX_GLOBAL_NEWS_LIMIT"
GLOBAL_DEFAULT_NEWS_LIMIT = 3
GLOBAL_MAX_NEWS_LIMIT = 3
GDELT_GLOBAL_NEWS_LIMIT_ENV = "GDELT_GLOBAL_NEWS_LIMIT"
GDELT_DEFAULT_NEWS_LIMIT = 20
GDELT_MAX_NEWS_LIMIT = 40
ALPHA_VANTAGE_API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"
ALPHA_VANTAGE_GLOBAL_NEWS_LIMIT_ENV = "ALPHA_VANTAGE_GLOBAL_NEWS_LIMIT"
ALPHA_VANTAGE_DEFAULT_NEWS_LIMIT = 25
ALPHA_VANTAGE_MAX_NEWS_LIMIT = 50
GLOBAL_LLM_ARTICLE_LIMIT_ENV = "SECTOR_LLM_ARTICLE_LIMIT"
GLOBAL_DEFAULT_LLM_ARTICLE_LIMIT = 30
GLOBAL_MAX_LLM_ARTICLE_LIMIT = 60
GLOBAL_MIN_MACRO_RELEVANCE = 0.18
GLOBAL_BACKSTOP_SOURCE_QUALITY = 0.94
GLOBAL_BACKSTOP_RECENCY_WEIGHT = 0.80
GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
ALPHA_VANTAGE_NEWS_ENDPOINT = "https://www.alphavantage.co/query"
MARKETAUX_DAILY_REQUEST_LIMIT = 100
ALPHA_VANTAGE_DAILY_REQUEST_LIMIT = 25
GLOBAL_GDELT_ESCALATION_TARGET = 5
GLOBAL_ALPHA_VANTAGE_ESCALATION_TARGET = 10


def configured_global_news_lookback_days() -> int:
    raw_value = os.getenv(GLOBAL_NEWS_LOOKBACK_DAYS_ENV, "").strip()
    if not raw_value:
        return GLOBAL_NEWS_LOOKBACK_DAYS
    try:
        parsed = int(raw_value)
    except ValueError:
        return GLOBAL_NEWS_LOOKBACK_DAYS
    return max(1, min(parsed, 30))

SECTOR_KEYWORDS = {
    "communication_services": (
        "advertising",
        "streaming",
        "media",
        "telecom",
        "social media",
        "broadband",
        "wireless",
    ),
    "consumer_discretionary": (
        "consumer spending",
        "retail",
        "auto",
        "automotive",
        "travel",
        "airline",
        "hotel",
        "restaurant",
        "e-commerce",
    ),
    "consumer_staples": (
        "grocery",
        "food",
        "beverage",
        "household",
        "consumer staples",
        "pricing power",
    ),
    "energy": (
        "oil",
        "crude",
        "opec",
        "natural gas",
        "refinery",
        "energy",
        "fuel",
        "lng",
    ),
    "financials": (
        "bank",
        "lending",
        "credit",
        "interest rate",
        "yield",
        "capital markets",
        "insurance",
        "payment network",
    ),
    "healthcare": (
        "drug",
        "fda",
        "biotech",
        "pharma",
        "medicare",
        "healthcare",
        "medical device",
        "clinical trial",
    ),
    "industrials": (
        "aerospace",
        "defense",
        "manufacturing",
        "factory",
        "industrial",
        "shipping",
        "freight",
        "infrastructure",
    ),
    "technology": (
        "ai",
        "artificial intelligence",
        "chip",
        "semiconductor",
        "cloud",
        "software",
        "data center",
        "cybersecurity",
    ),
}

GLOBAL_CATALYST_KEYWORDS = (
    "oil",
    "crude",
    "opec",
    "hormuz",
    "red sea",
    "strait",
    "inflation",
    "cpi",
    "ppi",
    "interest rate",
    "central bank",
    "federal reserve",
    "fiscal",
    "budget",
    "tax",
    "subsidy",
    "stimulus",
    "tariff",
    "trade policy",
    "sanction",
    "export ban",
    "export control",
    "geopolitical",
    "political",
    "election",
    "war",
    "conflict",
    "ceasefire",
    "port closure",
    "port strike",
    "labor strike",
    "union",
    "walkout",
    "pipeline",
    "outage",
    "cyberattack",
    "hack",
    "data breach",
    "earthquake",
    "flood",
    "hurricane",
    "wildfire",
    "storm",
    "drought",
    "consumer spending",
    "retail sales",
    "recession",
    "regulation",
    "merger",
    "supply chain",
    "factory fire",
    "mine shutdown",
    "rare earth",
    "foundry",
    "wafer",
    "earnings outlook",
    "guidance",
    "sales",
    "deliveries",
    "demand",
    "production",
    "capacity",
    "orders",
    "backlog",
    "margin",
    "price cut",
    "shortage",
    "oversupply",
    "reimbursement",
    "capital spending",
    "capex",
    "factory",
    "drug approval",
    "clinical trial",
    "ai",
    "semiconductor",
    "defense",
    "shipping",
)

GDELT_QUERY_TERMS = (
    "\"oil\"",
    "\"opec\"",
    "\"hormuz\"",
    "\"red sea\"",
    "\"war\"",
    "\"conflict\"",
    "\"ceasefire\"",
    "\"election\"",
    "\"trade policy\"",
    "\"budget\"",
    "\"tax\"",
    "\"sanction\"",
    "\"tariff\"",
    "\"export ban\"",
    "\"export control\"",
    "\"interest rate\"",
    "\"inflation\"",
    "\"federal reserve\"",
    "\"consumer spending\"",
    "\"shipping disruption\"",
    "\"port strike\"",
    "\"pipeline outage\"",
    "\"supply chain\"",
    "\"semiconductor\"",
    "\"chip shortage\"",
    "\"rare earth\"",
    "\"factory fire\"",
    "\"mine shutdown\"",
    "\"ai spending\"",
    "\"defense spending\"",
    "\"cyberattack\"",
    "\"data breach\"",
    "\"earthquake\"",
    "\"flood\"",
    "\"hurricane\"",
    "\"wildfire\"",
    "\"tesla sales\"",
)

ALPHA_VANTAGE_TOPICS = (
    "economy_macro",
    "economy_monetary",
    "economy_fiscal",
    "financial_markets",
    "energy_transportation",
    "technology",
    "manufacturing",
)


@dataclass(frozen=True)
class GlobalNewsArticle:
    article_id: str
    title: str
    text: str
    published_at: Optional[datetime]
    provider: str
    url: Optional[str]
    macro_relevance: float
    recency_weight: float
    source_quality: float
    weight: float
    cluster_size: int
    feed: str = "marketaux"


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def latest_article_date(articles: Sequence["GlobalNewsArticle"]) -> Optional[str]:
    published_dates = [
        article.published_at.astimezone(timezone.utc).date().isoformat()
        for article in articles
        if article.published_at is not None
    ]
    if not published_dates:
        return None
    return max(published_dates)


def configured_global_news_limit() -> int:
    raw_limit = os.getenv(GLOBAL_NEWS_LIMIT_ENV, str(GLOBAL_DEFAULT_NEWS_LIMIT))
    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        parsed_limit = GLOBAL_DEFAULT_NEWS_LIMIT
    return max(1, min(parsed_limit, GLOBAL_MAX_NEWS_LIMIT))


def configured_gdelt_news_limit() -> int:
    raw_limit = os.getenv(GDELT_GLOBAL_NEWS_LIMIT_ENV, str(GDELT_DEFAULT_NEWS_LIMIT))
    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        parsed_limit = GDELT_DEFAULT_NEWS_LIMIT
    return max(10, min(parsed_limit, GDELT_MAX_NEWS_LIMIT))


def configured_alpha_vantage_news_limit() -> int:
    raw_limit = os.getenv(ALPHA_VANTAGE_GLOBAL_NEWS_LIMIT_ENV, str(ALPHA_VANTAGE_DEFAULT_NEWS_LIMIT))
    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        parsed_limit = ALPHA_VANTAGE_DEFAULT_NEWS_LIMIT
    return max(10, min(parsed_limit, ALPHA_VANTAGE_MAX_NEWS_LIMIT))


def configured_llm_article_limit() -> int:
    raw_limit = os.getenv(GLOBAL_LLM_ARTICLE_LIMIT_ENV, str(GLOBAL_DEFAULT_LLM_ARTICLE_LIMIT))
    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        parsed_limit = GLOBAL_DEFAULT_LLM_ARTICLE_LIMIT
    return max(6, min(parsed_limit, GLOBAL_MAX_LLM_ARTICLE_LIMIT))


def alpha_vantage_api_key(required: bool = False) -> str:
    key = os.getenv(ALPHA_VANTAGE_API_KEY_ENV, "").strip()
    if key or not required:
        return key
    raise RuntimeError(f"{ALPHA_VANTAGE_API_KEY_ENV} is required for Alpha Vantage world-news enrichment.")


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


def compute_macro_relevance(title: str, text: str, entity_count: int) -> float:
    normalized_title = normalize_text(title)
    normalized_text = normalize_text(text)
    combined = " ".join(item for item in (normalized_title, normalized_text) if item).strip()
    if not combined:
        return 0.0

    catalyst_hits_title = sum(1 for phrase in GLOBAL_CATALYST_KEYWORDS if phrase in normalized_title)
    catalyst_hits_text = sum(1 for phrase in GLOBAL_CATALYST_KEYWORDS if phrase in normalized_text)
    sector_hits = sum(
        1
        for keywords in SECTOR_KEYWORDS.values()
        if any(keyword in combined for keyword in keywords)
    )

    score = 0.0
    score += min(0.40, catalyst_hits_title * 0.12)
    score += min(0.22, catalyst_hits_text * 0.05)
    score += min(0.22, sector_hits * 0.07)

    if entity_count >= 4:
        score += 0.12
    elif entity_count >= 2:
        score += 0.07

    if any(term in combined for term in ("sector", "industry", "global", "world", "economy", "market")):
        score += 0.10

    if entity_count <= 1 and catalyst_hits_title + catalyst_hits_text <= 1 and sector_hits <= 1:
        score *= 0.50

    return clamp(score, 0.0, 1.0)


def build_global_marketaux_query(published_after: datetime) -> str:
    params = {
        "must_have_entities": "true",
        "group_similar": "true",
        "language": "en",
        "sort": "published_at",
        "sort_order": "desc",
        "published_after": published_after.strftime("%Y-%m-%dT%H:%M"),
        "limit": str(configured_global_news_limit()),
        "api_token": marketaux_api_token(),
    }
    return f"{MARKETAUX_NEWS_ENDPOINT}?{urlencode(params)}"


def fetch_marketaux_global_payload(published_after: datetime) -> List[dict]:
    url = build_global_marketaux_query(published_after)
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
                category="global_news",
                daily_limit=MARKETAUX_DAILY_REQUEST_LIMIT,
            ):
                print("[WARN] [sector] Marketaux workflow budget exhausted, returning fallback data.")
                return []
            with urlopen(request, timeout=MARKETAUX_REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise RuntimeError("Unexpected Marketaux global news response shape")
            return payload["data"]
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 402 and allow_marketaux_fallback():
                print("[WARN] [sector] Marketaux usage limit reached, returning neutral fallback data.")
                return []
            if exc.code == 429 and allow_marketaux_fallback():
                print("[WARN] [sector] Marketaux rate limit reached, returning neutral fallback data.")
                return []
            last_error = RuntimeError(f"Marketaux HTTP {exc.code} while fetching sector news: {body[:160]}")
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc

        if attempt < NEWS_FETCH_ATTEMPTS:
            time_module.sleep(NEWS_FETCH_RETRY_SECONDS * attempt)

    raise RuntimeError(
        f"Unable to fetch recent global news after {NEWS_FETCH_ATTEMPTS} attempts: {last_error}"
    )


def build_gdelt_query() -> str:
    query = "(" + " OR ".join(GDELT_QUERY_TERMS) + ")"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(configured_gdelt_news_limit()),
        "timespan": f"{configured_global_news_lookback_days()}days",
        "format": "json",
    }
    return f"{GDELT_DOC_ENDPOINT}?{urlencode(params)}"


def fetch_gdelt_payload() -> List[dict]:
    consume_provider_budget("gdelt", units=1, category="global_news")
    url = build_gdelt_query()
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
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict):
                if isinstance(payload.get("articles"), list):
                    return payload["articles"]
                if isinstance(payload.get("data"), list):
                    return payload["data"]
            raise RuntimeError("Unexpected GDELT response shape")
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < NEWS_FETCH_ATTEMPTS:
            time_module.sleep(NEWS_FETCH_RETRY_SECONDS * attempt)

    raise RuntimeError(
        f"Unable to fetch GDELT world news after {NEWS_FETCH_ATTEMPTS} attempts: {last_error}"
    )


def build_alpha_vantage_query(published_after: datetime) -> str:
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": ",".join(ALPHA_VANTAGE_TOPICS),
        "time_from": published_after.strftime("%Y%m%dT%H%M"),
        "sort": "RELEVANCE",
        "limit": str(configured_alpha_vantage_news_limit()),
        "apikey": alpha_vantage_api_key(required=True),
    }
    return f"{ALPHA_VANTAGE_NEWS_ENDPOINT}?{urlencode(params)}"


def fetch_alpha_vantage_payload(published_after: datetime) -> List[dict]:
    api_key = alpha_vantage_api_key(required=False)
    if not api_key:
        print("[WARN] [sector] Alpha Vantage API key missing, skipping NEWS_SENTIMENT feed.")
        return []

    url = build_alpha_vantage_query(published_after)
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
                "alpha_vantage",
                units=1,
                category="global_news",
                daily_limit=ALPHA_VANTAGE_DAILY_REQUEST_LIMIT,
            ):
                print("[WARN] [sector] Alpha Vantage workflow budget exhausted, skipping NEWS_SENTIMENT feed.")
                return []
            with urlopen(request, timeout=MARKETAUX_REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("feed"), list):
                return payload["feed"]
            if isinstance(payload, dict):
                message = (
                    str(payload.get("Information", "")).strip()
                    or str(payload.get("Note", "")).strip()
                    or str(payload.get("Error Message", "")).strip()
                )
                raise RuntimeError(message or "Unexpected Alpha Vantage response shape")
            raise RuntimeError("Unexpected Alpha Vantage response shape")
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        if attempt < NEWS_FETCH_ATTEMPTS:
            time_module.sleep(NEWS_FETCH_RETRY_SECONDS * attempt)

    raise RuntimeError(
        f"Unable to fetch Alpha Vantage world news after {NEWS_FETCH_ATTEMPTS} attempts: {last_error}"
    )


def normalize_alpha_vantage_item(item: dict) -> Optional[dict]:
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
    ticker_items = item.get("ticker_sentiment", [])
    entity_count = 0
    if isinstance(ticker_items, list):
        entity_count = len([entry for entry in ticker_items if isinstance(entry, dict)])
    entity_count = max(entity_count, min(4, len(topics)))
    topic_summary = f"Topics: {', '.join(topics[:4])}." if topics else ""
    sentiment_label = str(item.get("overall_sentiment_label", "")).strip()
    sentiment_summary = f"Overall sentiment label: {sentiment_label}." if sentiment_label else ""

    return {
        "feed": "alpha_vantage",
        "title": title,
        "description": summary,
        "snippet": " ".join(part for part in (topic_summary, sentiment_summary) if part).strip(),
        "published_at": iso_utc(published_at) if published_at else None,
        "source": provider,
        "url": url,
        "entities": [{} for _ in range(entity_count)],
        "similar": [],
    }


def normalize_gdelt_item(item: dict) -> Optional[dict]:
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
        "published_at": iso_utc(published_at) if published_at else None,
        "source": provider,
        "url": url,
        "entities": [],
        "similar": [],
    }


def build_global_article(item: dict, now: datetime, index: int) -> Optional[GlobalNewsArticle]:
    title, text = extract_title_and_text(item)
    if not text:
        return None

    published_at = parse_iso_datetime(item.get("published_at"))
    lookback_days = configured_global_news_lookback_days()
    if published_at is not None and published_at < now - timedelta(days=lookback_days):
        return None

    entity_count = len(item.get("entities", []) or [])
    macro_relevance = compute_macro_relevance(title, text, entity_count)

    provider = str(item.get("source", "") or "unknown source").strip() or "unknown source"
    url = item.get("url")
    recency_weight = compute_recency_weight(published_at, now)
    source_quality = source_quality_weight(provider, url)
    text_length_weight = 1.0 if len(text) >= 180 else 0.88
    cluster_size = 1 + len(item.get("similar", []) or [])
    cluster_support = min(1.0 + 0.06 * max(0, cluster_size - 1), 1.30)
    quality_backstop = (
        source_quality >= GLOBAL_BACKSTOP_SOURCE_QUALITY
        and recency_weight >= GLOBAL_BACKSTOP_RECENCY_WEIGHT
        and (entity_count >= 2 or cluster_size >= 2)
    )
    if macro_relevance < GLOBAL_MIN_MACRO_RELEVANCE and not quality_backstop:
        return None

    effective_macro_relevance = max(
        macro_relevance,
        GLOBAL_MIN_MACRO_RELEVANCE if quality_backstop else macro_relevance,
    )
    weight = effective_macro_relevance * recency_weight * source_quality * text_length_weight * cluster_support

    return GlobalNewsArticle(
        article_id=f"news_{index + 1}",
        title=title or f"news_{index + 1}",
        text=text,
        published_at=published_at,
        provider=provider,
        url=url if isinstance(url, str) else None,
        macro_relevance=effective_macro_relevance,
        recency_weight=recency_weight,
        source_quality=source_quality,
        weight=weight,
        cluster_size=cluster_size,
        feed=str(item.get("feed", "marketaux")).strip() or "marketaux",
    )


def count_relevant_global_articles(raw_sources: Dict[str, List[dict]], now: datetime) -> int:
    seen_titles = set()
    count = 0
    article_index = 0
    for items in raw_sources.values():
        for item in items:
            if not isinstance(item, dict):
                continue
            article = build_global_article(item, now, article_index)
            article_index += 1
            if article is None:
                continue
            title_key = normalize_text(article.title)
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            count += 1
    return count


def fetch_recent_global_articles() -> List[GlobalNewsArticle]:
    configured_fixture = os.getenv(MARKETAUX_GLOBAL_FIXTURE_PATH_ENV, "").strip()
    now = datetime.now(timezone.utc)
    if configured_fixture:
        fixture_path = Path(configured_fixture)
        with fixture_path.open("r", encoding="utf-8") as handle:
            raw_news = json.load(handle)
        if not isinstance(raw_news, list):
            raise RuntimeError(f"Global Marketaux fixture must be a list in {fixture_path}")
        print(f"[INFO] [sector] Loaded {len(raw_news)} fixture news items from {fixture_path}.")
        raw_sources = {"fixture": raw_news}
    else:
        published_after = datetime.now(timezone.utc) - timedelta(days=configured_global_news_lookback_days())
        raw_sources = {}
        if not os.getenv("MARKETAUX_API_TOKEN", "").strip() and allow_marketaux_fallback():
            print("[WARN] [sector] Marketaux token missing, continuing with other world-news sources.")
            raw_sources["marketaux"] = []
        else:
            try:
                raw_sources["marketaux"] = fetch_marketaux_global_payload(published_after)
            except Exception as exc:
                print(f"[WARN] [sector] Marketaux global feed failed: {exc}")
                record_runtime_event(
                    "Marketaux global-news feed failed and fallback feeds were used.",
                    provider="marketaux",
                )
                raw_sources["marketaux"] = []

        if count_relevant_global_articles({"marketaux": raw_sources.get("marketaux", [])}, now) < GLOBAL_GDELT_ESCALATION_TARGET:
            try:
                raw_sources["gdelt"] = [
                    item
                    for item in (
                        normalize_gdelt_item(raw_item)
                        for raw_item in fetch_gdelt_payload()
                    )
                    if item is not None
                ]
            except Exception as exc:
                print(f"[WARN] [sector] GDELT world-news feed failed: {exc}")
                record_runtime_event(
                    "GDELT global-news feed failed during this run.",
                    provider="gdelt",
                )
                raw_sources["gdelt"] = []
        else:
            raw_sources["gdelt"] = []

        if count_relevant_global_articles(
            {
                "marketaux": raw_sources.get("marketaux", []),
                "gdelt": raw_sources.get("gdelt", []),
            },
            now,
        ) < GLOBAL_ALPHA_VANTAGE_ESCALATION_TARGET:
            try:
                raw_sources["alpha_vantage"] = [
                    item
                    for item in (
                        normalize_alpha_vantage_item(raw_item)
                        for raw_item in fetch_alpha_vantage_payload(published_after)
                    )
                    if item is not None
                ]
            except Exception as exc:
                print(f"[WARN] [sector] Alpha Vantage NEWS_SENTIMENT feed failed: {exc}")
                record_runtime_event(
                    "Alpha Vantage global-news fallback failed during this run.",
                    provider="alpha_vantage",
                )
                raw_sources["alpha_vantage"] = []
        else:
            raw_sources["alpha_vantage"] = []

    raw_news = [
        item
        for items in raw_sources.values()
        for item in items
        if isinstance(item, dict)
    ]

    print("\n=== Fetching recent global sector news ===")
    print(f"[sector] raw news count: {len(raw_news)}")
    for source_name, items in raw_sources.items():
        print(f"[sector] {source_name} items: {len(items)}")

    articles: List[GlobalNewsArticle] = []
    seen_titles = set()
    for index, item in enumerate(raw_news):
        article = build_global_article(item, now, index)
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
            article.published_at or datetime.fromtimestamp(0, tz=timezone.utc),
        ),
        reverse=True,
    )

    print(f"[sector] relevant macro/sector articles used: {len(articles)}")
    return articles


def select_articles_for_llm(
    articles: Sequence[GlobalNewsArticle],
    limit: int,
) -> List[GlobalNewsArticle]:
    if limit <= 0 or not articles:
        return []

    balanced_feed_cap = max(3, limit // 3)
    gdelt_cap = min(10, balanced_feed_cap)
    alpha_cap = min(10, balanced_feed_cap)
    marketaux_cap = max(1, limit - gdelt_cap - alpha_cap)
    feed_caps = {
        "gdelt": gdelt_cap,
        "alpha_vantage": alpha_cap,
        "marketaux": marketaux_cap,
        "fixture": limit,
    }
    selected: List[GlobalNewsArticle] = []
    selected_titles = set()
    feed_counts: Dict[str, int] = {}

    for article in articles:
        if len(selected) >= limit:
            break
        title_key = normalize_text(article.title)
        if title_key in selected_titles:
            continue
        feed = article.feed or "marketaux"
        cap = feed_caps.get(feed, limit)
        if feed_counts.get(feed, 0) >= cap:
            continue
        selected.append(article)
        selected_titles.add(title_key)
        feed_counts[feed] = feed_counts.get(feed, 0) + 1

    if len(selected) < limit:
        for article in articles:
            if len(selected) >= limit:
                break
            title_key = normalize_text(article.title)
            if title_key in selected_titles:
                continue
            selected.append(article)
            selected_titles.add(title_key)

    return selected


def build_analysis_prompt(
    articles: Sequence[GlobalNewsArticle],
    sectors: Sequence[str],
    symbol_metadata: Dict[str, Dict[str, str]],
) -> str:
    lines = [
        "Review the following recent world, macro, geopolitical, policy, supply-chain, sector, and company-impact news items.",
        "Infer the causal chain from event to market impact before assigning winners and losers.",
        "Think in terms of transmission such as supply shocks, pricing power, fuel costs, demand shifts, rates, regulation, capex, reimbursement, logistics, labor disruption, cyber risk, weather, natural disasters, and geopolitics.",
        "Example: a Hormuz disruption can imply tighter oil supply, higher crude prices, stronger upstream energy revenue, and pressure on fuel-intensive businesses.",
        "Include broad catalysts such as elections, sanctions, tariffs, export controls, shipping disruptions, strikes, natural disasters, cyberattacks, and important company sales or guidance shocks when they can move sectors.",
        "For each item, decide which tracked sectors and tracked symbols are likely beneficiaries or losers over the next 1-15 trading days.",
        "Use tracked symbols only when the article clearly affects a specific company or a narrow subset of tracked companies.",
        "If the transmission is broad but not symbol-specific, prefer sectors over symbols.",
        "If an item is too narrow, stale, or not useful for short-horizon market impact, keep beneficiary and hurt lists empty and set low market_relevance.",
        "",
        "Allowed sectors:",
        ", ".join(f"{sector} ({sector_display_name(sector)})" for sector in sectors),
        "",
        "Tracked symbols:",
    ]

    for symbol, meta in symbol_metadata.items():
        lines.append(
            f"- {symbol}: {meta['company_name']} [{sector_display_name(meta['sector'])}]"
        )

    lines.extend(
        [
            "",
        "News items:",
        ]
    )

    for article in articles:
        published_at = iso_utc(article.published_at) if article.published_at else "unknown"
        compact_text = article.text.replace("\n", " ").strip()[:420]
        lines.extend(
            [
                f"- article_id: {article.article_id}",
                f"  title: {article.title}",
                f"  provider: {article.provider}",
                f"  published_at: {published_at}",
                f"  summary: {compact_text}",
            ]
        )

    return "\n".join(lines)


def build_sector_review_schema(
    sectors: Sequence[str],
    tracked_symbols: Sequence[str],
) -> Dict[str, Any]:
    article_schema = {
        "type": "object",
        "properties": {
            "article_id": {"type": "string"},
            "event_type": {"type": "string"},
            "transmission_channel": {"type": "string"},
            "affected_inputs": {
                "type": "array",
                "items": {"type": "string"},
            },
            "horizon": {
                "type": "string",
                "enum": ["1-3d", "1w", "1-2w", "2-4w", "unclear"],
            },
            "market_relevance": {"type": "number"},
            "magnitude": {"type": "number"},
            "confidence": {"type": "number"},
            "beneficiary_sectors": {
                "type": "array",
                "items": {"type": "string", "enum": list(sectors)},
            },
            "hurt_sectors": {
                "type": "array",
                "items": {"type": "string", "enum": list(sectors)},
            },
            "beneficiary_symbols": {
                "type": "array",
                "items": {"type": "string", "enum": list(tracked_symbols)},
            },
            "hurt_symbols": {
                "type": "array",
                "items": {"type": "string", "enum": list(tracked_symbols)},
            },
            "reason": {"type": "string"},
        },
        "required": [
            "article_id",
            "event_type",
            "transmission_channel",
            "affected_inputs",
            "horizon",
            "market_relevance",
            "magnitude",
            "confidence",
            "beneficiary_sectors",
            "hurt_sectors",
            "beneficiary_symbols",
            "hurt_symbols",
            "reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "articles": {
                "type": "array",
                "items": article_schema,
            },
        },
        "required": ["summary", "articles"],
        "additionalProperties": False,
    }


def request_sector_review(
    articles: Sequence[GlobalNewsArticle],
    sectors: Sequence[str],
    symbol_metadata: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    tracked_symbols = list(symbol_metadata.keys())
    return request_structured_response(
        system_prompt=(
            "You are a financial market-impact analyst. Return JSON only. "
            "Focus on short-horizon effects from broad world news: macro, politics, geopolitics, war, elections, energy, supply shocks, demand shifts, regulation, labor issues, natural disasters, cyber incidents, and important company events. "
            "Extract the event type and transmission channel explicitly before mapping winners and losers. "
            "Reason through first-order and second-order effects on sectors before mapping tracked symbols. "
            "Map impacts to the allowed sectors and tracked symbols only. "
            "Do not recommend trades directly."
        ),
        user_prompt=build_analysis_prompt(articles, sectors, symbol_metadata),
        schema_name="sector_impact_review",
        schema=build_sector_review_schema(sectors, tracked_symbols),
    )


def normalize_review_payload(
    review: Dict[str, Any],
    articles: Sequence[GlobalNewsArticle],
    sectors: Sequence[str],
    tracked_symbols: Sequence[str],
) -> List[Dict[str, Any]]:
    valid_ids = {article.article_id for article in articles}
    valid_sectors = set(sectors)
    valid_symbols = set(tracked_symbols)

    normalized: Dict[str, Dict[str, Any]] = {
        article.article_id: {
            "article_id": article.article_id,
            "event_type": "unclear",
            "transmission_channel": "",
            "affected_inputs": [],
            "horizon": "unclear",
            "market_relevance": 0.0,
            "magnitude": 0.0,
            "confidence": 0.0,
            "beneficiary_sectors": [],
            "hurt_sectors": [],
            "beneficiary_symbols": [],
            "hurt_symbols": [],
            "reason": "No usable sector impact extracted.",
        }
        for article in articles
    }

    for item in review.get("articles", []) if isinstance(review.get("articles"), list) else []:
        if not isinstance(item, dict):
            continue
        article_id = str(item.get("article_id", "")).strip()
        if article_id not in valid_ids:
            continue

        event_type = str(item.get("event_type", "")).strip() or "unclear"
        transmission_channel = str(item.get("transmission_channel", "")).strip()
        affected_inputs = [
            str(value).strip()
            for value in item.get("affected_inputs", [])
            if str(value).strip()
        ]
        horizon = str(item.get("horizon", "")).strip()
        if horizon not in {"1-3d", "1w", "1-2w", "2-4w", "unclear"}:
            horizon = "unclear"
        beneficiary = [
            sector
            for sector in item.get("beneficiary_sectors", [])
            if isinstance(sector, str) and sector in valid_sectors
        ]
        hurt = [
            sector
            for sector in item.get("hurt_sectors", [])
            if isinstance(sector, str) and sector in valid_sectors
        ]
        beneficiary_symbols = [
            symbol
            for symbol in item.get("beneficiary_symbols", [])
            if isinstance(symbol, str) and symbol in valid_symbols
        ]
        hurt_symbols = [
            symbol
            for symbol in item.get("hurt_symbols", [])
            if isinstance(symbol, str) and symbol in valid_symbols
        ]
        normalized[article_id] = {
            "article_id": article_id,
            "event_type": event_type,
            "transmission_channel": transmission_channel,
            "affected_inputs": list(dict.fromkeys(affected_inputs)),
            "horizon": horizon,
            "market_relevance": clamp(float(item.get("market_relevance", 0.0)), 0.0, 1.0),
            "magnitude": clamp(float(item.get("magnitude", 0.0)), 0.0, 1.0),
            "confidence": clamp(float(item.get("confidence", 0.0)), 0.0, 1.0),
            "beneficiary_sectors": list(dict.fromkeys(beneficiary)),
            "hurt_sectors": list(dict.fromkeys(hurt)),
            "beneficiary_symbols": list(dict.fromkeys(beneficiary_symbols)),
            "hurt_symbols": list(dict.fromkeys(hurt_symbols)),
            "reason": str(item.get("reason", "")).strip() or "No explicit reason returned.",
        }

    return [normalized[article.article_id] for article in articles]


def classify_direction(score: float, confidence: float) -> str:
    if confidence < 0.30:
        return "neutral"
    if score >= 0.57:
        return "bullish"
    if score <= 0.43:
        return "bearish"
    return "neutral"


def aggregate_sector_scores(
    articles: Sequence[GlobalNewsArticle],
    reviews: Sequence[Dict[str, Any]],
    sectors: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    sector_totals: Dict[str, Dict[str, Any]] = {
        sector: {
            "positive_weight": 0.0,
            "negative_weight": 0.0,
            "confidence_weight": 0.0,
            "article_count": 0,
            "contributions": [],
        }
        for sector in sectors
    }

    review_by_id = {review["article_id"]: review for review in reviews}
    for article in articles:
        review = review_by_id.get(article.article_id)
        if not review:
            continue

        market_relevance = clamp(float(review["market_relevance"]), 0.0, 1.0)
        magnitude = clamp(float(review["magnitude"]), 0.0, 1.0)
        confidence = clamp(float(review["confidence"]), 0.0, 1.0)
        effective_weight = article.weight * market_relevance * magnitude
        if effective_weight <= 0.0:
            continue

        beneficiaries = review["beneficiary_sectors"]
        losers = review["hurt_sectors"]

        if beneficiaries:
            positive_share = effective_weight / len(beneficiaries)
            for sector in beneficiaries:
                sector_totals[sector]["positive_weight"] += positive_share
                sector_totals[sector]["confidence_weight"] += positive_share * confidence
                sector_totals[sector]["article_count"] += 1
                sector_totals[sector]["contributions"].append(
                    {
                        "impact": "positive",
                        "weight": positive_share,
                        "confidence": confidence,
                        "event_type": str(review.get("event_type", "")).strip() or "unclear",
                        "transmission_channel": str(review.get("transmission_channel", "")).strip(),
                        "affected_inputs": list(review.get("affected_inputs", []))
                        if isinstance(review.get("affected_inputs"), list)
                        else [],
                        "horizon": str(review.get("horizon", "")).strip() or "unclear",
                        "reason": review["reason"],
                        "article": article,
                    }
                )

        if losers:
            negative_share = effective_weight / len(losers)
            for sector in losers:
                sector_totals[sector]["negative_weight"] += negative_share
                sector_totals[sector]["confidence_weight"] += negative_share * confidence
                sector_totals[sector]["article_count"] += 1
                sector_totals[sector]["contributions"].append(
                    {
                        "impact": "negative",
                        "weight": negative_share,
                        "confidence": confidence,
                        "event_type": str(review.get("event_type", "")).strip() or "unclear",
                        "transmission_channel": str(review.get("transmission_channel", "")).strip(),
                        "affected_inputs": list(review.get("affected_inputs", []))
                        if isinstance(review.get("affected_inputs"), list)
                        else [],
                        "horizon": str(review.get("horizon", "")).strip() or "unclear",
                        "reason": review["reason"],
                        "article": article,
                    }
                )

    aggregated: Dict[str, Dict[str, Any]] = {}
    for sector, totals in sector_totals.items():
        positive = totals["positive_weight"]
        negative = totals["negative_weight"]
        coverage = positive + negative
        average_confidence = (
            totals["confidence_weight"] / coverage
            if coverage > 1e-9
            else 0.0
        )
        net = positive - negative
        score = 0.5 + 0.45 * math.tanh(net / 0.85) if coverage > 0 else 0.5
        confidence = clamp(
            0.20
            + min(coverage, 1.8) / 1.8 * 0.45
            + average_confidence * 0.20
            + min(totals["article_count"], 6) / 6 * 0.10,
            0.20,
            0.95,
        )

        ranked_contributions = sorted(
            totals["contributions"],
            key=lambda item: item["weight"] * max(0.25, item["confidence"]),
            reverse=True,
        )
        top_reasons = [item["reason"] for item in ranked_contributions[:2]]
        supporting_articles = [
            {
                "title": contribution["article"].title,
                "feed": contribution["article"].feed,
                "provider": contribution["article"].provider,
                "url": contribution["article"].url,
                "published_at": (
                    iso_utc(contribution["article"].published_at)
                    if contribution["article"].published_at
                    else None
                ),
                "impact": contribution["impact"],
                "weight": round(contribution["weight"], 3),
                "event_type": contribution["event_type"],
                "transmission_channel": contribution["transmission_channel"],
                "affected_inputs": contribution["affected_inputs"],
                "horizon": contribution["horizon"],
                "reason": contribution["reason"],
            }
            for contribution in ranked_contributions[:3]
        ]
        last_updated = (
            latest_article_date([item["article"] for item in ranked_contributions[:3]])
            if coverage > 0
            else None
        )

        aggregated[sector] = {
            "sector": sector,
            "display_name": sector_display_name(sector),
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "direction": classify_direction(score, confidence),
            "last_updated": last_updated,
            "reasons": top_reasons
            if top_reasons
            else [f"No strong recent sector catalyst was found for {sector_display_name(sector)}."],
            "supporting_articles": supporting_articles,
        }

    return aggregated


def aggregate_symbol_scores(
    articles: Sequence[GlobalNewsArticle],
    reviews: Sequence[Dict[str, Any]],
    symbol_metadata: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    symbol_totals: Dict[str, Dict[str, Any]] = {
        symbol: {
            "positive_weight": 0.0,
            "negative_weight": 0.0,
            "confidence_weight": 0.0,
            "article_count": 0,
            "contributions": [],
        }
        for symbol in symbol_metadata
    }

    review_by_id = {review["article_id"]: review for review in reviews}
    for article in articles:
        review = review_by_id.get(article.article_id)
        if not review:
            continue

        market_relevance = clamp(float(review["market_relevance"]), 0.0, 1.0)
        magnitude = clamp(float(review["magnitude"]), 0.0, 1.0)
        confidence = clamp(float(review["confidence"]), 0.0, 1.0)
        effective_weight = article.weight * market_relevance * magnitude
        if effective_weight <= 0.0:
            continue

        beneficiaries = review["beneficiary_symbols"]
        losers = review["hurt_symbols"]

        if beneficiaries:
            positive_share = effective_weight / len(beneficiaries)
            for symbol in beneficiaries:
                symbol_totals[symbol]["positive_weight"] += positive_share
                symbol_totals[symbol]["confidence_weight"] += positive_share * confidence
                symbol_totals[symbol]["article_count"] += 1
                symbol_totals[symbol]["contributions"].append(
                    {
                        "impact": "positive",
                        "weight": positive_share,
                        "confidence": confidence,
                        "event_type": str(review.get("event_type", "")).strip() or "unclear",
                        "transmission_channel": str(review.get("transmission_channel", "")).strip(),
                        "affected_inputs": list(review.get("affected_inputs", []))
                        if isinstance(review.get("affected_inputs"), list)
                        else [],
                        "horizon": str(review.get("horizon", "")).strip() or "unclear",
                        "reason": review["reason"],
                        "article": article,
                    }
                )

        if losers:
            negative_share = effective_weight / len(losers)
            for symbol in losers:
                symbol_totals[symbol]["negative_weight"] += negative_share
                symbol_totals[symbol]["confidence_weight"] += negative_share * confidence
                symbol_totals[symbol]["article_count"] += 1
                symbol_totals[symbol]["contributions"].append(
                    {
                        "impact": "negative",
                        "weight": negative_share,
                        "confidence": confidence,
                        "event_type": str(review.get("event_type", "")).strip() or "unclear",
                        "transmission_channel": str(review.get("transmission_channel", "")).strip(),
                        "affected_inputs": list(review.get("affected_inputs", []))
                        if isinstance(review.get("affected_inputs"), list)
                        else [],
                        "horizon": str(review.get("horizon", "")).strip() or "unclear",
                        "reason": review["reason"],
                        "article": article,
                    }
                )

    aggregated: Dict[str, Dict[str, Any]] = {}
    for symbol, metadata in symbol_metadata.items():
        totals = symbol_totals[symbol]
        positive = totals["positive_weight"]
        negative = totals["negative_weight"]
        coverage = positive + negative
        average_confidence = (
            totals["confidence_weight"] / coverage
            if coverage > 1e-9
            else 0.0
        )
        net = positive - negative
        score = 0.5 + 0.45 * math.tanh(net / 0.70) if coverage > 0 else 0.5
        confidence = clamp(
            0.20
            + min(coverage, 1.5) / 1.5 * 0.45
            + average_confidence * 0.22
            + min(totals["article_count"], 4) / 4 * 0.08,
            0.20,
            0.95,
        )

        ranked_contributions = sorted(
            totals["contributions"],
            key=lambda item: item["weight"] * max(0.25, item["confidence"]),
            reverse=True,
        )
        top_reasons = [item["reason"] for item in ranked_contributions[:2]]
        supporting_articles = [
            {
                "title": contribution["article"].title,
                "feed": contribution["article"].feed,
                "provider": contribution["article"].provider,
                "url": contribution["article"].url,
                "published_at": (
                    iso_utc(contribution["article"].published_at)
                    if contribution["article"].published_at
                    else None
                ),
                "impact": contribution["impact"],
                "weight": round(contribution["weight"], 3),
                "event_type": contribution["event_type"],
                "transmission_channel": contribution["transmission_channel"],
                "affected_inputs": contribution["affected_inputs"],
                "horizon": contribution["horizon"],
                "reason": contribution["reason"],
            }
            for contribution in ranked_contributions[:3]
        ]
        last_updated = (
            latest_article_date([item["article"] for item in ranked_contributions[:3]])
            if coverage > 0
            else None
        )

        aggregated[symbol] = {
            "symbol": symbol,
            "company_name": metadata["company_name"],
            "sector": metadata["sector"],
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "direction": classify_direction(score, confidence),
            "last_updated": last_updated,
            "reasons": top_reasons
            if top_reasons
            else [f"No strong recent world-news catalyst clearly favored {metadata['company_name']}."],
            "supporting_articles": supporting_articles,
        }

    return aggregated


def build_neutral_sector_payload(
    sectors: Sequence[str],
    symbol_metadata: Dict[str, Dict[str, str]],
    generated_at: datetime,
    articles: Sequence[GlobalNewsArticle],
    model_name: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    neutral_scores = {
        sector: {
            "sector": sector,
            "display_name": sector_display_name(sector),
            "score": 0.5,
            "confidence": 0.2,
            "direction": "neutral",
            "last_updated": None,
            "reasons": [
                error
                if error
                else f"No recent broad catalyst clearly favored the {sector_display_name(sector)} sector."
            ],
            "supporting_articles": [],
        }
        for sector in sectors
    }
    neutral_symbol_scores = {
        symbol: {
            "symbol": symbol,
            "company_name": metadata["company_name"],
            "sector": metadata["sector"],
            "score": 0.5,
            "confidence": 0.2,
            "direction": "neutral",
            "last_updated": None,
            "reasons": [
                error
                if error
                else f"No recent world-news catalyst clearly favored {metadata['company_name']}."
            ],
            "supporting_articles": [],
        }
        for symbol, metadata in symbol_metadata.items()
    }

    return {
        "generated_at": iso_utc(generated_at),
        "last_updated": latest_article_date(articles) or generated_at.date().isoformat(),
        "lookback_days": configured_global_news_lookback_days(),
        "article_count": len(articles),
        "source_count": len({article.provider.lower() for article in articles}),
        "llm_model": model_name,
        "summary": error or "Sector analysis was neutral because no reliable broad catalysts were found.",
        "sector_scores": neutral_scores,
        "symbol_scores": neutral_symbol_scores,
        "events": [],
    }


def resolve_universe_csv_path() -> Path:
    configured = os.getenv(UNIVERSE_CSV_PATH_ENV, "").strip()
    return Path(configured) if configured else DEFAULT_UNIVERSE_CSV_PATH


def build_sector_scores_payload(
    *,
    articles: Sequence[GlobalNewsArticle],
    symbol_metadata: Dict[str, Dict[str, str]],
    sectors: Sequence[str],
    generated_at: Optional[datetime] = None,
    llm_article_limit: Optional[int] = None,
) -> Dict[str, Any]:
    effective_generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0)
    model_name = openai_model()

    if not articles:
        return build_neutral_sector_payload(
            sectors=sectors,
            symbol_metadata=symbol_metadata,
            generated_at=effective_generated_at,
            articles=[],
            model_name=model_name,
        )

    articles_for_llm = select_articles_for_llm(
        articles,
        llm_article_limit or configured_llm_article_limit(),
    )
    try:
        raw_review = request_sector_review(articles_for_llm, sectors, symbol_metadata)
        normalized_reviews = normalize_review_payload(
            raw_review,
            articles_for_llm,
            sectors,
            list(symbol_metadata.keys()),
        )
        sector_scores = aggregate_sector_scores(articles_for_llm, normalized_reviews, sectors)
        symbol_scores = aggregate_symbol_scores(articles_for_llm, normalized_reviews, symbol_metadata)
        return {
            "generated_at": iso_utc(effective_generated_at),
            "last_updated": latest_article_date(articles_for_llm) or effective_generated_at.date().isoformat(),
            "lookback_days": configured_global_news_lookback_days(),
            "article_count": len(articles_for_llm),
            "source_count": len({article.provider.lower() for article in articles_for_llm}),
            "llm_model": model_name,
            "summary": str(raw_review.get("summary", "")).strip()
            or "Recent broad market news was classified into sector winners and losers.",
            "sector_scores": sector_scores,
            "symbol_scores": symbol_scores,
            "events": [
                {
                    "article_id": review["article_id"],
                    "title": article.title,
                    "feed": article.feed,
                    "provider": article.provider,
                    "url": article.url,
                    "published_at": iso_utc(article.published_at) if article.published_at else None,
                    "event_type": review["event_type"],
                    "transmission_channel": review["transmission_channel"],
                    "affected_inputs": review["affected_inputs"],
                    "horizon": review["horizon"],
                    "market_relevance": round(review["market_relevance"], 2),
                    "magnitude": round(review["magnitude"], 2),
                    "confidence": round(review["confidence"], 2),
                    "beneficiary_sectors": review["beneficiary_sectors"],
                    "hurt_sectors": review["hurt_sectors"],
                    "beneficiary_symbols": review["beneficiary_symbols"],
                    "hurt_symbols": review["hurt_symbols"],
                    "reason": review["reason"],
                }
                for article, review in zip(articles_for_llm, normalized_reviews)
            ],
        }
    except Exception as exc:
        if not allow_llm_fallback():
            raise
        warning = (
            "Sector LLM analysis failed, so the weekly sector overlay was set to neutral. "
            f"Error: {exc}"
        )
        print(f"[WARN] {warning}")
        record_runtime_event(
            warning,
            provider="openai",
        )
        return build_neutral_sector_payload(
            sectors=sectors,
            symbol_metadata=symbol_metadata,
            generated_at=effective_generated_at,
            articles=articles_for_llm,
            model_name=model_name,
            error=warning,
        )


def main() -> None:
    set_pipeline_scope("sector_scores")
    universe_path = resolve_universe_csv_path()
    symbol_metadata = load_symbol_metadata(universe_path)
    sectors = load_active_sectors(universe_path)
    articles = fetch_recent_global_articles()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0)
    payload = build_sector_scores_payload(
        articles=articles,
        symbol_metadata=symbol_metadata,
        sectors=sectors,
        generated_at=generated_at,
    )
    payload = attach_data_quality(payload, scope="sector_scores")

    with SECTOR_SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\nsector_scores.json updated.")


if __name__ == "__main__":
    main()
