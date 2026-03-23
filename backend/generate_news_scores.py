import json
import math
import os
import re
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

NEWS_SCORES_PATH = "news_scores.json"
UNIVERSE_CSV_PATH = "universe.csv"
MARKETAUX_NEWS_ENDPOINT = "https://api.marketaux.com/v1/news/all"

MARKETAUX_API_TOKEN_ENV = "MARKETAUX_API_TOKEN"
MARKETAUX_NEWS_LIMIT_ENV = "MARKETAUX_NEWS_LIMIT"
MARKETAUX_DEFAULT_NEWS_LIMIT = 3
MARKETAUX_REQUEST_TIMEOUT_SECONDS = 20

NEWS_LOOKBACK_DAYS = 5
MAX_ARTICLES_PER_SYMBOL = 10
MIN_RELEVANCE_SCORE = 0.45
RECENCY_HALFLIFE_HOURS = 36.0
NEWS_FETCH_ATTEMPTS = 3
NEWS_FETCH_RETRY_SECONDS = 1.5

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


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def load_universe(path: str = UNIVERSE_CSV_PATH) -> List[Tuple[str, str]]:
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
    return max(1, min(parsed_limit, MAX_ARTICLES_PER_SYMBOL))


def marketaux_api_token() -> str:
    token = os.getenv(MARKETAUX_API_TOKEN_ENV, "").strip()
    if token:
        return token
    raise RuntimeError(
        f"{MARKETAUX_API_TOKEN_ENV} is required for news generation. "
        "Set a free Marketaux API token for weekly runs."
    )


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


def fetch_marketaux_payload(symbol: str, published_after: datetime) -> List[dict]:
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
            with urlopen(request, timeout=MARKETAUX_REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise RuntimeError("Unexpected Marketaux response shape")
            return payload["data"]
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (401, 402, 403):
                raise RuntimeError(f"Marketaux authentication failed for {symbol}: HTTP {exc.code} {body[:160]}") from exc
            if exc.code == 429:
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
    if published_at is not None and published_at < now - timedelta(days=NEWS_LOOKBACK_DAYS):
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
    )


def fetch_raw_news(symbol: str) -> List[dict]:
    published_after = datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)
    return fetch_marketaux_payload(symbol, published_after)


def fetch_symbol_news(symbol: str, company_name: str) -> List[NewsArticle]:
    raw_news = fetch_raw_news(symbol)

    print(f"\n=== Fetching news for {symbol} (Marketaux) ===")
    print(f"[{symbol}] raw news count: {len(raw_news)}")

    now = datetime.now(timezone.utc)
    articles: List[NewsArticle] = []
    seen_titles = set()

    for idx, item in enumerate(raw_news):
        article = build_marketaux_article(item, symbol, company_name, now)
        print(
            f"[{symbol}] item {idx}: time={item.get('published_at')} title='{str(item.get('title', ''))[:120]}'"
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
            article.published_at or datetime.fromtimestamp(0, tz=timezone.utc),
            article.weight,
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


def sentiment_to_news_score(raw_sent: float) -> float:
    x = clamp(raw_sent, -1.0, 1.0)
    return 0.5 + 0.5 * x


def build_news_reasons(
    symbol: str,
    company_name: str,
    summary: str,
    signal: Dict[str, float],
    articles: List[NewsArticle],
) -> List[str]:
    reasons = [
        (
            f"Marketaux entity coverage rated the news flow {signal['dominant_signal']} at {signal['news_score']:.2f}, "
            f"after adjusting for relevance, recency, source quality, and story concentration."
        ),
        (
            f"Evidence quality: {len(articles)} articles across {signal['source_count']} sources, "
            f"{signal['effective_article_count']:.1f} effective weighted articles, "
            f"average relevance {signal['average_relevance']:.2f}, average source quality "
            f"{signal['average_source_quality']:.2f}, confidence {signal['news_confidence']:.2f}."
        ),
    ]

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

    if summary:
        reasons.append(f"News summary for {symbol} ({company_name}): {summary}")

    return reasons


def serialize_article_evidence(article: NewsArticle) -> Dict[str, object]:
    return {
        "title": article.title,
        "provider": article.provider,
        "url": article.url,
        "published_at": article.published_at.isoformat().replace("+00:00", "Z") if article.published_at else None,
        "relevance_score": round(article.relevance_score, 2),
        "sentiment": round(article_sentiment_value(article), 2),
    }


def main():
    _, summarizer_pipe = init_models()
    universe = load_universe()

    out: Dict[str, dict] = {}
    today_str = datetime.now(timezone.utc).date().isoformat()

    for symbol, company_name in universe:
        articles = fetch_symbol_news(symbol, company_name)

        if not articles:
            print(f"[{symbol}] No relevant recent news from Marketaux - setting neutral 0.50")
            out[symbol] = {
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
                    "No recently identified relevant Marketaux articles passed the relevance and freshness filters, so the news contribution is neutral."
                ],
                "top_headlines": [],
                "top_articles": [],
                "last_updated": today_str,
            }
            continue

        signal = compute_finbert_sentiment(None, articles)
        summary = summarize_texts(summarizer_pipe, articles)
        reasons = build_news_reasons(symbol, company_name, summary, signal, articles)

        out[symbol] = {
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
            "top_articles": [serialize_article_evidence(article) for article in articles[:5]],
            "last_updated": today_str,
        }

    with open(NEWS_SCORES_PATH, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)

    print("\nnews_scores.json updated.")


if __name__ == "__main__":
    main()
