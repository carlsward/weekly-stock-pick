import argparse
import hashlib
import json
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher

import pandas as pd
import torch
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline

NEWS_SCORES_PATH = "news_scores.json"
UNIVERSE_CSV_PATH = "universe.csv"
EXTRA_NEWS_PATH = Path("extra_news.json")
ALT_NEWS_PATH = Path("alt_news.json")
CONFIG_PATH = Path("config.json")

FINBERT_MODEL_NAME = "ProsusAI/finbert"
GENERAL_SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Hur långt tillbaka vi tittar på nyheter
NEWS_LOOKBACK_DAYS = 5
MAX_ARTICLES_PER_SYMBOL = 10
MAX_PER_PROVIDER = 3
SENTIMENT_MAX_LENGTH = 512
SENTIMENT_DECAY_HALF_LIFE_DAYS = 2.0  # dagar till halverad vikt
SENTIMENT_CACHE_PATH = Path(".news_sentiment_cache.json")
SENTIMENT_MODEL_VERSION = FINBERT_MODEL_NAME
GENERAL_SENTIMENT_VERSION = GENERAL_SENTIMENT_MODEL_NAME
SENTIMENT_CONFIDENCE_THRESHOLD = 0.55
SENTIMENT_ENSEMBLE_WEIGHTS = (0.65, 0.35)  # finbert, general
SEMANTIC_SIM_THRESHOLD = 0.35

PROVIDER_TRUST: Dict[str, float] = {
    "bloomberg": 1.0,
    "reuters": 1.0,
    "the wall street journal": 0.95,
    "financial times": 0.95,
    "seeking alpha": 0.8,
    "motley fool": 0.8,
    "benzinga": 0.8,
}


# ---------- Cache-hantering ----------

def load_sentiment_cache() -> Dict[str, dict]:
    if not SENTIMENT_CACHE_PATH.exists():
        return {}
    try:
        with open(SENTIMENT_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def save_sentiment_cache(cache: Dict[str, dict]) -> None:
    tmp = SENTIMENT_CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(SENTIMENT_CACHE_PATH)


def load_config() -> dict:
    defaults = {
        "provider_trust": PROVIDER_TRUST,
        "sentiment_confidence_threshold": SENTIMENT_CONFIDENCE_THRESHOLD,
        "sentiment_ensemble_weights": SENTIMENT_ENSEMBLE_WEIGHTS,
        "semantic_similarity_threshold": SEMANTIC_SIM_THRESHOLD,
        "news_feed_urls": [],
    }
    if not CONFIG_PATH.exists():
        return defaults
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if not isinstance(cfg, dict):
                return defaults
            for k in defaults:
                defaults[k] = cfg.get(k, defaults[k])
    except Exception:
        return defaults
    return defaults


def load_extra_news(symbol: str) -> List[dict]:
    """
    Ladda extra nyheter från lokal JSON (för att bredda täckningen utan nätberoende).
    Förväntat format:
    {
      "AAPL": [
        {"title": "...", "summary": "...", "description": "...", "provider": "source", "pubDate": "2025-01-01T12:00:00Z"}
      ]
    }
    """
    sources = [EXTRA_NEWS_PATH, ALT_NEWS_PATH]
    merged: List[dict] = []
    for path in sources:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get(symbol.upper(), [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                merged.append(
                    {
                        "content": {
                            "title": item.get("title"),
                            "summary": item.get("summary"),
                            "description": item.get("description"),
                            "pubDate": item.get("pubDate"),
                            "provider": item.get("provider"),
                            "link": item.get("link"),
                        },
                        "publisher": item.get("provider"),
                        "link": item.get("link"),
                    }
                )
        except Exception:
            continue
    return merged


def fetch_remote_feeds(urls: List[str]) -> List[dict]:
    items: List[dict] = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = resp.read()
            parsed = json.loads(data.decode("utf-8"))
            if isinstance(parsed, list):
                items.extend(parsed)
            elif isinstance(parsed, dict):
                # accept dict of symbol -> list
                for sym_items in parsed.values():
                    if isinstance(sym_items, list):
                        items.extend(sym_items)
        except Exception as e:
            print(f"[WARN] Failed to fetch remote feed {url}: {e}")
            continue
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "content": {
                    "title": item.get("title"),
                    "summary": item.get("summary") or item.get("description"),
                    "description": item.get("description"),
                    "pubDate": item.get("pubDate") or item.get("publishedAt"),
                    "provider": item.get("provider") or item.get("source"),
                    "link": item.get("link") or item.get("url"),
                },
                "publisher": item.get("provider") or item.get("source"),
                "link": item.get("link") or item.get("url"),
            }
        )
    return normalized


# ---------- Hjälpfunktioner för universet ----------

def load_universe(path: str = UNIVERSE_CSV_PATH) -> List[Tuple[str, str]]:
    """
    Läser in aktieuniverset från CSV.
    Förväntat format: symbol,company_name,active
    """
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
        raise RuntimeError("Inga aktier i universe.csv med active=1")
    return rows


# ---------- Device-hantering (enkel) ----------

def detect_device_for_logs() -> None:
    """
    Bara logging – vi låter pipelines köra på CPU (device=-1) för robusthet.
    """
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("Device set to use mps:0")
        elif torch.cuda.is_available():
            print("Device set to use cuda:0")
        else:
            print("Device set to use cpu")
    except Exception:
        print("Device detection failed, using CPU")


# ---------- Nyhets-hantering ----------

def parse_news_time(raw: dict) -> Optional[datetime]:
    """
    Försöker plocka ut tid från yfinance-nyhetsobjekt.
    Returnerar UTC-aware datetime eller None.
    """
    content = raw.get("content") or {}
    ts = content.get("pubDate") or content.get("displayTime")
    # Vissa poster har epoch-sekunder i providerPublishTime
    if ts is None:
        epoch = raw.get("providerPublishTime") or content.get("providerPublishTime")
        if isinstance(epoch, (int, float)):
            try:
                return datetime.fromtimestamp(epoch, tz=timezone.utc)
            except Exception:
                return None

    if not ts:
        return None
    try:
        # Ex: "2025-11-18T11:00:42Z"
        if isinstance(ts, str) and ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts) if isinstance(ts, str) else None
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def build_symbol_keywords(symbol: str, company_name: str) -> List[str]:
    """
    Bygger en lista med nyckelord som vi använder för att avgöra
    om en nyhetsartikel verkligen handlar om bolaget.
    """
    sym = symbol.lower()

    # Plocka ut "viktiga" ord ur company_name
    stop_words = {"inc", "corp", "corporation", "plc", "sa", "co", "ltd", "company", "&"}
    name_parts = [
        w.lower()
        for w in company_name.replace(",", " ").replace(".", " ").split()
        if w.lower() not in stop_words and len(w) > 2
    ]

    # Special-synonymer för vissa tickers
    manual: Dict[str, List[str]] = {
        "GOOGL": ["google", "alphabet"],
        "GOOG": ["google", "alphabet"],
        "META": ["meta", "facebook"],
        "BRK.B": ["berkshire"],
        "BRK.A": ["berkshire"],
        "NVDA": ["nvidia"],
        "TSLA": ["tesla"],
        "MSFT": ["microsoft"],
        "AAPL": ["apple"],
        "AMZN": ["amazon"],
    }

    extra = manual.get(symbol.upper(), [])
    keywords = set([sym] + name_parts + extra)
    return [k for k in keywords if k]


def fuzzy_company_match(text: str, company_name: str, threshold: float = 0.42) -> bool:
    """
    Enkel fuzzy-match mellan text och företagsnamn för att minska falska positiva.
    """
    text = text.lower()
    target = company_name.lower()
    ratio = SequenceMatcher(None, text, target).ratio()
    return ratio >= threshold


def is_relevant_news_item(
    item: dict,
    symbol: str,
    company_name: str,
) -> bool:
    """
    Filtrerar bort generella marknadsartiklar.
    Kräver att någon nyckelfras kopplad till bolaget finns i titel/summary/description.
    """
    content = item.get("content") or {}
    title = (content.get("title") or "").lower()
    summary = (content.get("summary") or "").lower()
    desc = (content.get("description") or "").lower()
    text = " ".join([title, summary, desc])

    if not text.strip():
        return False

    keywords = build_symbol_keywords(symbol, company_name)
    if not keywords:
        return False

    words = set(re.findall(r"[a-z0-9]+", text))
    hits = [k for k in keywords if k in words or f"{k}s" in words]
    if hits:
        return True

    # Fuzzy fallback på företagsnamn för att fånga stavningar/korta titlar
    return fuzzy_company_match(title + " " + summary, company_name)


def fetch_symbol_news(symbol: str, company_name: str, extra_feed_items: Optional[List[dict]] = None) -> List[Dict[str, str]]:
    """
    Hämtar nyhetstexter (titel + summary) för ett bolag, filtrerar på datum och relevans.
    Returnerar en lista med metadata + text per artikel.
    """
    ticker = yf.Ticker(symbol)
    raw_news = []
    for attempt in range(3):
        try:
            raw_news = ticker.news or []
            break
        except Exception as e:
            if attempt == 2:
                raise
            sleep_time = 1 + attempt
            print(f"[{symbol}] news fetch failed (attempt {attempt+1}): {e} – retrying in {sleep_time}s")
            time.sleep(sleep_time)

    # Lägg till lokala extra nyheter om de finns
    raw_news += load_extra_news(symbol)
    if extra_feed_items:
        raw_news += extra_feed_items

    print(f"\n=== Hämtar nyheter för {symbol} ===")
    print(f"[{symbol}] raw news count: {len(raw_news)}")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=NEWS_LOOKBACK_DAYS)

    # Sortera på tid (nyaste först) så vi inte råkar plocka gamla om ordningen ändras
    sortable_news = []
    for item in raw_news:
        t = parse_news_time(item)
        sortable_news.append((t, item))
    sortable_news.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    articles: List[Dict[str, str]] = []
    seen_titles: set[str] = set()
    seen_links: set[str] = set()
    per_provider: Dict[str, int] = {}

    for idx, (t, item) in enumerate(sortable_news):
        content = item.get("content") or {}
        title = content.get("title") or ""
        summary = content.get("summary") or ""
        desc = content.get("description") or ""
        provider = (item.get("publisher") or content.get("provider") or "").strip().lower()
        link = (item.get("link") or content.get("link") or "").strip().lower()

        print(f"[{symbol}] item {idx}: time={t} title='{title}'")

        # Tidfilter – hoppa över odaterade artiklar för att undvika uråldrigt brus
        if t is None or t < cutoff:
            continue

        # Relevansfilter
        if not is_relevant_news_item(item, symbol, company_name):
            continue

        # Dubbeldetektion på titel/länk
        norm_title = title.strip().lower()
        if norm_title and norm_title in seen_titles:
            continue
        if link and link in seen_links:
            continue

        # Begränsa övervikt från en och samma källa
        if provider:
            per_provider[provider] = per_provider.get(provider, 0) + 1
            if per_provider[provider] > MAX_PER_PROVIDER:
                continue

        combined = " ".join([title, summary, desc]).strip()
        if combined:
            articles.append(
                {
                    "title": title,
                    "provider": provider or "",
                    "published_at": t.isoformat(),
                    "text": combined,
                    "link": link,
                }
            )
            if norm_title:
                seen_titles.add(norm_title)
            if link:
                seen_links.add(link)

        if len(articles) >= MAX_ARTICLES_PER_SYMBOL:
            break

    print(f"[{symbol}] texts used: {len(articles)}")
    return articles


# ---------- Modellinit ----------

def init_models():
    print("Laddar FinBERT-modell...")
    detect_device_for_logs()

    finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    finbert_pipe = pipeline(
        "sentiment-analysis",
        model=finbert_model,
        tokenizer=finbert_tokenizer,
        device=-1,  # CPU för robusthet
        return_all_scores=True,
    )

    print("Laddar generell sentiment-modell...")
    general_tokenizer = AutoTokenizer.from_pretrained(GENERAL_SENTIMENT_MODEL_NAME)
    general_model = AutoModelForSequenceClassification.from_pretrained(GENERAL_SENTIMENT_MODEL_NAME)
    general_pipe = pipeline(
        "sentiment-analysis",
        model=general_model,
        tokenizer=general_tokenizer,
        device=-1,
        return_all_scores=True,
    )

    print("Laddar summarizer-modell...")
    detect_device_for_logs()
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)
    summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
    summarizer_pipe = pipeline(
        "summarization",
        model=summarizer_model,
        tokenizer=summarizer_tokenizer,
        device=-1,
    )

    print("Laddar embeddings-modell...")
    embed_pipe = pipeline(
        "feature-extraction",
        model=EMBEDDING_MODEL_NAME,
        tokenizer=EMBEDDING_MODEL_NAME,
        device=-1,
    )

    return finbert_pipe, general_pipe, summarizer_pipe, embed_pipe


# ---------- Sentiment + sammanfattning ----------

def sentiment_cache_key(symbol: str, article: Dict[str, str]) -> str:
    title = (article.get("title") or "").strip().lower()
    provider = (article.get("provider") or "").strip().lower()
    published = article.get("published_at") or ""
    blob = f"{symbol}|{provider}|{published}|{title}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def compute_ensemble_sentiment(
    finbert_pipe,
    general_pipe,
    articles: List[Dict[str, str]],
    symbol: str,
    cache: Dict[str, dict],
) -> float:
    """
    Kör FinBERT + generell sentiment och beräknar ett viktat snitt i intervallet [-1, 1].
    Viktning baseras på ålder (halveringstid SENTIMENT_DECAY_HALF_LIFE_DAYS) och källtillit.
    Lågkonfidensartiklar filtreras bort.
    """
    if not articles:
        return 0.0

    to_score: List[str] = []
    to_score_idx: List[int] = []
    for idx, article in enumerate(articles):
        key = sentiment_cache_key(symbol, article)
        article["cache_key"] = key
        cached = cache.get(key)
        if cached and cached.get("finbert_version") == SENTIMENT_MODEL_VERSION and cached.get("general_version") == GENERAL_SENTIMENT_VERSION:
            article["raw_sentiment"] = float(cached.get("sentiment", 0.0))
            article["finbert_sentiment"] = float(cached.get("finbert_sentiment", 0.0))
            article["general_sentiment"] = float(cached.get("general_sentiment", 0.0))
            article["sentiment_weight"] = float(cached.get("weight", 1.0))
            article["confidence"] = float(cached.get("confidence", 1.0))
        else:
            to_score.append(article["text"])
            to_score_idx.append(idx)

    if to_score:
        finbert_results = finbert_pipe(
            to_score,
            truncation=True,
            max_length=SENTIMENT_MAX_LENGTH,
            return_all_scores=True,
        )
        general_results = general_pipe(
            to_score,
            truncation=True,
            max_length=SENTIMENT_MAX_LENGTH,
            return_all_scores=True,
        )
        for idx, f_scores, g_scores in zip(to_score_idx, finbert_results, general_results):
            f_pos = next((float(s["score"]) for s in f_scores if "pos" in s["label"].lower()), 0.0)
            f_neg = next((float(s["score"]) for s in f_scores if "neg" in s["label"].lower()), 0.0)
            f_conf = max(float(s["score"]) for s in f_scores) if f_scores else 0.0
            finbert_sent = f_pos - f_neg

            # general sentiment labels usually "POSITIVE"/"NEGATIVE"
            g_label_score = max(g_scores, key=lambda s: s["score"]) if g_scores else {"label": "neutral", "score": 0.0}
            g_conf = float(g_label_score.get("score", 0.0))
            g_sent = g_conf if "pos" in g_label_score.get("label", "").lower() else -g_conf

            conf = max(f_conf, g_conf)
            combined = SENTIMENT_ENSEMBLE_WEIGHTS[0] * finbert_sent + SENTIMENT_ENSEMBLE_WEIGHTS[1] * g_sent

            articles[idx]["raw_sentiment"] = combined
            articles[idx]["finbert_sentiment"] = finbert_sent
            articles[idx]["general_sentiment"] = g_sent
            articles[idx]["confidence"] = conf

    now = datetime.now(timezone.utc)
    weighted: List[float] = []
    weights: List[float] = []

    for article in articles:
        raw_sent = float(article.get("raw_sentiment", 0.0))
        conf = float(article.get("confidence", 0.0))
        if conf < SENTIMENT_CONFIDENCE_THRESHOLD:
            continue

        published = article.get("published_at")
        age_days = 0.0
        try:
            if isinstance(published, str):
                dt = datetime.fromisoformat(published)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now - dt).total_seconds() / 86400)
        except Exception:
            age_days = 0.0

        decay = 0.5 ** (age_days / SENTIMENT_DECAY_HALF_LIFE_DAYS)
        provider = (article.get("provider") or "").strip().lower()
        provider_weight = PROVIDER_TRUST.get(provider, 0.85 if provider else 0.8)
        weight = decay * provider_weight

        weighted.append(raw_sent * weight)
        weights.append(weight)

        article["sentiment_weight"] = weight
        article["provider_weight"] = provider_weight

        key = article.get("cache_key")
        if key:
            cache[key] = {
                "sentiment": raw_sent,
                "finbert_sentiment": float(article.get("finbert_sentiment", 0.0)),
                "general_sentiment": float(article.get("general_sentiment", 0.0)),
                "weight": weight,
                "confidence": conf,
                "finbert_version": SENTIMENT_MODEL_VERSION,
                "general_version": GENERAL_SENTIMENT_VERSION,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }

    if not weights or sum(weights) == 0:
        return 0.0
    return sum(weighted) / sum(weights)


def summarize_texts(pipe, articles: List[Dict[str, str]]) -> str:
    """
    Summerar alla texter till en kort sammanfattning (1–2 meningar).
    """
    if not articles:
        return ""

    joined = " ".join(a["text"] for a in articles)

    try:
        out = pipe(
            joined,
            max_length=80,   # kortare sammanfattning
            min_length=30,
            do_sample=False,
            truncation=True,
        )
        if out and isinstance(out, list):
            return out[0].get("summary_text", "").strip()
    except Exception as e:
        print(f"[WARN] Summarization failed: {e}")

    # Fallback: ta bara de första titlarna
    return " ".join(t[:120] for t in texts[:2])


def get_embedding(embed_pipe, text: str) -> torch.Tensor:
    """
    Beräknar en genomsnittlig embeddingsvektor för en text.
    """
    outputs = embed_pipe(text, truncation=True, max_length=128)
    # pipeline returnerar list[list[list[float]]] (batch x tokens x dim)
    if not outputs:
        return torch.zeros(1)
    arr = torch.tensor(outputs[0])  # tokens x dim
    if arr.ndim != 2:
        return torch.zeros(1)
    return arr.mean(dim=0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, dim=1)
    b_norm = torch.nn.functional.normalize(b, dim=1)
    return float((a_norm @ b_norm.t()).mean())




def sentiment_to_news_score(raw_sent: float) -> float:
    """
    Mappar råsentiment [-1, 1] till [0, 1].
    0 = starkt negativt, 0.5 = neutralt, 1 = starkt positivt.
    """
    x = max(-1.0, min(1.0, raw_sent))
    return 0.5 + 0.5 * x


def filter_semantic_relevance(
    articles: List[Dict[str, str]],
    company_name: str,
    embed_pipe,
    company_embedding: torch.Tensor,
) -> List[Dict[str, str]]:
    """
    Filtrerar artiklar baserat på semantisk likhet mellan rubrik/summary och företagsnamnet.
    """
    if company_embedding.numel() == 0:
        return articles

    kept: List[Dict[str, str]] = []
    for a in articles:
        text = (a.get("title", "") + " " + a.get("text", ""))[:512]
        emb = get_embedding(embed_pipe, text)
        sim = cosine_sim(company_embedding, emb)
        a["semantic_sim"] = sim
        if sim >= SEMANTIC_SIM_THRESHOLD:
            kept.append(a)
    return kept


# ---------- Huvudlogik ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Kör utan att skriva news_scores.json")
    args = parser.parse_args()

    cfg = load_config()
    global PROVIDER_TRUST, SENTIMENT_CONFIDENCE_THRESHOLD, SENTIMENT_ENSEMBLE_WEIGHTS, SEMANTIC_SIM_THRESHOLD
    PROVIDER_TRUST = cfg.get("provider_trust", PROVIDER_TRUST)
    SENTIMENT_CONFIDENCE_THRESHOLD = cfg.get("sentiment_confidence_threshold", SENTIMENT_CONFIDENCE_THRESHOLD)
    SENTIMENT_ENSEMBLE_WEIGHTS = tuple(cfg.get("sentiment_ensemble_weights", SENTIMENT_ENSEMBLE_WEIGHTS))  # type: ignore
    SEMANTIC_SIM_THRESHOLD = cfg.get("semantic_similarity_threshold", SEMANTIC_SIM_THRESHOLD)
    remote_feeds = fetch_remote_feeds(cfg.get("news_feed_urls", []))

    finbert_pipe, general_pipe, summarizer_pipe, embed_pipe = init_models()
    universe = load_universe()
    cache = load_sentiment_cache()

    out: Dict[str, dict] = {}
    today_str = datetime.now(timezone.utc).date().isoformat()
    out_path = Path(NEWS_SCORES_PATH)

    for symbol, company_name in universe:
        try:
            try:
                company_embedding = get_embedding(embed_pipe, f"{symbol} {company_name}")
            except Exception:
                company_embedding = torch.zeros(1)
            articles = fetch_symbol_news(symbol, company_name, extra_feed_items=remote_feeds)
            articles = filter_semantic_relevance(articles, company_name, embed_pipe, company_embedding)

            if not articles:
                print(f"[{symbol}] No relevant recent news – setting neutral 0.50")
                out[symbol] = {
                    "news_score": 0.5,
                    "news_reasons": [
                        "No recently identified relevant news articles for this company – "
                        "the news contribution is set to neutral (0.50)."
                    ],
                    "raw_sentiment": 0.0,
                    "article_count": 0,
                    "last_updated": today_str,
                    "articles": [],
                    "decay_half_life_days": SENTIMENT_DECAY_HALF_LIFE_DAYS,
                }
            else:
                raw_sent = compute_ensemble_sentiment(finbert_pipe, general_pipe, articles, symbol, cache)
                news_score = sentiment_to_news_score(raw_sent)
                summary = summarize_texts(summarizer_pipe, articles)

                reasons = [
                    f"News sentiment (AI model, FinBERT) based on {len(articles)} articles from the last few days (time-decayed).",
                    f"News summary for {symbol} ({company_name}): {summary}",
                ]

                articles_info = [
                    {
                        "title": a.get("title", "")[:200],
                        "provider": a.get("provider", ""),
                        "published_at": a.get("published_at"),
                        "raw_sentiment": round(float(a.get("raw_sentiment", 0.0)), 4),
                        "weight": round(float(a.get("sentiment_weight", 1.0)), 3),
                        "provider_weight": round(float(a.get("provider_weight", 1.0)), 3),
                        "link": a.get("link", ""),
                        "semantic_sim": round(float(a.get("semantic_sim", 0.0)), 3),
                        "finbert_sentiment": round(float(a.get("finbert_sentiment", 0.0)), 4),
                        "general_sentiment": round(float(a.get("general_sentiment", 0.0)), 4),
                        "confidence": round(float(a.get("confidence", 0.0)), 3),
                    }
                    for a in articles
                ]

                out[symbol] = {
                    "news_score": round(news_score, 2),
                    "news_reasons": reasons,
                    "raw_sentiment": round(raw_sent, 4),
                    "article_count": len(articles),
                    "last_updated": today_str,
                    "articles": articles_info,
                    "decay_half_life_days": SENTIMENT_DECAY_HALF_LIFE_DAYS,
                }

        except Exception as e:
            print(f"[WARN] Failed to process {symbol}: {e}")
            out[symbol] = {
                "news_score": 0.5,
                "news_reasons": [
                    f"News processing failed ({e}); using neutral news contribution (0.50)."
                ],
                "raw_sentiment": 0.0,
                "article_count": 0,
                "last_updated": today_str,
                "articles": [],
                "decay_half_life_days": SENTIMENT_DECAY_HALF_LIFE_DAYS,
            }

        if not args.dry_run:
            # Skriv inkrementellt för att inte tappa framsteg om något avbryts
            tmp_path = out_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            tmp_path.replace(out_path)

    if not args.dry_run:
        save_sentiment_cache(cache)
        print("\nnews_scores.json uppdaterad.")
    else:
        print("\nDry run – ingen fil skrivs.")


if __name__ == "__main__":
    main()
