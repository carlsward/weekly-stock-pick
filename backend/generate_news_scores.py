import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import yfinance as yf
from transformers import pipeline


# Samma watchlist som i generate_pick.py
WATCHLIST = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corporation"),
    ("GOOGL", "Alphabet Inc."),
]

# Hur långt tillbaka vi accepterar nyheter
MAX_NEWS_AGE_DAYS = 14
MAX_ARTICLES_PER_SYMBOL = 10

# Modellnamn – öppna, gratis HuggingFace-modeller
FINBERT_MODEL_NAME = "ProsusAI/finbert"
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"


@dataclass
class NewsResult:
    symbol: str
    news_score: float          # 0–1, 0.5 = neutralt
    news_reasons: List[str]
    raw_sentiment: float       # -1..1
    article_count: int
    last_updated: str


def unix_to_datetime(ts: Any) -> datetime | None:
    """Försök tolka providerPublishTime till datetime i UTC."""
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(ts, str):
        # Bästa gissning – många APIs ger ISO 8601
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def fetch_news_texts(symbol: str) -> List[str]:
    """
    Hämta nyhetstexter för ett bolag via yfinance.
    Returnerar en lista av textstycken (headline + summary).
    """
    print(f"\n=== Hämtar nyheter för {symbol} ===")
    ticker = yf.Ticker(symbol)

    raw_news = getattr(ticker, "news", []) or []
    print(f"[{symbol}] raw news count: {len(raw_news)}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_NEWS_AGE_DAYS)

    texts: List[str] = []
    for item in raw_news:
        title = item.get("title") or ""
        summary = item.get("summary") or ""
        provider_time = unix_to_datetime(item.get("providerPublishTime"))

        if provider_time is not None:
            if provider_time < cutoff:
                continue
        # Om vi inte kan tolka tid – ta hellre med än kasta
        if not (title or summary):
            continue

        combined = f"{title}. {summary}".strip()
        texts.append(combined)

        if len(texts) >= MAX_ARTICLES_PER_SYMBOL:
            break

    print(f"[{symbol}] recent texts used: {len(texts)}")
    return texts


def build_pipelines():
    """
    Ladda FinBERT och summarizer som pipelines.
    Körs en gång, återanvänds för alla bolag.
    """
    print("Laddar FinBERT-modell...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL_NAME,
        tokenizer=FINBERT_MODEL_NAME,
        truncation=True,
    )

    print("Laddar summarizer-modell...")
    summarizer_pipe = pipeline(
        "summarization",
        model=SUMMARIZER_MODEL_NAME,
        tokenizer=SUMMARIZER_MODEL_NAME,
    )

    return sentiment_pipe, summarizer_pipe


def analyse_symbol_news(
    symbol: str,
    company_name: str,
    sentiment_pipe,
    summarizer_pipe,
) -> NewsResult:
    """
    Kör hela nyhets-analysen för ett bolag.
    Om inga texter finns -> neutral score 0.50 + placeholder-reason.
    """
    texts = fetch_news_texts(symbol)

    if not texts:
        print(f"[{symbol}] Inga nyhetstexter – sätter neutral 0.50")
        neutral_reason = (
            "Inga nyligen identifierade nyhetsartiklar för bolaget "
            "– nyhetsbidraget sätts till neutralt (0.50)."
        )
        return NewsResult(
            symbol=symbol,
            news_score=0.5,
            news_reasons=[neutral_reason],
            raw_sentiment=0.0,
            article_count=0,
            last_updated=datetime.now().date().isoformat(),
        )

    # Slå ihop texterna till ett längre dokument
    combined = " ".join(texts)
    # Begränsa längd för summarizer/sentiment (annars kan det bli väldigt långt)
    combined_for_models = combined[:4000]

    # 1) Summarization
    print(f"[{symbol}] Kör summarizer på {len(combined_for_models)} tecken...")
    summary_output = summarizer_pipe(
        combined_for_models,
        max_length=180,
        min_length=60,
        do_sample=False,
    )
    summary_text = summary_output[0]["summary_text"].strip()

    # 2) Sentiment med FinBERT
    print(f"[{symbol}] Kör FinBERT-sentiment...")
    sentiment_output = sentiment_pipe(combined_for_models[:512])[0]
    label = sentiment_output["label"].lower()
    score = float(sentiment_output["score"])

    # Mappa till -1..1
    if "positive" in label:
        raw_sentiment = +score
    elif "negative" in label:
        raw_sentiment = -score
    else:
        raw_sentiment = 0.0

    # Normalisera till 0..1
    news_score = 0.5 + 0.5 * raw_sentiment
    news_score = max(0.0, min(1.0, news_score))

    reasons: List[str] = [
        f"Nyhetsanalys (AI-modell) för {company_name}:",
        f"Sammanfattning: {summary_text}",
        f"Sentiment enligt FinBERT: {label} (score={score:.2f}).",
    ]

    return NewsResult(
        symbol=symbol,
        news_score=news_score,
        news_reasons=reasons,
        raw_sentiment=raw_sentiment,
        article_count=len(texts),
        last_updated=datetime.now().date().isoformat(),
    )


def main():
    sentiment_pipe, summarizer_pipe = build_pipelines()

    results: Dict[str, Dict[str, Any]] = {}

    for symbol, company_name in WATCHLIST:
        try:
            res = analyse_symbol_news(
                symbol=symbol,
                company_name=company_name,
                sentiment_pipe=sentiment_pipe,
                summarizer_pipe=summarizer_pipe,
            )
            results[symbol] = {
                "news_score": res.news_score,
                "news_reasons": res.news_reasons,
                "raw_sentiment": res.raw_sentiment,
                "article_count": res.article_count,
                "last_updated": res.last_updated,
            }
        except Exception as e:
            print(f"[{symbol}] Fel i nyhetsanalys: {e}")
            neutral_reason = (
                "Tekniskt fel vid nyhetsanalys – nyhetsbidraget sätts till neutralt (0.50)."
            )
            results[symbol] = {
                "news_score": 0.5,
                "news_reasons": [neutral_reason],
                "raw_sentiment": 0.0,
                "article_count": 0,
                "last_updated": datetime.now().date().isoformat(),
            }

    with open("news_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nnews_scores.json uppdaterad.")


if __name__ == "__main__":
    main()
