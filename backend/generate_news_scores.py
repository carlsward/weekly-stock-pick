import json
from datetime import date, timedelta
from typing import List, Dict

import yfinance as yf
from transformers import pipeline

# Samma watchlist som i generate_pick.py
WATCHLIST = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corporation"),
    ("GOOGL", "Alphabet Inc."),
]

# Hur många dagar bakåt vi tittar på nyheter
NEWS_DAYS = 5
# Max antal artiklar per bolag som vi analyserar
MAX_ARTICLES_PER_SYMBOL = 15

# -----------------------------
# Initiera modeller (öppna & gratis)
# -----------------------------

# Finansiell sentiment-modell (FinBERT)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# Kompakt sammanfattningsmodell (distilbart)
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6"
)

# Engelska -> svenska översättning
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-sv"
)


def fetch_news_for_symbol(symbol: str) -> List[Dict]:
    """
    Hämtar nyheter via yfinance.Ticker.news.
    Returnerar en lista av dicts med title/summary/publishTime.
    """
    ticker = yf.Ticker(symbol)
    news_items = ticker.news or []

    # Sortera efter publiceringstid, senaste först
    news_items = sorted(
        news_items,
        key=lambda x: x.get("providerPublishTime", 0),
        reverse=True,
    )

    # Begränsa antal artiklar
    return news_items[:MAX_ARTICLES_PER_SYMBOL]


def is_recent(item: Dict, days: int = NEWS_DAYS) -> bool:
    """
    Filtrera bort gammal info baserat på providerPublishTime (unix-sekunder).
    """
    ts = item.get("providerPublishTime")
    if not ts:
        return True  # om okänt – ta med
    published = date.fromtimestamp(ts)
    return (date.today() - published) <= timedelta(days=days)


def analyse_sentiment(texts: List[str]) -> float:
    """
    Kör FinBERT på en lista av texter.
    Returnerar ett sentimentvärde i intervallet [-1, 1].
    """
    if not texts:
        return 0.0

    results = sentiment_analyzer(texts)
    scores = []

    for res in results:
        label = res["label"].lower()
        score = float(res["score"])
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:  # neutral
            scores.append(0.0)

    if not scores:
        return 0.0

    avg = sum(scores) / len(scores)
    # Klipp till [-1, 1] för säkerhets skull
    return max(-1.0, min(1.0, avg))


def summarise_news_sv(symbol: str, company_name: str, texts: List[str]) -> str:
    """
    Sammanfatta nyhetsflödet till en kort svensk text.
    Steg:
    1) slå ihop rubriker och snippets
    2) kör engelsk sammanfattning
    3) översätt till svenska
    """
    if not texts:
        return "Inga tydliga nyhetsdrivare identifierades den här perioden."

    # Bygg en engelsk text att sammanfatta
    joined = " ".join(texts)
    # Sammanfatta (engelska)
    summary_en = summarizer(
        joined,
        max_length=120,
        min_length=40,
        do_sample=False,
    )[0]["summary_text"]

    # Översätt till svenska
    summary_sv = translator(summary_en)[0]["translation_text"]

    return f"Nyhetsanalys (AI-modell) för {symbol} / {company_name}: {summary_sv}"


def analyse_symbol_news(symbol: str, company_name: str) -> Dict:
    """
    Komplett AI-analys per bolag:
    - hämtar nyheter
    - filtrerar på senaste dagarna
    - sentimentanalys
    - sammanfattning (svenska)
    - bygger strukturen som används i news_scores.json
    """
    items = fetch_news_for_symbol(symbol)
    recent_items = [n for n in items if is_recent(n)]

    # Bygg texter av rubrik + ev. summary
    texts: List[str] = []
    for item in recent_items:
        title = item.get("title") or ""
        summary = item.get("summary") or ""
        if summary:
            txt = f"{title}. {summary}"
        else:
            txt = title
        # Kort trimming så inte allt blir för långt
        txt = txt.strip()
        if txt:
            texts.append(txt)

    if not texts:
        return {
            "news_score": 0.5,
            "news_reasons": [
                "Inga nyligen identifierade nyhetsartiklar för bolaget – nyhetsbidraget sätts till neutralt (0.50)."
            ],
            "raw_sentiment": 0.0,
            "article_count": 0,
            "last_updated": date.today().isoformat(),
        }

    sentiment_value = analyse_sentiment(texts)
    # Normalisera till [0,1]
    news_score = 0.5 + 0.5 * sentiment_value

    # Skapa ett par reasons
    sentiment_label_sv = (
        "övervägande positiv"
        if sentiment_value > 0.15
        else "övervägande negativ"
        if sentiment_value < -0.15
        else "nära neutral"
    )

    summary_sv = summarise_news_sv(symbol, company_name, texts)

    reasons = [
        f"Nyhetsflödet senaste {NEWS_DAYS} dagarna bedöms som {sentiment_label_sv} (news_score={news_score:.2f}).",
        summary_sv,
    ]

    return {
        "news_score": news_score,
        "news_reasons": reasons,
        "raw_sentiment": sentiment_value,
        "article_count": len(texts),
        "last_updated": date.today().isoformat(),
    }


def main():
    """
    Körs en gång per dag/vecka:
    - analyserar alla bolag i WATCHLIST
    - skriver news_scores.json som används av generate_pick.py
    """
    result: Dict[str, Dict] = {}

    for symbol, name in WATCHLIST:
        try:
            print(f"[INFO] Analyserar nyheter för {symbol}...")
            analysis = analyse_symbol_news(symbol, name)
            result[symbol] = analysis
        except Exception as e:
            print(f"[WARN] Kunde inte analysera nyheter för {symbol}: {e}")

    with open("news_scores.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[INFO] news_scores.json uppdaterad.")


if __name__ == "__main__":
    main()
