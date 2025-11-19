import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline

NEWS_SCORES_PATH = "news_scores.json"
UNIVERSE_CSV_PATH = "universe.csv"

FINBERT_MODEL_NAME = "ProsusAI/finbert"
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"

# Hur långt tillbaka vi tittar på nyheter
NEWS_LOOKBACK_DAYS = 5
MAX_ARTICLES_PER_SYMBOL = 10


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

def parse_news_time(raw: dict) -> datetime | None:
    """
    Försöker plocka ut tid från yfinance-nyhetsobjekt.
    """
    content = raw.get("content") or {}
    ts = content.get("pubDate") or content.get("displayTime")
    if not ts:
        return None
    try:
        # Ex: "2025-11-18T11:00:42Z"
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
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

    keywords = build_symbol_keywords(symbol, company_name)
    if not text.strip():
        return False

    return any(k in text for k in keywords)


def fetch_symbol_news(symbol: str, company_name: str) -> List[str]:
    """
    Hämtar nyhetstexter (titel + summary) för ett bolag, filtrerar på datum och relevans.
    Returnerar en lista med textstycken som kan sentimentanalyseras/summeras.
    """
    ticker = yf.Ticker(symbol)
    raw_news = ticker.news or []

    print(f"\n=== Hämtar nyheter för {symbol} ===")
    print(f"[{symbol}] raw news count: {len(raw_news)}")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=NEWS_LOOKBACK_DAYS)

    texts: List[str] = []
    for idx, item in enumerate(raw_news[: MAX_ARTICLES_PER_SYMBOL * 2]):
        t = parse_news_time(item)
        content = item.get("content") or {}
        title = content.get("title") or ""
        summary = content.get("summary") or ""
        desc = content.get("description") or ""

        print(f"[{symbol}] item {idx}: time={t} title='{title}'")

        # Tidfilter
        if t is not None and t < cutoff:
            continue

        # Relevansfilter
        if not is_relevant_news_item(item, symbol, company_name):
            continue

        combined = " ".join([title, summary, desc]).strip()
        if combined:
            texts.append(combined)

        if len(texts) >= MAX_ARTICLES_PER_SYMBOL:
            break

    print(f"[{symbol}] texts used: {len(texts)}")
    return texts


# ---------- Modellinit ----------

def init_models():
    print("Laddar FinBERT-modell...")
    detect_device_for_logs()

    finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=finbert_model,
        tokenizer=finbert_tokenizer,
        device=-1,  # CPU för robusthet
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

    return sentiment_pipe, summarizer_pipe


# ---------- Sentiment + sammanfattning ----------

def compute_finbert_sentiment(pipe, texts: List[str]) -> float:
    """
    Kör FinBERT på varje text och beräknar ett snitt i intervallet [-1, 1].
    """
    if not texts:
        return 0.0

    scored: List[float] = []
    # Begränsa längden per text (tecken) för säkerhets skull
    inputs = [t[:512] for t in texts]

    results = pipe(inputs, truncation=True)
    for res in results:
        label = res["label"].lower()
        score = float(res["score"])
        if "positive" in label:
            scored.append(+score)
        elif "negative" in label:
            scored.append(-score)
        else:
            scored.append(0.0)

    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def summarize_texts(pipe, texts: List[str]) -> str:
    """
    Summerar alla texter till en relativt kort sammanfattning.
    """
    if not texts:
        return ""

    joined = " ".join(texts)
    joined = joined[:4000]  # grov begränsning
    try:
        out = pipe(
            joined,
            max_length=80,   # kortare sammanfattning
            min_length=30,   # fortfarande minst några meningar
            do_sample=False,
            truncation=True,
        )
        if out and isinstance(out, list):
            return out[0].get("summary_text", "").strip()
    except Exception as e:
        print(f"[WARN] Summarization failed: {e}")
    # Fallback: använd bara ihopslagna titlar
    return " ".join(t[:200] for t in texts[:3])



def sentiment_to_news_score(raw_sent: float) -> float:
    """
    Mappar råsentiment [-1, 1] till [0, 1].
    0 = starkt negativt, 0.5 = neutralt, 1 = starkt positivt.
    """
    x = max(-1.0, min(1.0, raw_sent))
    return 0.5 + 0.5 * x


# ---------- Huvudlogik ----------

def main():
    sentiment_pipe, summarizer_pipe = init_models()
    universe = load_universe()

    out: Dict[str, dict] = {}
    today_str = datetime.now(timezone.utc).date().isoformat()

    for symbol, company_name in universe:
        texts = fetch_symbol_news(symbol, company_name)

        if not texts:
            print(f"[{symbol}] Inga relevanta nyheter – sätter neutral 0.50")
            out[symbol] = {
                "news_score": 0.5,
                "news_reasons": [
                    "Inga nyligen identifierade relevanta nyhetsartiklar för bolaget – "
                    "nyhetsbidraget sätts till neutralt (0.50)."
                ],
                "raw_sentiment": 0.0,
                "article_count": 0,
                "last_updated": today_str,
            }
            continue

        raw_sent = compute_finbert_sentiment(sentiment_pipe, texts)
        news_score = sentiment_to_news_score(raw_sent)
        summary = summarize_texts(summarizer_pipe, texts)

        reasons = [
            f"Nyhetsanalys (AI-modell, FinBERT) baserad på {len(texts)} artiklar senaste dagarna.",
            f"Sammanfattning för {symbol} ({company_name}): {summary}",
        ]

        out[symbol] = {
            "news_score": round(news_score, 2),
            "news_reasons": reasons,
            "raw_sentiment": round(raw_sent, 4),
            "article_count": len(texts),
            "last_updated": today_str,
        }


    with open(NEWS_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nnews_scores.json uppdaterad.")


if __name__ == "__main__":
    main()
