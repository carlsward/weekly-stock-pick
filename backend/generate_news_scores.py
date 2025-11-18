# backend/generate_news_scores.py

import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import yfinance as yf

from generate_pick import WATCHLIST  # (symbol, company_name)


# ----------------------------
#  Device-val (CPU / MPS / CUDA)
# ----------------------------

def get_device() -> Tuple[int, str]:
    if torch.backends.mps.is_available():
        print("Device set to use mps:0")
        return 0, "mps"
    if torch.cuda.is_available():
        print("Device set to use cuda:0")
        return 0, "cuda"
    print("Device set to use cpu")
    return -1, "cpu"


print("Laddar FinBERT-modell...")
device_index, _device_name = get_device()

FINBERT_MODEL_NAME = "ProsusAI/finbert"

_finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
_finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
finbert_pipeline = pipeline(
    "sentiment-analysis",
    model=_finbert_model,
    tokenizer=_finbert_tokenizer,
    device=device_index,
)

print("Laddar summarizer-modell...")
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"
summarizer_pipeline = pipeline(
    "summarization",
    model=SUMMARIZER_MODEL_NAME,
    device=device_index,
)


# ----------------------------
#  Hjälpfunktioner
# ----------------------------

def parse_item_datetime(item: Dict[str, Any]) -> datetime | None:
    """
    Försöker tolka tid från både nya och gamla yfinance-format:

    Nya: item["content"]["pubDate"] = "2025-11-18T11:00:42Z"
    Gamla: item["providerPublishTime"] = epoch-sekunder
    """
    content = item.get("content") or {}
    iso_str = content.get("pubDate") or item.get("pubDate")
    if iso_str:
        try:
            # Ex: "2025-11-18T11:00:42Z"
            if iso_str.endswith("Z"):
                iso_str = iso_str.replace("Z", "+00:00")
            return datetime.fromisoformat(iso_str).astimezone(timezone.utc)
        except Exception:
            pass

    ts = item.get("providerPublishTime")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    return None


def extract_title_and_text(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Plockar ut title + summary från både nya och gamla format.
    Returnerar (title, full_text).
    """
    content = item.get("content") or {}

    title = (
        content.get("title")
        or item.get("title")
        or ""
    ).strip()

    summary = (
        content.get("summary")
        or item.get("summary")
        or ""
    ).strip()

    # vi kan även falla tillbaka till "description" om summary saknas
    if not summary:
        summary = (
            content.get("description")
            or item.get("description")
            or ""
        ).strip()

    if summary:
        full_text = f"{title}. {summary}" if title else summary
    else:
        full_text = title

    return title, full_text.strip()


def fetch_news_texts_for_symbol(
    symbol: str,
    max_items: int = 20,
    max_age_days: int = 3,
) -> List[str]:
    """
    Hämtar nyheter via yfinance.Ticker(symbol).news och
    returnerar en lista med textsträngar (title + summary)
    för artiklar senaste max_age_days dagarna.
    """
    ticker = yf.Ticker(symbol)
    raw_news = ticker.news or []
    print(f"[{symbol}] raw news count: {len(raw_news)}")

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max_age_days)
    texts: List[str] = []

    for idx, item in enumerate(raw_news[:max_items]):
        dt = parse_item_datetime(item)
        title, full_text = extract_title_and_text(item)

        print(f"[{symbol}] item {idx}: time={dt} title={repr(title)}")

        if dt is None or dt < cutoff:
            continue
        if not full_text:
            continue

        texts.append(full_text)

    print(f"[{symbol}] texts used: {len(texts)}")
    return texts


def sentiment_to_scalar(label: str, score: float) -> float:
    """
    Mappa FinBERTs label + score till ett tal i intervallet [-1, 1].
    """
    label = label.lower()
    if label == "positive":
        return +score
    if label == "negative":
        return -score
    # neutral
    return 0.0


def analyze_news_with_finbert(texts: List[str]) -> Tuple[float, float]:
    """
    Kör FinBERT på varje text och returnerar:
    - avg_raw: medelvärde i [-1, 1]
    - normalized: mappad till [0, 1]
    """
    if not texts:
        return 0.0, 0.5

    scores: List[float] = []
    for t in texts:
        try:
            res = finbert_pipeline(t[:512])[0]  # klipp för säkerhets skull
            raw = sentiment_to_scalar(res["label"], float(res["score"]))
            scores.append(raw)
        except Exception as e:
            print(f"[WARN] FinBERT fel på text: {e}")

    if not scores:
        return 0.0, 0.5

    avg_raw = sum(scores) / len(scores)
    # mappa [-1, 1] -> [0, 1]
    normalized = 0.5 + 0.5 * max(-1.0, min(1.0, avg_raw))
    return avg_raw, normalized


def summarize_texts(texts: List[str], max_chars: int = 4000) -> str:
    """
    Slår ihop texter och kör en enkel summarization.
    """
    if not texts:
        return ""

    combined = "\n".join(texts)
    combined = combined[:max_chars]

    try:
        summary = summarizer_pipeline(
            combined,
            max_length=180,
            min_length=60,
            do_sample=False,
        )[0]["summary_text"]
        return summary.strip()
    except Exception as e:
        print(f"[WARN] Summarizer fel: {e}")
        return ""


def build_news_entry(symbol: str) -> Dict[str, Any]:
    texts = fetch_news_texts_for_symbol(symbol)

    if not texts:
        # fallback – exakt samma text som du har idag
        return {
            "news_score": 0.50,
            "news_reasons": [
                "Inga nyligen identifierade nyhetsartiklar för bolaget – nyhetsbidraget sätts till neutralt (0.50)."
            ],
            "raw_sentiment": 0.0,
            "article_count": 0,
            "last_updated": datetime.today().date().isoformat(),
        }

    avg_raw, normalized = analyze_news_with_finbert(texts)
    summary = summarize_texts(texts)

    reasons = [
        f"Nyhetsanalys (AI-modell, FinBERT) baserad på {len(texts)} artiklar senaste dagarna.",
    ]
    if summary:
        reasons.append(f"Sammanfattning: {summary}")

    return {
        "news_score": round(normalized, 2),
        "news_reasons": reasons,
        "raw_sentiment": round(avg_raw, 4),
        "article_count": len(texts),
        "last_updated": datetime.today().date().isoformat(),
    }


# ----------------------------
#  Huvudflöde
# ----------------------------

def main() -> None:
    result: Dict[str, Dict[str, Any]] = {}

    for symbol, company_name in WATCHLIST:
        print(f"\n=== Hämtar nyheter för {symbol} ===")
        entry = build_news_entry(symbol)
        result[symbol] = entry

    with open("news_scores.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nnews_scores.json uppdaterad.")


if __name__ == "__main__":
    main()
