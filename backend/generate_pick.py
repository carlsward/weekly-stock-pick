import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import pandas as pd
import yfinance as yf

# ======= Konfig =======

MODEL_VERSION = "v1.0"  # uppdatera när du ändrar scoring-logik

UNIVERSE_CSV_PATH = "universe.csv"
NEWS_SCORES_PATH = "news_scores.json"


# ======= Datamodeller =======

@dataclass
class StockCandidate:
    symbol: str
    company_name: str
    reasons: List[str]
    score: float          # total score (teknik + nyheter)
    risk_level: str       # "low" / "medium" / "high"
    base_score: float     # bara momentum/vol
    news_score: float     # 0..1
    raw_momentum: float
    raw_vol: float


# ======= Hjälpfunktioner =======

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


def fetch_daily_closes(symbol: str, max_days: int = 20) -> List[float]:
    """
    Hämta senaste dagliga stängningskurserna från Yahoo Finance.
    Vi tar ca 1 månads data och plockar ut de senaste max_days dagarna.
    """
    data = yf.download(
        tickers=symbol,
        period="1mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    closes = data["Close"]

    # Om vi får en DataFrame, ta första kolumnen
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]

    closes = closes.dropna()
    if len(closes) == 0:
        raise RuntimeError(f"Inga prisdata för {symbol}")

    # Ta de senaste max_days värdena, gör ordningen: senaste först
    values = closes.tail(max_days).values  # numpy-array
    closes_list = list(values)[::-1]       # index 0 = senaste
    return closes_list


def load_news_scores(path: str = NEWS_SCORES_PATH) -> Dict[str, dict]:
    """
    Läser in nyhetsbaserade scores per symbol.
    Struktur:
    {
      "MSFT": {
        "news_score": 0.7,
        "news_reasons": ["...", "..."],
        "raw_sentiment": 0.12,
        "article_count": 10,
        "last_updated": "2025-11-18"
      },
      ...
    }
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        print(f"[WARN] Hittar inte {path} – kör utan nyhetsscore (neutral).")
    return {}


def compute_metrics(closes: List[float]) -> tuple[float, float]:
    """
    Beräkna enkel 5-dagars momentum och volatilitet (stdav på dagsavkastningar).
    Antag: closes[0] = senaste stängning, closes[5] = 6 dagar tillbaka.
    """
    if len(closes) < 6:
        raise ValueError("För få datapunkter för att beräkna 5-dagars momentum")

    # Momentum: prisförändring senaste 5 handelsdagar
    momentum_5d = closes[0] / closes[5] - 1.0

    # Dagsavkastningar
    returns: List[float] = []
    for i in range(len(closes) - 1):
        r = closes[i] / closes[i + 1] - 1.0
        returns.append(r)

    if not returns:
        raise ValueError("Kunde inte beräkna avkastningar")

    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    vol = math.sqrt(var_r)

    return momentum_5d, vol


def classify_risk(vol: float) -> str:
    """
    Klassificera enkel risknivå baserat på volatilitet (std på dagsavkastningar).
    Trösklarna kan justeras senare.
    """
    if vol < 0.015:
        return "low"
    elif vol < 0.035:
        return "medium"
    else:
        return "high"


def compute_base_score(momentum: float, vol: float) -> float:
    """
    Enkel riskjusterad score:
    - Belönar positivt momentum
    - Straffar hög volatilitet
    """
    return momentum - vol


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_candidate(
    symbol: str,
    company_name: str,
    news_info: Optional[dict] = None,
) -> StockCandidate:

    closes = fetch_daily_closes(symbol)
    momentum, vol = compute_metrics(closes)
    risk_level = classify_risk(vol)
    base_score = compute_base_score(momentum, vol)

    # Nyhetsdel – default neutral 0.5
    news_score = 0.5
    news_reasons: List[str] = []
    if news_info:
        news_score = float(news_info.get("news_score", 0.5))
        reasons = news_info.get("news_reasons")
        if isinstance(reasons, list):
            news_reasons = [str(r) for r in reasons]

    # Kombinera teknisk + nyhetsscore
    #  - news_score ~0.5 neutralt
    #  - >0.5 boostar, <0.5 drar ned
    news_factor = 0.5 + news_score  # 0.5..1.5 ungefär
    total_score = base_score * news_factor

    reasons = [
        f"Senaste 5 handelsdagarna: cirka {format_pct(momentum)} prisutveckling.",
        f"Uppmätt dagsvolatilitet kring {format_pct(vol)}, klassad som {risk_level} risk.",
        "Riskjusterad modellscore (momentum minus volatilitet) ger en relativt attraktiv profil.",
    ]

    if news_info:
        reasons.append("Nyhetsanalys (AI-modell, sammanfattning av flera källor):")
        if news_reasons:
            reasons.extend(news_reasons)

    return StockCandidate(
        symbol=symbol,
        company_name=company_name,
        reasons=reasons,
        score=total_score,
        risk_level=risk_level,
        base_score=base_score,
        news_score=news_score,
        raw_momentum=momentum,
        raw_vol=vol,
    )


def get_candidates() -> List[StockCandidate]:
    universe = load_universe()
    news_scores = load_news_scores()
    candidates: List[StockCandidate] = []
    for symbol, name in universe:
        try:
            candidate = build_candidate(
                symbol=symbol,
                company_name=name,
                news_info=news_scores.get(symbol),
            )
            candidates.append(candidate)
        except Exception as e:
            print(f"[WARN] Hoppar över {symbol}: {e}")
    if not candidates:
        raise RuntimeError("Inga kandidater kunde genereras")
    return candidates


def select_best_candidate(candidates: List[StockCandidate]) -> StockCandidate:
    # Bästa totala score oavsett risknivå
    return max(candidates, key=lambda c: c.score)


def select_best_per_risk(candidates: List[StockCandidate]) -> Dict[str, StockCandidate]:
    """
    Välj bästa kandidat per risknivå ("low", "medium", "high") baserat på score.
    Returnerar en dict: risk -> kandidat.
    """
    buckets: Dict[str, List[StockCandidate]] = {
        "low": [],
        "medium": [],
        "high": [],
    }

    for c in candidates:
        key = c.risk_level if c.risk_level in buckets else "high"
        buckets[key].append(c)

    best_per_risk: Dict[str, StockCandidate] = {}
    for risk, lst in buckets.items():
        if lst:
            best_per_risk[risk] = max(lst, key=lambda x: x.score)

    return best_per_risk


# ======= JSON-byggare =======

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_output_json(candidate: StockCandidate) -> dict:
    today = date.today()
    week_start = today.isoformat()
    week_end = (today + timedelta(days=7)).isoformat()

    return {
        "symbol": candidate.symbol,
        "company_name": candidate.company_name, 
        "week_start": week_start,
        "week_end": week_end,
        "reasons": candidate.reasons,
        "score": candidate.score,
        "risk": candidate.risk_level,
    }


def update_history(current_pick: dict, history_path: str = "history.json") -> None:
    """
    Lägg till/uppdatera en rad i historiken för veckans pick.
    Sparas som en lista av entries i history.json.
    """
    entry = {
        "logged_at": date.today().isoformat(),
        "symbol": current_pick["symbol"],
        "company_name": current_pick["company_name"],
        "week_start": current_pick["week_start"],
        "week_end": current_pick["week_end"],
        "score": current_pick.get("score"),
        "risk": current_pick.get("risk"),
        "model_version": current_pick.get("model_version", MODEL_VERSION),
    }

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
            if not isinstance(history, list):
                history = []
    except FileNotFoundError:
        history = []

    # Ta bort ev. tidigare entry för samma vecka + symbol
    history = [
        h for h in history
        if not (h.get("week_start") == entry["week_start"] and h.get("symbol") == entry["symbol"])
    ]

    history.append(entry)

    # Sortera på week_start, behåll bara senaste 100 posterna
    history.sort(key=lambda h: h.get("week_start", ""))
    history = history[-100:]

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ======= main =======

def main():
    print("[INFO] Genererar kandidater...")
    candidates = get_candidates()

    # Bästa totalt
    best_overall = select_best_candidate(candidates)
    current_pick = build_output_json(best_overall)

    with open("current_pick.json", "w", encoding="utf-8") as f:
        json.dump(current_pick, f, ensure_ascii=False, indent=2)

    # Bästa per risknivå
    best_per_risk = select_best_per_risk(candidates)
    risk_output = {
        risk: build_output_json(candidate)
        for risk, candidate in best_per_risk.items()
    }

    with open("risk_picks.json", "w", encoding="utf-8") as f:
        json.dump(risk_output, f, ensure_ascii=False, indent=2)

    # Uppdatera historik
    update_history(current_pick)

    print("[INFO] current_pick.json, risk_picks.json och history.json uppdaterade.")


if __name__ == "__main__":
    main()
