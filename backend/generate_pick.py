import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict

import pandas as pd
import yfinance as yf


# Liten watchlist – kan utökas senare
WATCHLIST = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corporation"),
    ("GOOGL", "Alphabet Inc."),
]


@dataclass
class StockCandidate:
    symbol: str
    company_name: str
    reasons: List[str]
    score: float          # total score (teknik + nyheter)
    risk_level: str       # "low" / "medium" / "high"


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


def load_news_scores(path: str = "news_scores.json") -> Dict[str, dict]:
    """
    Läser in nyhetsbaserade scores per symbol.

    Struktur (exempel):
    {
      "MSFT": {
        "news_score": 0.7,
        "news_reasons": [
          "Starkt flöde av positiva analyser.",
          "Flera uppgraderingar senaste veckan."
        ]
      },
      ...
    }

    Tanken är att denna fil senare fylls på av en LLM som
    väger ihop många nyhetskällor.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
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


def compute_model_score(momentum: float, vol: float) -> float:
    """
    Enkel riskjusterad modellscore:
    - Belönar positivt momentum
    - Straffar hög volatilitet
    """
    return momentum - vol


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_candidate(
    symbol: str,
    company_name: str,
    news_info: dict | None = None,
    news_weight: float = 0.5,
) -> StockCandidate:
    """
    Bygger upp en kandidat baserat på prisdata + ev. nyhetsinfo.

    news_info förväntas ha nycklar:
      - "news_score": float (t.ex. -1.0 .. +1.0)
      - "news_reasons": list[str]
    """
    closes = fetch_daily_closes(symbol)
    momentum, vol = compute_metrics(closes)
    risk_level = classify_risk(vol)

    # Modellscore baserad på prisdata
    model_score = compute_model_score(momentum, vol)

    # Nyhetsscore (från LLM/extern process)
    news_score = 0.0
    news_reasons: List[str] = []
    if isinstance(news_info, dict):
        try:
            news_score = float(news_info.get("news_score", 0.0))
        except (TypeError, ValueError):
            news_score = 0.0

        nr = news_info.get("news_reasons")
        if isinstance(nr, list):
            news_reasons = [str(x) for x in nr]

    # Total score = modellscore + vikt * nyhetsscore
    total_score = model_score + news_weight * news_score

    reasons: List[str] = [
        f"Senaste 5 handelsdagarna: cirka {format_pct(momentum)} prisutveckling.",
        f"Uppmätt dagsvolatilitet kring {format_pct(vol)}, klassad som {risk_level} risk.",
        "Riskjusterad modellscore (momentum minus volatilitet) ger en relativt attraktiv profil.",
    ]

    # Lägg till sammanfattande nyhetsrad
    if news_score != 0.0:
        direction = "övervägande positivt" if news_score > 0 else "övervägande negativt"
        reasons.append(
            f"Nyhetsanalys (AI-modell) indikerar {direction} sentiment för bolaget (news_score={news_score:.2f})."
        )

    # Lägg till mer detaljerade nyhetsmotiveringar om de finns
    for r in news_reasons:
        reasons.append(r)

    # Om vi inte hade någon nyhetsdata alls – gör det tydligt att det är ett hål
    if not news_reasons and news_score == 0.0:
        reasons.append(
            "Nyhetsanalys saknas för tillfället – endast prisbaserad modell ligger till grund för rekommendationen."
        )

    return StockCandidate(
        symbol=symbol,
        company_name=company_name,
        reasons=reasons,
        score=total_score,
        risk_level=risk_level,
    )


def get_candidates(news_scores: Dict[str, dict]) -> List[StockCandidate]:
    """
    Bygger kandidater för alla aktier i WATCHLIST, med ev. nyhetsdata.
    """
    candidates: List[StockCandidate] = []

    for symbol, name in WATCHLIST:
        try:
            news_info = news_scores.get(symbol)
            candidate = build_candidate(symbol, name, news_info=news_info)
            candidates.append(candidate)
        except Exception as e:
            print(f"Hoppar över {symbol}: {e}")

    if not candidates:
        raise RuntimeError("Inga kandidater kunde genereras")
    return candidates


def select_best_candidate(candidates: List[StockCandidate]) -> StockCandidate:
    # Bästa totala score oavsett risknivå
    return max(candidates, key=lambda c: c.score)


def select_best_per_risk(candidates: List[StockCandidate]) -> Dict[str, StockCandidate]:
    """
    Välj bästa kandidat per risknivå ("low", "medium", "high") baserat på total score.
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


def main() -> None:
    # 1) Läs in nyhetsscorer (framtida LLM-output)
    news_scores = load_news_scores()

    # 2) Bygg kandidater
    candidates = get_candidates(news_scores=news_scores)

    # 3) Bästa totalt (samma fil som appen redan läser)
    best_overall = select_best_candidate(candidates)
    current_pick = build_output_json(best_overall)

    with open("current_pick.json", "w", encoding="utf-8") as f:
        json.dump(current_pick, f, ensure_ascii=False, indent=2)

    # 4) Bästa per risknivå – ny fil som appen också använder
    best_per_risk = select_best_per_risk(candidates)
    risk_output = {
        risk: build_output_json(candidate)
        for risk, candidate in best_per_risk.items()
    }

    with open("risk_picks.json", "w", encoding="utf-8") as f:
        json.dump(risk_output, f, ensure_ascii=False, indent=2)

    # 5) Uppdatera historik
    update_history(current_pick)


if __name__ == "__main__":
    main()
