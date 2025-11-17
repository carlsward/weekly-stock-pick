import json
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

import requests


ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

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
    score: float          # högre = bättre
    risk_level: str       # "low" / "medium" / "high"


def fetch_daily_closes(symbol: str, api_key: str, max_days: int = 20) -> List[float]:
    """Hämta senaste dagliga stängningskurserna för en aktie."""
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "compact",
    }
    response = requests.get(ALPHAVANTAGE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    time_series = data.get("Time Series (Daily)")
    if not time_series:
        raise RuntimeError(f"Inga prisdata för {symbol}: {data}")

    # Sortera datum i fallande ordning (senaste först)
    dates = sorted(time_series.keys(), reverse=True)
    closes: List[float] = []

    for d in dates[:max_days]:
        close_str = time_series[d]["4. close"]
        closes.append(float(close_str))

    return closes


def compute_metrics(closes: List[float]) -> tuple[float, float]:
    """
    Beräkna enkel 5-dagars momentum och volatilitet (stdav på dagsavkastningar).
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
    Trösklarna är grova och kan kalibreras senare.
    """
    if vol < 0.015:
        return "low"
    elif vol < 0.035:
        return "medium"
    else:
        return "high"


def compute_score(momentum: float, vol: float) -> float:
    """
    Enkel riskjusterad score:
    - Belönar positivt momentum
    - Straffar hög volatilitet
    """
    return momentum - vol


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_candidate(symbol: str, company_name: str, api_key: str) -> StockCandidate:
    closes = fetch_daily_closes(symbol, api_key)
    momentum, vol = compute_metrics(closes)
    risk_level = classify_risk(vol)
    score = compute_score(momentum, vol)

    reasons = [
        f"Senaste 5 handelsdagarna: cirka {format_pct(momentum)} prisutveckling.",
        f"Uppmätt dagsvolatilitet kring {format_pct(vol)}, klassad som {risk_level} risk.",
        "Riskjusterad modellscore (momentum minus volatilitet) ger en relativt attraktiv profil.",
    ]

    return StockCandidate(
        symbol=symbol,
        company_name=company_name,
        reasons=reasons,
        score=score,
        risk_level=risk_level,
    )


def get_candidates(api_key: str) -> List[StockCandidate]:
    candidates: List[StockCandidate] = []
    for symbol, name in WATCHLIST:
        try:
            candidate = build_candidate(symbol, name, api_key)
            candidates.append(candidate)
        except Exception as e:
            # Logga men krascha inte hela pipelinen om en ticker strular
            print(f"Hoppar över {symbol}: {e}")
    if not candidates:
        raise RuntimeError("Inga kandidater kunde genereras")
    return candidates


def select_best_candidate(candidates: List[StockCandidate]) -> StockCandidate:
    # Just nu: bästa riskjusterade score oavsett risknivå
    return max(candidates, key=lambda c: c.score)


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


def main():
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY saknas i environment")

    candidates = get_candidates(api_key=api_key)

    # Här integrerar vi din risk-idé i logiken:
    # varje kandidat får en risknivå. Just nu väljer vi bästa totala,
    # men strukturen är redo att t.ex. ta bästa per risknivå senare.
    best = select_best_candidate(candidates)
    output = build_output_json(best)

    with open("current_pick.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
