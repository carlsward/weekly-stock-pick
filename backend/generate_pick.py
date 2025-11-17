import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

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
    score: float          # högre = bättre
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

    closes_series = data["Close"].dropna()
    if len(closes_series) == 0:
        raise RuntimeError(f"Inga prisdata för {symbol}")

    # Ta de senaste max_days värdena, gör ordningen: senaste först
    values = closes_series.tail(max_days).tolist()
    closes = list(reversed(values))  # index 0 = senaste
    return closes


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


def compute_score(momentum: float, vol: float) -> float:
    """
    Enkel riskjusterad score:
    - Belönar positivt momentum
    - Straffar hög volatilitet
    """
    return momentum - vol


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_candidate(symbol: str, company_name: str) -> StockCandidate:
    closes = fetch_daily_closes(symbol)
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


def get_candidates() -> List[StockCandidate]:
    candidates: List[StockCandidate] = []
    for symbol, name in WATCHLIST:
        try:
            candidate = build_candidate(symbol, name)
            candidates.append(candidate)
        except Exception as e:
            print(f"Hoppar över {symbol}: {e}")
    if not candidates:
        raise RuntimeError("Inga kandidater kunde genereras")
    return candidates


def select_best_candidate(candidates: List[StockCandidate]) -> StockCandidate:
    # Nu: bästa riskjusterade score oavsett risknivå
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
    candidates = get_candidates()
    best = select_best_candidate(candidates)
    output = build_output_json(best)

    with open("current_pick.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
