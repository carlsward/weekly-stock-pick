import json
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import List


@dataclass
class StockCandidate:
    symbol: str
    company_name: str
    reasons: List[str]
    score: float  # högre = bättre


def get_candidates() -> List[StockCandidate]:
    """
    Här definierar vi några mock-kandidater.
    Senare kan vi ersätta detta med riktig logik:
    - hämta nyheter
    - beräkna sentiment
    - räkna fram score
    """
    return [
        StockCandidate(
            symbol="AAPL",
            company_name="Apple Inc.",
            reasons=[
                "Starkt nyhetsflöde efter senaste kvartalsrapporten med positiv analytikerreaktion.",
                "Stabil intjäning och kassaflöde med historiskt lägre volatilitet än många tech-peers.",
                "Teknisk bild visar positiv kortsiktig trend över sitt 20-dagars glidande medelvärde.",
            ],
            score=0.82,
        ),
        StockCandidate(
            symbol="MSFT",
            company_name="Microsoft Corporation",
            reasons=[
                "Positivt momentum inom molnverksamheten enligt senaste rapporter.",
                "Hög lönsamhet och stark balansräkning med nettokassa.",
                "Brett diversifierad intäktsbas inom både företags- och konsumentsegment.",
            ],
            score=0.88,
        ),
        StockCandidate(
            symbol="GOOGL",
            company_name="Alphabet Inc.",
            reasons=[
                "Stabil annonsaffär och växande molnverksamhet.",
                "Stark historik av kassaflöde och återköp av aktier.",
                "Exponering mot AI-tillväxt genom flera produktlinjer.",
            ],
            score=0.79,
        ),
    ]


def select_best_candidate(candidates: List[StockCandidate]) -> StockCandidate:
    # Just nu: välj den med högst score
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
        # Bra att ha med internt score-värde för framtida debug
        "score": candidate.score,
    }


def main():
    candidates = get_candidates()
    best = select_best_candidate(candidates)
    output = build_output_json(best)

    with open("current_pick.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
