import json
from datetime import date, timedelta

def generate_mock_pick():
    today = date.today()
    week_start = today.isoformat()
    week_end = (today + timedelta(days=7)).isoformat()

    pick = {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "week_start": week_start,
        "week_end": week_end,
        "reasons": [
            "Starkt nyhetsflöde efter senaste kvartalsrapporten med positiv analytikerreaktion.",
            "Stabil intjäning och kassaflöde med historiskt lägre volatilitet än många tech-peers.",
            "Teknisk bild visar positiv kortsiktig trend över sitt 20-dagars glidande medelvärde."
        ]
    }

    with open("current_pick.json", "w", encoding="utf-8") as f:
        json.dump(pick, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generate_mock_pick()
