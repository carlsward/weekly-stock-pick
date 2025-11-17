import json

WATCHLIST = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corporation"),
    ("GOOGL", "Alphabet Inc."),
]


def build_news_entry(symbol: str, company_name: str) -> dict:
    """
    Här kommer du senare att:
    - hämta nyhetsrubriker
    - skicka dem till ett LLM
    - få tillbaka news_score + news_reasons

    Just nu är det bara en placeholder.
    """
    return {
        "news_score": 0.5,
        "news_reasons": [
            f"Placeholder-analys för {company_name}.",
            "Här kommer AI-genererade nyhetsargument senare."
        ]
    }


def main():
    data = {}
    for symbol, name in WATCHLIST:
        data[symbol] = build_news_entry(symbol, name)

    with open("news_scores.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
