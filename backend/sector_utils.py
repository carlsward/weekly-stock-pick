from pathlib import Path
from typing import Dict, List

import pandas as pd

SECTOR_DISPLAY_NAMES = {
    "communication_services": "Communication Services",
    "consumer_discretionary": "Consumer Discretionary",
    "consumer_staples": "Consumer Staples",
    "energy": "Energy",
    "financials": "Financials",
    "healthcare": "Healthcare",
    "industrials": "Industrials",
    "technology": "Technology",
}


def normalize_sector_name(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def sector_display_name(sector: str) -> str:
    normalized = normalize_sector_name(sector)
    return SECTOR_DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())


def load_sector_map(path: Path) -> Dict[str, str]:
    metadata = load_symbol_metadata(path)
    return {symbol: values["sector"] for symbol, values in metadata.items()}


def load_symbol_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(path)
    df = df[df.get("active", 1) == 1]
    if "sector" not in df.columns:
        raise RuntimeError("universe.csv must include a sector column for all active symbols")

    symbol_metadata: Dict[str, Dict[str, str]] = {}
    missing: List[str] = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        company_name = str(row.get("company_name", "")).strip()
        sector = normalize_sector_name(row.get("sector", ""))
        if not symbol:
            continue
        if not company_name:
            missing.append(symbol)
            continue
        if not sector:
            missing.append(symbol)
            continue
        symbol_metadata[symbol] = {
            "company_name": company_name,
            "sector": sector,
        }

    if missing:
        raise RuntimeError(
            "All active symbols must include company_name and sector in universe.csv. Missing data for: "
            + ", ".join(sorted(missing))
        )

    if not symbol_metadata:
        raise RuntimeError("No active symbol metadata found in universe.csv")

    return symbol_metadata


def load_active_sectors(path: Path) -> List[str]:
    return sorted(set(load_sector_map(path).values()))
