import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import pandas as pd
import yfinance as yf

# ======= Konfig =======

MODEL_VERSION = "v1.2"  # uppdatera när du ändrar scoring-logik
VOL_WEIGHT = 0.7        # hur hårt volatilitet straffas i standardiserat mått
NEWS_ALPHA = 0.6        # hur mycket nyhetsfaktorn påverkar (0.5 neutralt)
NEWS_FACTOR_MIN = 0.7
NEWS_FACTOR_MAX = 1.3
MIN_ARTICLES_FOR_NEWS = 2
PRICE_CACHE_PATH = "price_cache.json"
MIN_DOLLAR_VOL = 5_000_000  # minsta genomsnittliga dollarvolym för att undvika illikvida namn

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
    news_factor: float
    raw_momentum: float
    raw_momentum_20d: float
    raw_vol: float
    raw_drawdown: float
    avg_dollar_vol: float
    momentum_z: float
    momentum_20d_z: float
    vol_z: float
    drawdown_z: float
    dollar_vol_z: float


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


def fetch_daily_closes(symbol: str, cache: Dict[str, dict], max_days: int = 40) -> tuple[List[float], List[float]]:
    """
    Hämta senaste dagliga stängningskurserna från Yahoo Finance.
    Vi tar ca 1 månads data och plockar ut de senaste max_days dagarna.
    """
    data = None
    for attempt in range(3):
        try:
            data = yf.download(
                tickers=symbol,
                period="2mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            break
        except Exception as e:
            if attempt == 2:
                data = None
                break
            sleep_time = 1 + attempt
            print(f"[{symbol}] pris-hämtning misslyckades (försök {attempt+1}): {e} – väntar {sleep_time}s")
            time.sleep(sleep_time)

    closes = None
    volumes = None
    if data is not None:
        try:
            closes = data["Close"]
            volumes = data["Volume"]
        except Exception:
            closes = None
            volumes = None

    # Om vi får en DataFrame, ta första kolumnen
    if closes is not None and isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    if volumes is not None and isinstance(volumes, pd.DataFrame):
        volumes = volumes.iloc[:, 0]

    if closes is not None:
        closes = closes.dropna()
        if len(closes) > 0:
            aligned_vols = None
            if volumes is not None:
                try:
                    aligned_vols = volumes.loc[closes.index]
                    aligned_vols = aligned_vols.fillna(0)
                except Exception:
                    aligned_vols = None
            values = closes.tail(max_days).values
            close_list = list(values)[::-1]
            vol_list = []
            if aligned_vols is not None:
                vol_values = aligned_vols.tail(max_days).values
                vol_list = list(vol_values)[::-1]
            cache[symbol] = {"close": close_list, "volume": vol_list}
            return close_list, vol_list

    # Fallback till cache
    cached = cache.get(symbol)
    if cached:
        print(f"[{symbol}] Använder cachelagda prisdata.")
        return cached.get("close", []), cached.get("volume", [])

    raise RuntimeError(f"Inga prisdata för {symbol}")


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


def compute_price_features(closes: List[float], volumes: List[float]) -> Dict[str, float]:
    """
    Beräkna momentum och volatilitet + enkel max drawdown.
    Antag: closes[0] = senaste stängning.
    """
    if len(closes) < 6:
        raise ValueError("För få datapunkter för att beräkna 5-dagars momentum")

    momentum_5d = closes[0] / closes[5] - 1.0

    # Momentum 20d om möjligt
    if len(closes) >= 21:
        momentum_20d = closes[0] / closes[20] - 1.0
    else:
        momentum_20d = momentum_5d

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

    # Max drawdown över senaste ~20 dagar
    chron = list(reversed(closes))  # äldst först
    peak = chron[0]
    max_dd = 0.0
    for price in chron:
        if price > peak:
            peak = price
        drawdown = (price / peak) - 1.0
        if drawdown < max_dd:
            max_dd = drawdown

    avg_dollar_vol = 0.0
    if volumes and closes:
        # align lengths
        n = min(len(closes), len(volumes))
        if n > 0:
            dollar_vol = [float(closes[i]) * float(volumes[i]) for i in range(n)]
            avg_dollar_vol = sum(dollar_vol) / n

    return {
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "vol": vol,
        "max_drawdown": abs(max_dd),  # positiv storlek
        "avg_dollar_vol": avg_dollar_vol,
    }


def classify_risk(vol: float) -> str:
    """
    Klassificera enkel risknivå baserat på volatilitet (std på dagsavkastningar).

    Nuvarande trösklar (daglig stdavkastning):
    - < 1.0%  -> low
    - 1.0–2.0% -> medium
    - > 2.0%  -> high

    Justera vid behov om för få/för många aktier hamnar i high.
    """
    if vol < 0.01:
        return "low"
    elif vol < 0.02:
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


def mean_and_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean_v = sum(values) / len(values)
    var_v = sum((v - mean_v) ** 2 for v in values) / len(values)
    std_v = math.sqrt(var_v)
    if std_v <= 1e-6:
        std_v = 1.0
    return mean_v, std_v


def z_score(value: float, mean_v: float, std_v: float) -> float:
    if std_v <= 1e-9:
        return 0.0
    return (value - mean_v) / std_v


def load_price_cache() -> Dict[str, List[float]]:
    try:
        with open(PRICE_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                normalized = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        closes = v.get("close")
                        vols = v.get("volume")
                        if isinstance(closes, list):
                            normalized[k] = {"close": closes, "volume": vols if isinstance(vols, list) else []}
                    elif isinstance(v, list):
                        normalized[k] = {"close": v, "volume": []}
                return normalized
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return {}


def save_price_cache(cache: Dict[str, List[float]]) -> None:
    try:
        with open(PRICE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_candidate(
    symbol: str,
    company_name: str,
    momentum: float,
    momentum_20d: float,
    vol: float,
    drawdown: float,
    avg_dollar_vol: float,
    momentum_z: float,
    momentum_20d_z: float,
    vol_z: float,
    drawdown_z: float,
    dollar_vol_z: float,
    news_info: Optional[dict] = None,
) -> StockCandidate:

    risk_level = classify_risk(vol)
    base_score = (
        0.6 * momentum_z
        + 0.35 * momentum_20d_z
        - VOL_WEIGHT * vol_z
        - 0.3 * drawdown_z
        + 0.1 * dollar_vol_z
    )

    # Nyhetsdel – default neutral 0.5
    news_score = 0.5
    news_reasons: List[str] = []
    article_count = MIN_ARTICLES_FOR_NEWS
    if news_info:
        news_score = float(news_info.get("news_score", 0.5))
        article_count = int(news_info.get("article_count", MIN_ARTICLES_FOR_NEWS))
        reasons = news_info.get("news_reasons")
        if isinstance(reasons, list):
            news_reasons = [str(r) for r in reasons]

    if news_info and article_count < MIN_ARTICLES_FOR_NEWS:
        news_score = 0.5
        if news_reasons:
            news_reasons.append(
                "News contribution neutralized due to limited recent articles."
            )

    # Kombinera teknisk + nyhetsscore
    #  - news_score ~0.5 neutralt
    #  - >0.5 boostar, <0.5 drar ned
    news_factor = 1 + NEWS_ALPHA * (news_score - 0.5)
    news_factor = max(NEWS_FACTOR_MIN, min(NEWS_FACTOR_MAX, news_factor))
    total_score = base_score * news_factor

    reasons = [
        f"Price performance over the last 5 trading days: about {format_pct(momentum)} (z={momentum_z:.2f}).",
        f"20-day momentum: about {format_pct(momentum_20d)} (z={momentum_20d_z:.2f}).",
        f"Measured daily volatility around {format_pct(vol)}, classified as {risk_level} risk (z={vol_z:.2f}).",
        f"Recent max drawdown ~{format_pct(-drawdown)} (z={drawdown_z:.2f}).",
        f"Avg dollar volume ≈ ${avg_dollar_vol:,.0f} (z={dollar_vol_z:.2f}).",
        f"Base score (0.6*mom5z + 0.35*mom20z − {VOL_WEIGHT:.2f}×volz − 0.3×ddz + 0.1×liq_z): {base_score:.3f}.",
    ]

    if news_info:
        reasons.append("News analysis (AI model, summary across multiple sources):")
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
        news_factor=news_factor,
        raw_momentum=momentum,
        raw_momentum_20d=momentum_20d,
        raw_vol=vol,
        raw_drawdown=drawdown,
        avg_dollar_vol=avg_dollar_vol,
        momentum_z=momentum_z,
        momentum_20d_z=momentum_20d_z,
        vol_z=vol_z,
        drawdown_z=drawdown_z,
        dollar_vol_z=dollar_vol_z,
    )


def get_candidates() -> List[StockCandidate]:
    universe = load_universe()
    news_scores = load_news_scores()
    candidates: List[StockCandidate] = []
    collected: List[dict] = []
    price_cache = load_price_cache()

    for symbol, name in universe:
        try:
            closes, volumes = fetch_daily_closes(symbol, cache=price_cache)
            features = compute_price_features(closes, volumes)
            if features["avg_dollar_vol"] < MIN_DOLLAR_VOL:
                print(f"[WARN] Hoppar över {symbol}: för låg dollarvolym ({features['avg_dollar_vol']:,.0f})")
                continue
            collected.append(
                {
                    "symbol": symbol,
                    "company_name": name,
                    "momentum_5d": features["momentum_5d"],
                    "momentum_20d": features["momentum_20d"],
                    "vol": features["vol"],
                    "drawdown": features["max_drawdown"],
                    "avg_dollar_vol": features["avg_dollar_vol"],
                    "news_info": news_scores.get(symbol),
                }
            )
        except Exception as e:
            print(f"[WARN] Hoppar över {symbol}: {e}")

    mom_mean, mom_std = mean_and_std([c["momentum_5d"] for c in collected])
    mom20_mean, mom20_std = mean_and_std([c["momentum_20d"] for c in collected])
    vol_mean, vol_std = mean_and_std([c["vol"] for c in collected])
    dd_mean, dd_std = mean_and_std([c["drawdown"] for c in collected])
    liq_mean, liq_std = mean_and_std([c["avg_dollar_vol"] for c in collected])

    for item in collected:
        mom_z = z_score(item["momentum_5d"], mom_mean, mom_std)
        mom20_z = z_score(item["momentum_20d"], mom20_mean, mom20_std)
        vol_z = z_score(item["vol"], vol_mean, vol_std)
        dd_z = z_score(item["drawdown"], dd_mean, dd_std)
        liq_z = z_score(item["avg_dollar_vol"], liq_mean, liq_std)
        candidate = build_candidate(
            symbol=item["symbol"],
            company_name=item["company_name"],
            momentum=item["momentum_5d"],
            momentum_20d=item["momentum_20d"],
            vol=item["vol"],
            drawdown=item["drawdown"],
            avg_dollar_vol=item["avg_dollar_vol"],
            momentum_z=mom_z,
            momentum_20d_z=mom20_z,
            vol_z=vol_z,
            drawdown_z=dd_z,
            dollar_vol_z=liq_z,
            news_info=item["news_info"],
        )
        candidates.append(candidate)

    save_price_cache(price_cache)
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
        # Risk-kvalitetsfilter
        if key == "low":
            if c.vol_z > 1.0 or c.drawdown_z > 1.0 or c.base_score < -0.2:
                continue
        elif key == "medium":
            if c.vol_z > 1.6 or c.drawdown_z > 1.4 or c.base_score < -0.4:
                continue
        else:  # high
            if c.momentum_z < -0.8:
                continue
        buckets[key].append(c)

    best_per_risk: Dict[str, StockCandidate] = {}
    for risk, lst in buckets.items():
        if lst:
            best_per_risk[risk] = max(lst, key=lambda x: x.score)
        else:
            # Fallback: ta bästa totala om riskhinken blev tom efter filter
            best_per_risk[risk] = max(candidates, key=lambda x: x.score)

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
        "model_version": MODEL_VERSION,
        "components": {
            "momentum": candidate.raw_momentum,
            "momentum_20d": candidate.raw_momentum_20d,
            "vol": candidate.raw_vol,
            "drawdown": candidate.raw_drawdown,
            "avg_dollar_vol": candidate.avg_dollar_vol,
            "momentum_z": candidate.momentum_z,
            "momentum_20d_z": candidate.momentum_20d_z,
            "vol_z": candidate.vol_z,
            "drawdown_z": candidate.drawdown_z,
            "dollar_vol_z": candidate.dollar_vol_z,
            "base_score": candidate.base_score,
            "news_score": candidate.news_score,
            "news_factor": candidate.news_factor,
        },
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skriv inte ut filer, skriv bara toppkandidater")
    parser.add_argument("--top-n", type=int, default=5, help="Hur många kandidater att visa i dry-run")
    args = parser.parse_args()

    print("[INFO] Genererar kandidater...")
    candidates = get_candidates()

    if not candidates:
        print("[WARN] Inga kandidater kunde genereras (troligen ingen prisdata tillgänglig). Försöker återanvända tidigare picks.")
        try:
            with open("current_pick.json", "r", encoding="utf-8") as f:
                current_pick = json.load(f)
        except Exception:
            today = date.today()
            current_pick = {
                "symbol": "N/A",
                "company_name": "Data unavailable",
                "week_start": today.isoformat(),
                "week_end": (today + timedelta(days=7)).isoformat(),
                "reasons": [
                    "No price data available for the current universe; using a neutral placeholder."
                ],
                "score": 0.0,
                "risk": "unknown",
                "model_version": MODEL_VERSION,
            }
        try:
            with open("risk_picks.json", "r", encoding="utf-8") as f:
                risk_output = json.load(f)
        except Exception:
            risk_output = {"low": current_pick, "medium": current_pick, "high": current_pick}

        with open("current_pick.json", "w", encoding="utf-8") as f:
            json.dump(current_pick, f, ensure_ascii=False, indent=2)
        with open("risk_picks.json", "w", encoding="utf-8") as f:
            json.dump(risk_output, f, ensure_ascii=False, indent=2)
        return

    if args.dry_run:
        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
        print("\nTop candidates (dry run):")
        for c in sorted_cands[: args.top_n]:
            print(
                f"{c.symbol} | score={c.score:.3f} | risk={c.risk_level} | "
                f"mom5={c.raw_momentum:.3%} z={c.momentum_z:.2f} | "
                f"mom20={c.raw_momentum_20d:.3%} z={c.momentum_20d_z:.2f} | "
                f"vol={c.raw_vol:.3%} z={c.vol_z:.2f} | "
                f"dd={c.raw_drawdown:.3%} z={c.drawdown_z:.2f} | "
                f"news={c.news_score:.2f} factor={c.news_factor:.2f}"
            )
        return

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
