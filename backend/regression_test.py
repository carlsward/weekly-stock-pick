"""
Lightweight regression check for the scoring pipeline without network calls.

Usage:
    python backend/regression_test.py

This builds synthetic candidates with fixed price features and news scores,
then asserts the ranking order so you notice accidental scoring regressions.
"""

from generate_pick import (
    mean_and_std,
    z_score,
    build_candidate,
)


def build_candidates():
    # Synthetic feature snapshots (latest close perspective)
    samples = [
        {
            "symbol": "AAA",
            "company_name": "Alpha Corp",
            "mom5": 0.08,
            "mom20": 0.12,
            "vol": 0.012,
            "dd": 0.03,
            "liq": 15_000_000,
            "news": {"news_score": 0.65, "article_count": 5, "news_reasons": ["Positive synthetic news."]},
        },
        {
            "symbol": "BBB",
            "company_name": "Beta Inc",
            "mom5": 0.02,
            "mom20": 0.04,
            "vol": 0.009,
            "dd": 0.015,
            "liq": 9_000_000,
            "news": {"news_score": 0.55, "article_count": 3, "news_reasons": ["Neutral news."]},
        },
        {
            "symbol": "CCC",
            "company_name": "Gamma Ltd",
            "mom5": -0.01,
            "mom20": 0.0,
            "vol": 0.018,
            "dd": 0.05,
            "liq": 6_000_000,
            "news": {"news_score": 0.35, "article_count": 4, "news_reasons": ["Negative news."]},
        },
    ]

    mom_mean, mom_std = mean_and_std([s["mom5"] for s in samples])
    mom20_mean, mom20_std = mean_and_std([s["mom20"] for s in samples])
    vol_mean, vol_std = mean_and_std([s["vol"] for s in samples])
    dd_mean, dd_std = mean_and_std([s["dd"] for s in samples])
    liq_mean, liq_std = mean_and_std([s["liq"] for s in samples])

    candidates = []
    for s in samples:
        c = build_candidate(
            symbol=s["symbol"],
            company_name=s["company_name"],
            momentum=s["mom5"],
            momentum_20d=s["mom20"],
            vol=s["vol"],
            drawdown=s["dd"],
            avg_dollar_vol=s["liq"],
            momentum_z=z_score(s["mom5"], mom_mean, mom_std),
            momentum_20d_z=z_score(s["mom20"], mom20_mean, mom20_std),
            vol_z=z_score(s["vol"], vol_mean, vol_std),
            drawdown_z=z_score(s["dd"], dd_mean, dd_std),
            dollar_vol_z=z_score(s["liq"], liq_mean, liq_std),
            news_info=s["news"],
        )
        candidates.append(c)
    return candidates


def test_ordering():
    cands = build_candidates()
    sorted_cands = sorted(cands, key=lambda c: c.score, reverse=True)
    symbols = [c.symbol for c in sorted_cands]
    # Expect AAA (strong momentum) > BBB (balanced) > CCC (negative/volatile)
    assert symbols == ["AAA", "BBB", "CCC"], f"Unexpected order: {symbols}"
    print("Regression check passed. Order:", symbols)


if __name__ == "__main__":
    test_ordering()
