## Backend

- Generate news scores: `python backend/generate_news_scores.py`
  - Optional: add local extra articles in `extra_news.json` (per-symbol list with title/summary/description/provider/pubDate/link) to broaden coverage.
  - Dry-run: `python backend/generate_news_scores.py --dry-run`
  - Sentiment cache stored in `.news_sentiment_cache.json`.
- Generate picks: `python backend/generate_pick.py`
  - Dry-run ranking: `python backend/generate_pick.py --dry-run --top-n 5`
- Quick checks: `backend/run_checks.sh` (regression test + byte-compile).
- Dependencies are pinned in `backend/requirements.txt`.  
