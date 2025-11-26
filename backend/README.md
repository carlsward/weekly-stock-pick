## Backend

- Generate news scores: `python backend/generate_news_scores.py`
  - Optional: add local extra articles in `extra_news.json` or `alt_news.json` (per-symbol list with title/summary/description/provider/pubDate/link) to broaden coverage.
  - Dry-run: `python backend/generate_news_scores.py --dry-run`
  - Sentiment cache stored in `.news_sentiment_cache.json`. Uses FinBERT + general sentiment + semantic filter (MiniLM embeddings).
- Generate picks: `python backend/generate_pick.py`
  - Dry-run ranking: `python backend/generate_pick.py --dry-run --top-n 5`
  - Uses a price cache `price_cache.json` as fallback if live downloads fail; filters out illiquid symbols (avg dollar vol threshold).
- Quick checks: `backend/run_checks.sh` (regression test + byte-compile).
- Dependencies are pinned in `backend/requirements.txt`.  
