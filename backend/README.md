## Backend overview

The backend publishes checked-in JSON files that the Android client reads directly:

- `current_pick.json`: the overall weekly release decision.
- `risk_picks.json`: the overall decision plus low/medium/high risk-profile decisions.
- `history.json`: one history entry per market week.
- `sector_scores.json`: a weekly world-news overlay built from broad market news over the last 3 days, with both sector-level and symbol-level impacts.
- `thesis_monitor.json`: a lighter daily monitor for the currently active weekly pick.
- `track_record.json`: release performance and benchmark-relative track record built from history.
- `monthly_pick.json`: the separate monthly conviction pick, refreshed on the first calendar day of each month.
- `monthly_history.json`: one history entry per monthly rebalance.

## News provider

- News generation uses Marketaux for article discovery.
- The scheduled workflow expects a `MARKETAUX_API_TOKEN` GitHub Actions secret.
- Company-specific news scoring is now a separate Marketaux-based layer that stays focused on direct company coverage, entity sentiment, recency, source quality, and story concentration.
- The backend defaults to `MARKETAUX_NEWS_LIMIT=3` locally, but the scheduled workflow overrides that to deeper weekly coverage for the company-news layer.
- For a real commercial release, the next Marketaux upgrade should still be a paid plan so each symbol can use deeper article coverage without being squeezed by provider limits.

## Price data

- Price history for the pick model now uses Stooq only.
- Yahoo Finance is no longer used anywhere in the backend pipeline.

## World-news overlay

- Every active symbol in `universe.csv` must include a `sector`.
- `generate_sector_scores.py` fetches broad financial and world news from the last 3 days from Marketaux, GDELT DOC 2.0, and Alpha Vantage NEWS_SENTIMENT, then asks OpenAI to extract causal market-impact events: event type, transmission channel, affected inputs, likely winning sectors, and likely hurt sectors/symbols.
- The workflow expects an `OPENAI_API_KEY` GitHub Actions secret for that world-news overlay. `ALPHA_VANTAGE_API_KEY` is optional but recommended for the extra Alpha Vantage feed.
- The pick model now keeps three layers separate:
  - `news_scores.json` is the company-specific Marketaux grading layer
  - `sector_scores.json` carries sector-level broad-market rotation and macro-event signals
  - `sector_scores.json` also carries symbol-level world-news impacts for tracked names when GPT sees a clear company-level transmission mechanism

## Contract guarantees

- Market weeks are anchored to `America/New_York`.
- A release is only published when both score and confidence thresholds are met.
- If a profile does not clear the thresholds, the contract returns `status = "no_pick"`.
- The daily thesis monitor only re-checks the active weekly pick. It does not rerank the whole universe.
- The monthly pick is generated separately from the weekly release and uses a 20-trading-day horizon, but the rebalance date is anchored to the first calendar day of the month.
- Freshness metadata is included in every dashboard payload:
  - `generated_at`
  - `data_as_of`
  - `expected_next_refresh_at`
  - `stale_after`

## Local checks

From the repo root:

```bash
python3 -m unittest discover -s backend/tests -t .
```

To generate fresh news locally:

```bash
export MARKETAUX_API_TOKEN=your_token_here
python3 backend/generate_news_scores.py
```

To generate the sector overlay locally:

```bash
export MARKETAUX_API_TOKEN=your_token_here
export OPENAI_API_KEY=your_openai_key_here
export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
python3 backend/generate_sector_scores.py
```

To generate the daily thesis monitor locally:

```bash
export MARKETAUX_API_TOKEN=your_token_here
export OPENAI_API_KEY=your_openai_key_here
export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
python3 backend/generate_thesis_monitor.py
```

To generate the track record locally:

```bash
python3 backend/generate_track_record.py
```

To generate the monthly pick locally:

```bash
python3 backend/generate_monthly_pick.py
```

From `backend/`:

```bash
python3 validate_outputs.py
```

## Scheduled generation

 GitHub Actions runs the main generation workflow once per week, generates company-specific news scores, generates the 3-day world-news overlay, builds the picks, refreshes `thesis_monitor.json` and `track_record.json`, validates the contracts, and then commits the refreshed JSON payloads.

For a one-off test run in GitHub Actions:

- Add `MARKETAUX_API_TOKEN` and `OPENAI_API_KEY` as repo secrets. Add `ALPHA_VANTAGE_API_KEY` too if you want the extra Alpha Vantage world-news feed.
- Open the `Generate weekly stock pick` workflow and use `Run workflow`.
- You can override the GPT model and article limits from the manual form.
- The run now uploads `current_pick.json`, `risk_picks.json`, `history.json`, `news_scores.json`, `sector_scores.json`, `thesis_monitor.json`, and `track_record.json` as workflow artifacts even if you choose not to commit them.

For the monthly release:

- Use `Generate monthly stock pick`.
- The scheduled job runs on the first calendar day of each month.
- It uploads `monthly_pick.json` and `monthly_history.json` as workflow artifacts, and can also commit them back to the repo.

For the cheap daily monitor:

- Use `Generate daily thesis monitor`.
- That workflow only refreshes `thesis_monitor.json` for the currently active overall pick.
- It still uses fresh company news plus the world-news GPT overlay, but it does not rerank the full universe.

For a faster smoke test in GitHub Actions:

- Use `Generate weekly stock pick smoke test`.
- That workflow runs the same backend pipeline against `universe_test.csv`, which currently contains 3 symbols.
- It also uses bundled Marketaux-style fixture articles so the world-news GPT layer can still run even when the live Marketaux quota is exhausted.
- It also generates `monthly_pick.json` against the same 3-symbol smoke universe.
- It validates and uploads artifacts, but it does not commit the smoke-test outputs back to the repository.
