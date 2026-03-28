## Backend overview

The backend publishes three checked-in JSON files that the Android client reads directly:

- `current_pick.json`: the overall weekly release decision.
- `risk_picks.json`: the overall decision plus low/medium/high risk-profile decisions.
- `history.json`: one history entry per market week.
- `sector_scores.json`: a weekly world-news overlay built from broad market news over the last 3 days, with both sector-level and symbol-level impacts.

## News provider

- News generation uses Marketaux for article discovery.
- The scheduled workflow expects a `MARKETAUX_API_TOKEN` GitHub Actions secret.
- Company-specific news scoring now also uses `OPENAI_API_KEY` to let GPT classify whether fetched articles are truly company-specific, broad market roundups, or weak listicles/opinions.
- The backend defaults to `MARKETAUX_NEWS_LIMIT=3` and `COMPANY_LLM_ARTICLE_LIMIT=5` locally, but the scheduled workflow now overrides that to deeper weekly coverage so GPT can filter a broader article set before scoring.
- For a real commercial release, the next Marketaux upgrade should still be a paid plan so each symbol can use deeper article coverage without being squeezed by provider limits.

## Price data

- Price history for the pick model now uses Stooq only.
- Yahoo Finance is no longer used anywhere in the backend pipeline.

## World-news overlay

- Every active symbol in `universe.csv` must include a `sector`.
- `generate_sector_scores.py` fetches broad financial and world news from the last 3 days and asks OpenAI to classify which sectors and tracked symbols are likely beneficiaries or losers.
- The workflow expects an `OPENAI_API_KEY` GitHub Actions secret for that world-news overlay.
- The pick model now keeps three layers separate:
  - `news_scores.json` is still the company-specific Marketaux grading layer, with GPT filtering out weak roundups and listicles before aggregation
  - `sector_scores.json` carries sector-level broad-market rotation signals
  - `sector_scores.json` also carries symbol-level world-news impacts for tracked names when GPT sees a clear company-level transmission mechanism

## Contract guarantees

- Market weeks are anchored to `America/New_York`.
- A release is only published when both score and confidence thresholds are met.
- If a profile does not clear the thresholds, the contract returns `status = "no_pick"`.
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
python3 backend/generate_sector_scores.py
```

From `backend/`:

```bash
python3 validate_outputs.py
```

## Scheduled generation

 GitHub Actions runs the generation workflow once per week, generates company-specific news scores, generates the 3-day world-news overlay, builds the picks, validates the contracts, and then commits the refreshed JSON payloads.

For a one-off test run in GitHub Actions:

- Add both `MARKETAUX_API_TOKEN` and `OPENAI_API_KEY` as repo secrets.
- Open the `Generate weekly stock pick` workflow and use `Run workflow`.
- You can override the GPT model and article limits from the manual form.
- The run now uploads `current_pick.json`, `risk_picks.json`, `history.json`, `news_scores.json`, and `sector_scores.json` as workflow artifacts even if you choose not to commit them.

For a faster smoke test in GitHub Actions:

- Use `Generate weekly stock pick smoke test`.
- That workflow runs the same backend pipeline against `universe_test.csv`, which currently contains 3 symbols.
- It validates and uploads artifacts, but it does not commit the smoke-test outputs back to the repository.
