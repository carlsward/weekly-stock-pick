## Backend overview

The backend publishes three checked-in JSON files that the Android client reads directly:

- `current_pick.json`: the overall weekly release decision.
- `risk_picks.json`: the overall decision plus low/medium/high risk-profile decisions.
- `history.json`: one history entry per market week.

## News provider

- News generation now uses Marketaux instead of Yahoo Finance for article discovery.
- The scheduled workflow expects a `MARKETAUX_API_TOKEN` GitHub Actions secret.
- The backend defaults to `MARKETAUX_NEWS_LIMIT=3`, which fits the Marketaux free plan and is enough for the current once-per-week, ~50-symbol universe.
- For a real commercial release, the free plan is acceptable for weekly batch generation, but the next upgrade should be a paid Marketaux plan so each symbol can use deeper article coverage and the app can support growth without squeezing into a 3-article cap.

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

From `backend/`:

```bash
python3 validate_outputs.py
```

## Scheduled generation

GitHub Actions runs the generation workflow once per week and validates the generated contracts before committing them.
