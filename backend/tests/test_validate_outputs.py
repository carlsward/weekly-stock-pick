import unittest
from pathlib import Path

from backend.validate_outputs import validate_repository_outputs


def write_payloads(base_dir: Path) -> None:
    (base_dir / "current_pick.json").write_text(
        """
        {
          "schema_version": 2,
          "model_version": "v2.0",
          "generated_at": "2026-03-16T12:00:00Z",
          "data_as_of": "2026-03-14",
          "expected_next_refresh_at": "2026-03-23T12:00:00Z",
          "stale_after": "2026-03-24T00:00:00Z",
          "market_context": {
            "timezone": "America/New_York",
            "week_id": "2026-W12",
            "week_label": "Mar 16 - 20, 2026",
            "week_start": "2026-03-16",
            "week_end": "2026-03-20"
          },
          "generation_summary": {
            "universe_size": 50,
            "evaluated_candidates": 48,
            "skipped_symbols": 2,
            "skipped_details": []
          },
          "selection_thresholds": {
            "overall_score": 0.1,
            "risk_scores": {
              "low": 0.08,
              "medium": 0.1,
              "high": 0.12
            },
            "minimum_confidence": 0.55
          },
          "selection": {
            "status": "picked",
            "status_reason": "ok",
            "threshold_score": 0.1,
            "threshold_confidence": 0.55,
            "pick": {
              "symbol": "MSFT",
              "company_name": "Microsoft Corporation",
              "risk": "low",
              "model_score": 0.2,
              "confidence_score": 0.82,
              "confidence_label": "high",
              "price_as_of": "2026-03-14",
              "news_as_of": "2026-03-14",
              "article_count": 4,
              "metrics": {
                "momentum_5d": 0.04,
                "daily_volatility": 0.01,
                "news_sentiment": 0.72,
                "raw_news_sentiment": 0.3
              },
              "score_breakdown": {
                "momentum": 0.25,
                "volatility_penalty": -0.08,
                "news_adjustment": 0.03,
                "total": 0.2
              },
              "reasons": ["a"]
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    (base_dir / "risk_picks.json").write_text(
        """
        {
          "schema_version": 2,
          "model_version": "v2.0",
          "generated_at": "2026-03-16T12:00:00Z",
          "data_as_of": "2026-03-14",
          "expected_next_refresh_at": "2026-03-23T12:00:00Z",
          "stale_after": "2026-03-24T00:00:00Z",
          "market_context": {
            "timezone": "America/New_York",
            "week_id": "2026-W12",
            "week_label": "Mar 16 - 20, 2026",
            "week_start": "2026-03-16",
            "week_end": "2026-03-20"
          },
          "generation_summary": {
            "universe_size": 50,
            "evaluated_candidates": 48,
            "skipped_symbols": 2,
            "skipped_details": []
          },
          "selection_thresholds": {
            "overall_score": 0.1,
            "risk_scores": {
              "low": 0.08,
              "medium": 0.1,
              "high": 0.12
            },
            "minimum_confidence": 0.55
          },
          "overall_selection": {
            "status": "picked",
            "status_reason": "ok",
            "threshold_score": 0.1,
            "threshold_confidence": 0.55,
            "pick": {
              "symbol": "MSFT",
              "company_name": "Microsoft Corporation",
              "risk": "low",
              "model_score": 0.2,
              "confidence_score": 0.82,
              "confidence_label": "high",
              "price_as_of": "2026-03-14",
              "news_as_of": "2026-03-14",
              "article_count": 4,
              "metrics": {
                "momentum_5d": 0.04,
                "daily_volatility": 0.01,
                "news_sentiment": 0.72,
                "raw_news_sentiment": 0.3
              },
              "score_breakdown": {
                "momentum": 0.25,
                "volatility_penalty": -0.08,
                "news_adjustment": 0.03,
                "total": 0.2
              },
              "reasons": ["a"]
            }
          },
          "risk_selections": {
            "low": {
              "status": "picked",
              "status_reason": "ok",
              "threshold_score": 0.08,
              "threshold_confidence": 0.55,
              "pick": {
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
                "risk": "low",
                "model_score": 0.2,
                "confidence_score": 0.82,
                "confidence_label": "high",
                "price_as_of": "2026-03-14",
                "news_as_of": "2026-03-14",
                "article_count": 4,
                "metrics": {
                  "momentum_5d": 0.04,
                  "daily_volatility": 0.01,
                  "news_sentiment": 0.72,
                  "raw_news_sentiment": 0.3
                },
                "score_breakdown": {
                  "momentum": 0.25,
                  "volatility_penalty": -0.08,
                  "news_adjustment": 0.03,
                  "total": 0.2
                },
                "reasons": ["a"]
              }
            },
            "medium": {
              "status": "no_pick",
              "status_reason": "none",
              "threshold_score": 0.1,
              "threshold_confidence": 0.55,
              "pick": null
            },
            "high": {
              "status": "no_pick",
              "status_reason": "none",
              "threshold_score": 0.12,
              "threshold_confidence": 0.55,
              "pick": null
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    (base_dir / "history.json").write_text(
        """
        {
          "schema_version": 2,
          "model_version": "v2.0",
          "generated_at": "2026-03-16T12:00:00Z",
          "entries": [
            {
              "week_id": "2026-W12",
              "week_start": "2026-03-16",
              "week_end": "2026-03-20",
              "week_label": "Mar 16 - 20, 2026",
              "logged_at": "2026-03-16T12:00:00Z",
              "status": "picked",
              "status_reason": "ok",
              "symbol": "MSFT",
              "company_name": "Microsoft Corporation",
              "risk": "low",
              "model_score": 0.2,
              "confidence_score": 0.82,
              "confidence_label": "high",
              "data_as_of": "2026-03-14",
              "model_version": "v2.0"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )


class ValidateOutputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(self.id().replace(".", "_"))
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for filename in ("current_pick.json", "risk_picks.json", "history.json"):
            path = self.temp_dir / filename
            if path.exists():
                path.unlink()
        if self.temp_dir.exists():
            self.temp_dir.rmdir()

    def test_validate_repository_outputs_accepts_valid_contract(self) -> None:
        write_payloads(self.temp_dir)
        validate_repository_outputs(self.temp_dir)

    def test_validate_repository_outputs_rejects_duplicate_history_weeks(self) -> None:
        write_payloads(self.temp_dir)
        history_path = self.temp_dir / "history.json"
        history_path.write_text(
            """
            {
              "schema_version": 2,
              "model_version": "v2.0",
              "generated_at": "2026-03-16T12:00:00Z",
              "entries": [
                {
                  "week_id": "2026-W12",
                  "week_start": "2026-03-16",
                  "week_end": "2026-03-20",
                  "week_label": "Mar 16 - 20, 2026",
                  "logged_at": "2026-03-16T12:00:00Z",
                  "status": "picked",
                  "status_reason": "ok",
                  "symbol": "MSFT",
                  "company_name": "Microsoft Corporation",
                  "risk": "low",
                  "model_score": 0.2,
                  "confidence_score": 0.82,
                  "confidence_label": "high",
                  "data_as_of": "2026-03-14",
                  "model_version": "v2.0"
                },
                {
                  "week_id": "2026-W12",
                  "week_start": "2026-03-23",
                  "week_end": "2026-03-27",
                  "week_label": "Mar 23 - 27, 2026",
                  "logged_at": "2026-03-23T12:00:00Z",
                  "status": "no_pick",
                  "status_reason": "dup",
                  "symbol": null,
                  "company_name": null,
                  "risk": null,
                  "model_score": null,
                  "confidence_score": null,
                  "confidence_label": null,
                  "data_as_of": "2026-03-21",
                  "model_version": "v2.0"
                }
              ]
            }
            """.strip(),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            validate_repository_outputs(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
