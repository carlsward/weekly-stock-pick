import unittest
from pathlib import Path

from backend.validate_outputs import validate_repository_outputs


def write_payloads(base_dir: Path) -> None:
    (base_dir / "universe.csv").write_text(
        "symbol,company_name,sector,active\n"
        "MSFT,Microsoft Corporation,technology,1\n"
        "XOM,Exxon Mobil Corporation,energy,1\n",
        encoding="utf-8",
    )

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
              "sector": "technology",
              "risk": "low",
              "model_score": 0.2,
              "confidence_score": 0.82,
              "confidence_label": "high",
              "price_as_of": "2026-03-14",
              "news_as_of": "2026-03-14",
              "macro_as_of": "2026-03-14",
              "sector_as_of": "2026-03-14",
              "article_count": 4,
              "metrics": {
                "momentum_5d": 0.04,
                "daily_volatility": 0.01,
                "news_sentiment": 0.72,
                "raw_news_sentiment": 0.3,
                "macro_sentiment": 0.59,
                "macro_confidence": 0.62,
                "sector_sentiment": 0.64,
                "sector_confidence": 0.71,
                "market_relative_5d": 0.02,
                "market_relative_20d": 0.05,
                "sector_relative_5d": 0.01,
                "sector_relative_20d": 0.03
              },
              "score_breakdown": {
                "momentum": 0.25,
                "market_relative_strength": 0.02,
                "sector_relative_strength": 0.01,
                "volatility_penalty": -0.08,
                "news_adjustment": 0.03,
                "macro_adjustment": 0.02,
                "sector_adjustment": 0.01,
                "total": 0.2
              },
              "reasons": ["a"],
              "macro_evidence": []
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
              "sector": "technology",
              "risk": "low",
              "model_score": 0.2,
              "confidence_score": 0.82,
              "confidence_label": "high",
              "price_as_of": "2026-03-14",
              "news_as_of": "2026-03-14",
              "macro_as_of": "2026-03-14",
              "sector_as_of": "2026-03-14",
              "article_count": 4,
              "metrics": {
                "momentum_5d": 0.04,
                "daily_volatility": 0.01,
                "news_sentiment": 0.72,
                "raw_news_sentiment": 0.3,
                "macro_sentiment": 0.59,
                "macro_confidence": 0.62,
                "sector_sentiment": 0.64,
                "sector_confidence": 0.71,
                "market_relative_5d": 0.02,
                "market_relative_20d": 0.05,
                "sector_relative_5d": 0.01,
                "sector_relative_20d": 0.03
              },
              "score_breakdown": {
                "momentum": 0.25,
                "market_relative_strength": 0.02,
                "sector_relative_strength": 0.01,
                "volatility_penalty": -0.08,
                "news_adjustment": 0.03,
                "macro_adjustment": 0.02,
                "sector_adjustment": 0.01,
                "total": 0.2
              },
              "reasons": ["a"],
              "macro_evidence": []
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
                "sector": "technology",
                "risk": "low",
                "model_score": 0.2,
                "confidence_score": 0.82,
                "confidence_label": "high",
                "price_as_of": "2026-03-14",
                "news_as_of": "2026-03-14",
                "macro_as_of": "2026-03-14",
                "sector_as_of": "2026-03-14",
                "article_count": 4,
                "metrics": {
                  "momentum_5d": 0.04,
                  "daily_volatility": 0.01,
                  "news_sentiment": 0.72,
                  "raw_news_sentiment": 0.3,
                  "macro_sentiment": 0.59,
                  "macro_confidence": 0.62,
                  "sector_sentiment": 0.64,
                  "sector_confidence": 0.71,
                  "market_relative_5d": 0.02,
                  "market_relative_20d": 0.05,
                  "sector_relative_5d": 0.01,
                  "sector_relative_20d": 0.03
                },
                "score_breakdown": {
                  "momentum": 0.25,
                  "market_relative_strength": 0.02,
                  "sector_relative_strength": 0.01,
                  "volatility_penalty": -0.08,
                  "news_adjustment": 0.03,
                  "macro_adjustment": 0.02,
                  "sector_adjustment": 0.01,
                  "total": 0.2
                },
                "reasons": ["a"],
                "macro_evidence": []
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

    (base_dir / "news_scores.json").write_text(
        """
        {
          "MSFT": {
            "news_score": 0.72,
            "news_confidence": 0.81,
            "raw_sentiment": 0.33,
            "calibrated_sentiment": 0.24,
            "article_count": 4,
            "effective_article_count": 2.8,
            "source_count": 3,
            "average_relevance": 0.82,
            "average_recency_weight": 0.91,
            "average_source_quality": 0.94,
            "provider_sentiment_coverage": 0.8,
            "dominant_weight_share": 0.42,
            "dominant_signal": "bullish",
            "news_reasons": ["Good evidence."],
            "top_articles": [],
            "last_updated": "2026-03-14",
            "analysis_method": "gpt_company_review",
            "llm_model": "gpt-5-mini"
          },
          "XOM": {
            "news_score": 0.46,
            "news_confidence": 0.63,
            "raw_sentiment": -0.08,
            "calibrated_sentiment": -0.05,
            "article_count": 2,
            "effective_article_count": 1.2,
            "source_count": 2,
            "average_relevance": 0.71,
            "average_recency_weight": 0.88,
            "average_source_quality": 0.9,
            "provider_sentiment_coverage": 0.5,
            "dominant_weight_share": 0.58,
            "dominant_signal": "neutral",
            "news_reasons": ["Mixed evidence."],
            "top_articles": [],
            "last_updated": "2026-03-14",
            "analysis_method": "heuristic_fallback",
            "llm_model": null
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    (base_dir / "sector_scores.json").write_text(
        """
        {
          "generated_at": "2026-03-16T12:00:00Z",
          "last_updated": "2026-03-16",
          "lookback_days": 3,
          "article_count": 6,
          "source_count": 4,
          "llm_model": "gpt-5-mini",
          "summary": "Recent broad market news favored technology over energy.",
          "sector_scores": {
            "energy": {
              "sector": "energy",
              "display_name": "Energy",
              "score": 0.42,
              "confidence": 0.71,
              "direction": "bearish",
              "last_updated": "2026-03-16",
              "reasons": ["Oil demand expectations weakened."],
              "supporting_articles": []
            },
            "technology": {
              "sector": "technology",
              "display_name": "Technology",
              "score": 0.64,
              "confidence": 0.73,
              "direction": "bullish",
              "last_updated": "2026-03-16",
              "reasons": ["AI infrastructure demand improved."],
              "supporting_articles": []
            }
          },
          "symbol_scores": {
            "MSFT": {
              "symbol": "MSFT",
              "company_name": "Microsoft Corporation",
              "sector": "technology",
              "score": 0.68,
              "confidence": 0.74,
              "direction": "bullish",
              "last_updated": "2026-03-16",
              "reasons": ["Enterprise AI demand improved."],
              "supporting_articles": []
            },
            "XOM": {
              "symbol": "XOM",
              "company_name": "Exxon Mobil Corporation",
              "sector": "energy",
              "score": 0.44,
              "confidence": 0.66,
              "direction": "bearish",
              "last_updated": "2026-03-16",
              "reasons": ["Oil demand expectations weakened for near-term producers."],
              "supporting_articles": []
            }
          },
          "events": []
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
        for filename in ("current_pick.json", "risk_picks.json", "history.json", "news_scores.json", "sector_scores.json", "universe.csv"):
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
