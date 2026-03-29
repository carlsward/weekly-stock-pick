import json
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

    (base_dir / "thesis_monitor.json").write_text(
        """
        {
          "schema_version": 2,
          "model_version": "v3.3",
          "generated_at": "2026-03-18T18:00:00Z",
          "data_as_of": "2026-03-18",
          "expected_next_refresh_at": "2026-03-19T18:00:00Z",
          "stale_after": "2026-03-20T06:00:00Z",
          "market_context": {
            "timezone": "America/New_York",
            "week_id": "2026-W12",
            "week_label": "Mar 16 - 20, 2026",
            "week_start": "2026-03-16",
            "week_end": "2026-03-20"
          },
          "source_dashboard_generated_at": "2026-03-16T12:00:00Z",
          "selection": {
            "status": "picked",
            "status_reason": "Live thesis monitor refreshed the active weekly pick.",
            "threshold_score": 0.1,
            "threshold_confidence": 0.55
          },
          "active_pick": {
            "symbol": "MSFT",
            "company_name": "Microsoft Corporation",
            "sector": "technology",
            "risk": "low",
            "model_score": 0.19,
            "confidence_score": 0.77,
            "confidence_label": "high",
            "price_as_of": "2026-03-18",
            "news_as_of": "2026-03-18",
            "macro_as_of": "2026-03-18",
            "sector_as_of": "2026-03-18",
            "article_count": 3,
            "metrics": {
              "momentum_5d": 0.05,
              "daily_volatility": 0.01,
              "news_sentiment": 0.69,
              "raw_news_sentiment": 0.28,
              "macro_sentiment": 0.6,
              "macro_confidence": 0.65,
              "sector_sentiment": 0.61,
              "sector_confidence": 0.7,
              "market_relative_5d": 0.02,
              "market_relative_20d": 0.04,
              "sector_relative_5d": 0.01,
              "sector_relative_20d": 0.03
            },
            "score_breakdown": {
              "momentum": 0.21,
              "market_relative_strength": 0.02,
              "sector_relative_strength": 0.01,
              "volatility_penalty": -0.06,
              "news_adjustment": 0.03,
              "macro_adjustment": 0.02,
              "sector_adjustment": 0.01,
              "total": 0.19
            },
            "reasons": ["Momentum stayed constructive."],
            "macro_evidence": [],
            "thesis_monitor": {
              "status": "healthy",
              "headline": "Support is intact",
              "summary": "Momentum, risk, and evidence are still aligned with the release thesis.",
              "alerts": [],
              "signals": [
                {
                  "label": "Momentum",
                  "state": "positive",
                  "value": "+5.0% 5D • +9.0% 20D",
                  "detail": "Price action is still supporting the release thesis."
                }
              ]
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    (base_dir / "track_record.json").write_text(
        """
        {
          "schema_version": 2,
          "model_version": "v3.3",
          "generated_at": "2026-03-16T12:00:00Z",
          "data_as_of": "2026-03-20",
          "expected_next_refresh_at": "2026-03-23T12:00:00Z",
          "stale_after": "2026-03-24T00:00:00Z",
          "market_context": {
            "timezone": "America/New_York",
            "week_id": "2026-W12",
            "week_label": "Mar 16 - 20, 2026",
            "week_start": "2026-03-16",
            "week_end": "2026-03-20"
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
          "summary": {
            "total_weeks": 1,
            "total_picks": 1,
            "no_pick_weeks": 0,
            "closed_picks": 1,
            "open_picks": 0
          },
          "risk_breakdown": {
            "low": {
              "pick_count": 1,
              "closed_pick_count": 1
            },
            "medium": {
              "pick_count": 0,
              "closed_pick_count": 0
            },
            "high": {
              "pick_count": 0,
              "closed_pick_count": 0
            }
          },
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
              "sector": "technology",
              "risk": "low",
              "model_score": 0.2,
              "confidence_score": 0.82,
              "confidence_label": "high",
              "data_as_of": "2026-03-14",
              "realized_5d_return": 0.034,
              "realized_5d_excess_return": 0.012,
              "realized_5d_sector_return": 0.028,
              "realized_5d_sector_excess_return": 0.006,
              "outcome": "win"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    (base_dir / "monthly_pick.json").write_text(
        """
        {
          "schema_version": 2,
          "model_version": "v3.3-monthly",
          "generated_at": "2026-04-01T12:00:00Z",
          "data_as_of": "2026-04-01",
          "expected_next_refresh_at": "2026-05-01T12:00:00Z",
          "stale_after": "2026-05-04T12:00:00Z",
          "period_context": {
            "timezone": "America/New_York",
            "month_id": "2026-04",
            "month_label": "April 2026",
            "month_start": "2026-04-01",
            "month_end": "2026-04-30",
            "rebalance_date": "2026-04-01",
            "horizon_trading_days": 20
          },
          "generation_summary": {
            "universe_size": 2,
            "evaluated_candidates": 2,
            "skipped_symbols": 0,
            "skipped_details": []
          },
          "selection_thresholds": {
            "overall_score": 0.12,
            "minimum_confidence": 0.60
          },
          "selection": {
            "status": "picked",
            "status_reason": "MSFT cleared the monthly release thresholds.",
            "threshold_score": 0.12,
            "threshold_confidence": 0.60,
            "pick": {
              "symbol": "MSFT",
              "company_name": "Microsoft Corporation",
              "sector": "technology",
              "risk": "low",
              "model_score": 0.24,
              "confidence_score": 0.81,
              "confidence_label": "high",
              "price_as_of": "2026-04-01",
              "news_as_of": "2026-04-01",
              "macro_as_of": "2026-04-01",
              "sector_as_of": "2026-04-01",
              "article_count": 5,
              "metrics": {
                "momentum_20d": 0.12,
                "momentum_60d": 0.26,
                "daily_volatility": 0.014,
                "market_relative_20d": 0.04,
                "market_relative_60d": 0.07,
                "sector_relative_20d": 0.03,
                "sector_relative_60d": 0.05,
                "news_sentiment": 0.63,
                "news_confidence": 0.72,
                "macro_sentiment": 0.61,
                "macro_confidence": 0.69,
                "sector_sentiment": 0.58,
                "sector_confidence": 0.66
              },
              "score_breakdown": {
                "trend_strength": 0.14,
                "relative_strength": 0.05,
                "participation": 0.02,
                "risk_control": -0.03,
                "technical_total": 0.18,
                "news_adjustment": 0.03,
                "macro_adjustment": 0.02,
                "sector_adjustment": 0.01,
                "signal_alignment": 0.0,
                "total": 0.24
              },
              "reasons": ["Trend and macro evidence aligned."],
              "news_evidence": [],
              "macro_evidence": []
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    (base_dir / "monthly_history.json").write_text(
        """
        {
          "schema_version": 2,
          "model_version": "v3.3-monthly",
          "generated_at": "2026-04-01T12:00:00Z",
          "entries": [
            {
              "month_id": "2026-04",
              "month_start": "2026-04-01",
              "month_end": "2026-04-30",
              "rebalance_date": "2026-04-01",
              "month_label": "April 2026",
              "logged_at": "2026-04-01T12:00:00Z",
              "status": "picked",
              "status_reason": "MSFT cleared the monthly release thresholds.",
              "symbol": "MSFT",
              "company_name": "Microsoft Corporation",
              "sector": "technology",
              "risk": "low",
              "model_score": 0.24,
              "confidence_score": 0.81,
              "confidence_label": "high",
              "data_as_of": "2026-04-01",
              "model_version": "v3.3-monthly",
              "realized_20d_return": 0.071,
              "realized_20d_excess_return": 0.018
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    data_quality = {
        "status": "healthy",
        "degraded_reason": None,
        "reasons": [],
        "provider_status": {},
    }
    for filename in (
        "current_pick.json",
        "risk_picks.json",
        "history.json",
        "news_scores.json",
        "sector_scores.json",
        "thesis_monitor.json",
        "track_record.json",
        "monthly_pick.json",
        "monthly_history.json",
    ):
        path = base_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["data_quality"] = data_quality
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ValidateOutputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(self.id().replace(".", "_"))
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for filename in (
            "current_pick.json",
            "risk_picks.json",
            "history.json",
            "news_scores.json",
            "sector_scores.json",
            "thesis_monitor.json",
            "track_record.json",
            "monthly_pick.json",
            "monthly_history.json",
            "universe.csv",
        ):
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
