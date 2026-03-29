import unittest
from datetime import datetime, timezone
from datetime import date
from unittest.mock import patch

from backend.generate_pick import GenerationStats
from backend.generate_monthly_pick import (
    MONTHLY_BLOCK_BASE_PRIORS,
    MONTHLY_MIN_CONFIDENCE_THRESHOLD,
    MONTHLY_SELECTION_THRESHOLD,
    MarketMonth,
    MonthlyCandidate,
    build_monthly_calibration,
    compute_monthly_score_breakdown,
    build_market_month,
    build_monthly_macro_snapshot,
    build_monthly_news_snapshot,
    enrich_monthly_history_realized_returns,
    main,
    select_monthly_candidate,
)
from backend.generate_pick import ModelCalibration


def build_monthly_candidate(
    symbol: str = "MSFT",
    total_score: float = 0.24,
    confidence_score: float = 0.78,
) -> MonthlyCandidate:
    return MonthlyCandidate(
        symbol=symbol,
        company_name=f"{symbol} Corporation",
        sector="technology",
        reasons=["Momentum and world-news stayed aligned."],
        risk_level="low",
        total_score=total_score,
        confidence_score=confidence_score,
        confidence_label="high" if confidence_score >= 0.75 else "medium",
        price_as_of="2026-04-01",
        news_as_of="2026-04-01",
        macro_as_of="2026-04-01",
        sector_as_of="2026-04-01",
        article_count=4,
        effective_article_count=2.4,
        source_count=3,
        average_relevance=0.82,
        momentum_20d=0.12,
        momentum_60d=0.26,
        volatility_20d=0.014,
        downside_volatility_20d=0.01,
        max_drawdown_60d=0.08,
        trend_gap_50d=0.05,
        positive_day_ratio_20d=0.65,
        volume_trend_10d=0.10,
        market_relative_20d=0.04,
        market_relative_60d=0.06,
        sector_relative_20d=0.03,
        sector_relative_60d=0.04,
        news_score=0.63,
        news_confidence=0.71,
        macro_score=0.61,
        macro_confidence=0.68,
        sector_score=0.59,
        sector_confidence=0.66,
        raw_sentiment=0.16,
        calibrated_sentiment=0.12,
        dominant_signal="bullish",
        score_breakdown={
            "trend_strength": 0.14,
            "relative_strength": 0.06,
            "participation": 0.02,
            "risk_control": -0.03,
            "technical_total": 0.19,
            "weighted_news": 0.03,
            "weighted_macro": 0.01,
            "weighted_sector": 0.01,
            "weighted_signal_alignment": 0.0,
            "technical_training_ic": 0.08,
            "technical_training_rows": 180,
            "block_weight_news": 0.14,
            "block_weight_macro": 0.12,
            "block_weight_sector": 0.09,
            "macro_dedup_penalty": 0.90,
            "sector_dedup_penalty": 0.92,
            "news_macro_overlap": 0.1,
            "news_sector_overlap": 0.05,
            "macro_sector_overlap": 0.08,
            "total": total_score,
        },
        news_evidence=[],
        macro_evidence=[],
    )


class GenerateMonthlyPickTests(unittest.TestCase):
    def test_build_market_month_anchors_to_calendar_first(self) -> None:
        market_month = build_market_month()

        self.assertEqual(1, market_month.rebalance_date.day)
        self.assertEqual(1, market_month.next_rebalance_date.day)

    def test_select_monthly_candidate_picks_best_qualified_name(self) -> None:
        winner = build_monthly_candidate("MSFT", total_score=0.25, confidence_score=0.81)
        runner_up = build_monthly_candidate("NVDA", total_score=0.21, confidence_score=0.74)

        selection = select_monthly_candidate([runner_up, winner])

        self.assertEqual("picked", selection.status)
        self.assertEqual("MSFT", selection.pick.symbol)
        self.assertGreaterEqual(selection.pick.total_score, MONTHLY_SELECTION_THRESHOLD)
        self.assertGreaterEqual(selection.pick.confidence_score, MONTHLY_MIN_CONFIDENCE_THRESHOLD)

    def test_select_monthly_candidate_returns_no_pick_when_no_candidates_exist(self) -> None:
        selection = select_monthly_candidate([])

        self.assertEqual("no_pick", selection.status)
        self.assertIsNone(selection.pick)
        self.assertIsNone(selection.best_candidate)

    def test_select_monthly_candidate_uses_explicit_price_failure_policy(self) -> None:
        selection = select_monthly_candidate(
            [],
            GenerationStats(
                universe_size=2,
                evaluated_candidates=0,
                skipped_symbols=2,
                skipped_details=[
                    {"symbol": "MSFT", "reason": "Unable to fetch usable price frame for MSFT after retries"},
                    {"symbol": "CVX", "reason": "Unable to fetch usable price frame for CVX after retries"},
                ],
                price_provider_failures=2,
            ),
        )

        self.assertEqual("no_pick", selection.status)
        self.assertIn("all live price providers failed", selection.status_reason.lower())
        self.assertIn("fabricated prices", selection.status_reason.lower())

    @patch("backend.generate_monthly_pick.now_utc")
    def test_build_monthly_news_snapshot_keeps_recent_week_old_news_active(self, now_utc_mock) -> None:
        now_utc_mock.return_value = datetime(2026, 4, 10, 12, tzinfo=timezone.utc)

        snapshot = build_monthly_news_snapshot(
            {
                "news_score": 0.68,
                "news_confidence": 0.74,
                "raw_sentiment": 0.18,
                "calibrated_sentiment": 0.14,
                "article_count": 3,
                "effective_article_count": 2.1,
                "source_count": 2,
                "average_relevance": 0.79,
                "dominant_signal": "bullish",
                "news_reasons": ["Durable contract win."],
                "last_updated": "2026-04-05",
                "top_articles": [],
            }
        )

        self.assertAlmostEqual(0.68, snapshot.news_score, places=4)
        self.assertAlmostEqual(0.74, snapshot.news_confidence, places=4)
        self.assertEqual("bullish", snapshot.dominant_signal)

    @patch("backend.generate_monthly_pick.now_utc")
    def test_build_monthly_macro_snapshot_downweights_short_horizon_events(self, now_utc_mock) -> None:
        now_utc_mock.return_value = datetime(2026, 4, 10, 12, tzinfo=timezone.utc)
        base_payload = {
            "last_updated": "2026-04-09",
            "symbol_scores": {
                "MSFT": {
                    "symbol": "MSFT",
                    "company_name": "Microsoft Corporation",
                    "sector": "technology",
                    "score": 0.72,
                    "confidence": 0.8,
                    "direction": "bullish",
                    "reasons": ["Broad catalyst helped software demand."],
                    "last_updated": "2026-04-09",
                    "supporting_articles": [
                        {
                            "title": "Catalyst",
                            "published_at": "2026-04-09",
                            "weight": 1.0,
                        }
                    ],
                }
            },
        }

        short_horizon_payload = {
            **base_payload,
            "symbol_scores": {
                "MSFT": {
                    **base_payload["symbol_scores"]["MSFT"],
                    "supporting_articles": [
                        {
                            "title": "Catalyst",
                            "published_at": "2026-04-09",
                            "horizon": "1-3d",
                            "weight": 1.0,
                        }
                    ],
                }
            },
        }
        durable_payload = {
            **base_payload,
            "symbol_scores": {
                "MSFT": {
                    **base_payload["symbol_scores"]["MSFT"],
                    "supporting_articles": [
                        {
                            "title": "Catalyst",
                            "published_at": "2026-04-09",
                            "horizon": "2-4w",
                            "weight": 1.0,
                        }
                    ],
                }
            },
        }

        short_snapshot = build_monthly_macro_snapshot("MSFT", "Microsoft Corporation", "technology", short_horizon_payload)
        durable_snapshot = build_monthly_macro_snapshot("MSFT", "Microsoft Corporation", "technology", durable_payload)

        self.assertLess(short_snapshot.macro_confidence, durable_snapshot.macro_confidence)
        self.assertLess(abs(short_snapshot.macro_score - 0.5), abs(durable_snapshot.macro_score - 0.5))

    @patch("backend.generate_monthly_pick.build_monthly_calibration_training_rows")
    def test_build_monthly_calibration_uses_realized_monthly_history_for_block_weights(
        self,
        training_rows_mock,
    ) -> None:
        training_rows_mock.return_value = []
        history_entries = [
            {
                "diagnostics": {
                    "news_signal_input": 0.12,
                    "macro_signal_input": 0.82,
                    "sector_signal_input": 0.08,
                },
                "realized_20d_excess_return": 0.12,
            },
            {
                "diagnostics": {
                    "news_signal_input": 0.08,
                    "macro_signal_input": 0.74,
                    "sector_signal_input": 0.10,
                },
                "realized_20d_excess_return": 0.10,
            },
            {
                "diagnostics": {
                    "news_signal_input": 0.05,
                    "macro_signal_input": 0.60,
                    "sector_signal_input": 0.05,
                },
                "realized_20d_excess_return": 0.07,
            },
            {
                "diagnostics": {
                    "news_signal_input": 0.03,
                    "macro_signal_input": 0.18,
                    "sector_signal_input": 0.04,
                },
                "realized_20d_excess_return": 0.02,
            },
            {
                "diagnostics": {
                    "news_signal_input": 0.02,
                    "macro_signal_input": -0.10,
                    "sector_signal_input": 0.03,
                },
                "realized_20d_excess_return": -0.03,
            },
            {
                "diagnostics": {
                    "news_signal_input": 0.01,
                    "macro_signal_input": -0.24,
                    "sector_signal_input": 0.02,
                },
                "realized_20d_excess_return": -0.06,
            },
        ]

        calibration = build_monthly_calibration([], {}, history_entries)

        self.assertEqual(len(history_entries), calibration.block_row_count)
        self.assertNotEqual(MONTHLY_BLOCK_BASE_PRIORS, calibration.block_weights)
        self.assertGreater(calibration.block_weights["macro"], MONTHLY_BLOCK_BASE_PRIORS["macro"])
        self.assertGreater(calibration.block_weights["macro"], calibration.block_weights["sector"])
        self.assertIn("monthly_history_realized_returns", calibration.source)

    def test_compute_monthly_score_breakdown_uses_monthly_feature_names(self) -> None:
        calibration = ModelCalibration(
            technical_weights={
                "monthly_momentum_short": 0.25,
                "monthly_momentum_medium": 0.25,
                "monthly_trend_gap": 0.10,
                "monthly_positive_ratio": 0.10,
                "monthly_volume_confirmation": 0.05,
                "monthly_market_relative": 0.10,
                "monthly_sector_relative": 0.05,
                "monthly_inverse_volatility": 0.04,
                "monthly_inverse_downside": 0.03,
                "monthly_inverse_drawdown": 0.03,
            },
            block_weights={"news": 0.14, "macro": 0.12, "sector": 0.09},
            technical_scale=0.34,
            training_row_count=0,
            training_ic=0.0,
            block_row_count=0,
            source="test",
        )

        breakdown = compute_monthly_score_breakdown(
            technical_features={
                "monthly_momentum_short": 0.5,
                "monthly_momentum_medium": 0.4,
                "monthly_trend_gap": 0.3,
                "monthly_positive_ratio": 0.2,
                "monthly_volume_confirmation": 0.1,
                "monthly_market_relative": 0.2,
                "monthly_sector_relative": 0.1,
                "monthly_inverse_volatility": -0.2,
                "monthly_inverse_downside": -0.1,
                "monthly_inverse_drawdown": -0.1,
            },
            news_score=0.6,
            news_confidence=0.7,
            macro_score=0.58,
            macro_confidence=0.66,
            sector_score=0.55,
            sector_confidence=0.61,
            calibration=calibration,
            layer_penalties={
                "macro_penalty": 0.9,
                "sector_penalty": 0.92,
                "news_macro_overlap": 0.1,
                "news_sector_overlap": 0.05,
                "macro_sector_overlap": 0.07,
            },
        )

        self.assertIn("technical_total", breakdown)
        self.assertIn("weighted_news", breakdown)
        self.assertGreater(breakdown["technical_total"], 0.0)

    @patch("backend.generate_monthly_pick.realized_forward_return_monthly")
    @patch("backend.generate_monthly_pick.monthly_history_realized_enrichment_enabled")
    def test_enrich_monthly_history_anchors_returns_to_rebalance_date(
        self,
        enrichment_enabled_mock,
        realized_forward_return_mock,
    ) -> None:
        enrichment_enabled_mock.return_value = True
        realized_forward_return_mock.return_value = (0.072, 0.018)

        entries = enrich_monthly_history_realized_returns(
            [
                {
                    "month_id": "2026-04",
                    "month_start": "2026-04-01",
                    "month_end": "2026-04-30",
                    "rebalance_date": "2026-04-01",
                    "status": "picked",
                    "symbol": "MSFT",
                }
            ]
        )

        self.assertEqual("2026-04-01", realized_forward_return_mock.call_args.args[1])
        self.assertEqual(0.072, entries[0]["realized_20d_return"])
        self.assertEqual(0.018, entries[0]["realized_20d_excess_return"])

    @patch("backend.generate_monthly_pick.write_json")
    @patch("backend.generate_monthly_pick.update_monthly_history")
    @patch("backend.generate_monthly_pick.get_monthly_candidates")
    @patch("backend.generate_monthly_pick.build_market_month")
    def test_main_writes_monthly_pick_payload(
        self,
        build_market_month_mock,
        get_monthly_candidates_mock,
        update_monthly_history_mock,
        write_json_mock,
    ) -> None:
        build_market_month_mock.return_value = MarketMonth(
            month_id="2026-04",
            month_label="April 2026",
            month_start=date(2026, 4, 1),
            month_end=date(2026, 4, 30),
            rebalance_date=date(2026, 4, 1),
            next_rebalance_date=date(2026, 5, 1),
            horizon_trading_days=20,
        )
        candidate = build_monthly_candidate("MSFT", total_score=0.26, confidence_score=0.82)
        get_monthly_candidates_mock.return_value = (
            [candidate],
            GenerationStats(
                universe_size=3,
                evaluated_candidates=1,
                skipped_symbols=2,
                skipped_details=[{"symbol": "JPM", "reason": "price"}],
            ),
        )

        main()

        payload = write_json_mock.call_args.args[1]
        self.assertEqual("2026-04", payload["period_context"]["month_id"])
        self.assertEqual("picked", payload["selection"]["status"])
        self.assertEqual("MSFT", payload["selection"]["pick"]["symbol"])
        update_monthly_history_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
