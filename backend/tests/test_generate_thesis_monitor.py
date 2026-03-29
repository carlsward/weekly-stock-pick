import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from backend.generate_pick import MIN_CONFIDENCE_THRESHOLD, StockCandidate
from backend.generate_thesis_monitor import build_live_candidate, main, resolve_source_pick_sector


def build_candidate(symbol: str = "MSFT") -> StockCandidate:
    return StockCandidate(
        symbol=symbol,
        company_name="Microsoft Corporation",
        reasons=["reason"],
        risk_level="low",
        total_score=0.19,
        confidence_score=0.78,
        confidence_label="high",
        price_as_of="2026-03-18",
        news_as_of="2026-03-18",
        article_count=3,
        effective_article_count=1.8,
        source_count=2,
        average_relevance=0.82,
        momentum_5d=0.05,
        momentum_20d=0.09,
        volatility=0.01,
        downside_volatility=0.008,
        max_drawdown=0.03,
        trend_gap=0.02,
        positive_day_ratio=0.7,
        volume_trend=0.14,
        market_relative_5d=0.01,
        market_relative_20d=0.02,
        sector_relative_5d=0.01,
        sector_relative_20d=0.02,
        news_score=0.69,
        news_confidence=0.74,
        macro_score=0.58,
        macro_confidence=0.64,
        raw_sentiment=0.18,
        calibrated_sentiment=0.15,
        dominant_signal="bullish",
        score_breakdown={
            "momentum_total": 0.23,
            "weighted_short_momentum": 0.08,
            "weighted_medium_momentum": 0.08,
            "weighted_trend_quality": 0.05,
            "weighted_volume_confirmation": 0.02,
            "weighted_market_relative": 0.01,
            "weighted_sector_relative": 0.01,
            "weighted_volatility_penalty": -0.02,
            "weighted_downside_penalty": -0.02,
            "weighted_drawdown_penalty": -0.01,
            "volatility_penalty_total": -0.05,
            "technical_total": 0.18,
            "weighted_news": 0.03,
            "weighted_macro": 0.01,
            "weighted_sector": 0.01,
            "weighted_signal_alignment": 0.0,
            "total": 0.19,
        },
        news_evidence=[],
        macro_evidence=[],
        macro_as_of="2026-03-18",
        sector_as_of="2026-03-18",
        sector="technology",
        sector_score=0.61,
        sector_confidence=0.7,
        sector_direction="bullish",
        sector_reasons=["reason"],
    )


class GenerateThesisMonitorTests(unittest.TestCase):
    @patch("backend.generate_thesis_monitor.load_sector_map")
    @patch("backend.generate_thesis_monitor.resolve_universe_csv_path")
    def test_resolve_source_pick_sector_falls_back_to_universe(
        self,
        resolve_universe_csv_path_mock,
        load_sector_map_mock,
    ) -> None:
        resolve_universe_csv_path_mock.return_value = "universe.csv"
        load_sector_map_mock.return_value = {"MSFT": "technology"}

        sector = resolve_source_pick_sector(
            {
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
            }
        )

        self.assertEqual("technology", sector)

    @patch("backend.generate_thesis_monitor.build_candidate")
    @patch("backend.generate_thesis_monitor.build_live_calibration")
    @patch("backend.generate_thesis_monitor.build_sector_scores_payload")
    @patch("backend.generate_thesis_monitor.fetch_recent_global_articles")
    @patch("backend.generate_thesis_monitor.score_symbol_news")
    @patch("backend.generate_thesis_monitor.init_models")
    def test_build_live_candidate_uses_weekly_model_calibration(
        self,
        init_models_mock,
        score_symbol_news_mock,
        fetch_recent_global_articles_mock,
        build_sector_scores_payload_mock,
        build_live_calibration_mock,
        build_candidate_mock,
    ) -> None:
        init_models_mock.return_value = (None, None)
        score_symbol_news_mock.return_value = {"news_score": 0.5}
        fetch_recent_global_articles_mock.return_value = []
        build_sector_scores_payload_mock.return_value = {"symbol_scores": {}, "sector_scores": {}}
        calibration = MagicMock()
        build_live_calibration_mock.return_value = calibration
        build_candidate_mock.return_value = build_candidate()

        build_live_candidate(
            {
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
                "sector": "technology",
            }
        )

        self.assertIs(build_candidate_mock.call_args.kwargs["calibration"], calibration)

    @patch("backend.generate_thesis_monitor.write_json")
    @patch("backend.generate_thesis_monitor.load_dashboard_source")
    def test_main_writes_no_pick_payload_when_no_active_selection(self, load_dashboard_source_mock, write_json_mock) -> None:
        load_dashboard_source_mock.return_value = {
            "generated_at": "2026-03-16T12:00:00Z",
            "data_as_of": "2026-03-14",
            "market_context": {
                "timezone": "America/New_York",
                "week_id": "2026-W12",
                "week_label": "Mar 16 - 20, 2026",
                "week_start": "2026-03-16",
                "week_end": "2026-03-20",
            },
            "overall_selection": {
                "status": "no_pick",
                "status_reason": "No active pick",
                "threshold_score": 0.10,
                "threshold_confidence": MIN_CONFIDENCE_THRESHOLD,
            },
        }

        main()

        payload = write_json_mock.call_args.args[1]
        self.assertEqual("no_pick", payload["selection"]["status"])
        self.assertIsNone(payload["active_pick"])

    @patch("backend.generate_thesis_monitor.write_json")
    @patch("backend.generate_thesis_monitor.build_live_candidate")
    @patch("backend.generate_thesis_monitor.load_dashboard_source")
    def test_main_writes_live_pick_payload(self, load_dashboard_source_mock, build_live_candidate_mock, write_json_mock) -> None:
        load_dashboard_source_mock.return_value = {
            "generated_at": "2026-03-16T12:00:00Z",
            "data_as_of": "2026-03-14",
            "market_context": {
                "timezone": "America/New_York",
                "week_id": "2026-W12",
                "week_label": "Mar 16 - 20, 2026",
                "week_start": "2026-03-16",
                "week_end": "2026-03-20",
            },
            "overall_selection": {
                "status": "picked",
                "status_reason": "MSFT cleared the thresholds.",
                "threshold_score": 0.10,
                "threshold_confidence": MIN_CONFIDENCE_THRESHOLD,
                "pick": {
                    "symbol": "MSFT",
                    "company_name": "Microsoft Corporation",
                    "sector": "technology",
                },
            },
        }
        build_live_candidate_mock.return_value = build_candidate()

        main()

        payload = write_json_mock.call_args.args[1]
        self.assertEqual("picked", payload["selection"]["status"])
        self.assertEqual("MSFT", payload["active_pick"]["symbol"])
        self.assertIn("thesis_monitor", payload["active_pick"])

    @patch("backend.generate_thesis_monitor.write_json")
    @patch(
        "backend.generate_thesis_monitor.build_live_candidate",
        side_effect=RuntimeError("Unable to fetch usable price frame for MSFT after retries"),
    )
    @patch("backend.generate_thesis_monitor.load_dashboard_source")
    def test_main_carries_forward_source_pick_when_live_refresh_fails(
        self,
        load_dashboard_source_mock,
        _build_live_candidate_mock,
        write_json_mock,
    ) -> None:
        load_dashboard_source_mock.return_value = {
            "generated_at": "2026-03-16T12:00:00Z",
            "data_as_of": "2026-03-14",
            "market_context": {
                "timezone": "America/New_York",
                "week_id": "2026-W12",
                "week_label": "Mar 16 - 20, 2026",
                "week_start": "2026-03-16",
                "week_end": "2026-03-20",
            },
            "overall_selection": {
                "status": "picked",
                "status_reason": "MSFT cleared the thresholds.",
                "threshold_score": 0.10,
                "threshold_confidence": MIN_CONFIDENCE_THRESHOLD,
                "pick": {
                    "symbol": "MSFT",
                    "company_name": "Microsoft Corporation",
                    "sector": "technology",
                    "thesis_monitor": {"status": "healthy"},
                },
            },
        }

        main()

        payload = write_json_mock.call_args.args[1]
        self.assertEqual("picked", payload["selection"]["status"])
        self.assertEqual("MSFT", payload["active_pick"]["symbol"])
        self.assertIn("carried forward", payload["selection"]["status_reason"].lower())
        self.assertEqual("degraded", payload["data_quality"]["status"])


if __name__ == "__main__":
    unittest.main()
