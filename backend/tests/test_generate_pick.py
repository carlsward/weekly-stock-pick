import json
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

from backend.generate_pick import (
    GenerationStats,
    MIN_CONFIDENCE_THRESHOLD,
    MarketWeek,
    MacroSnapshot,
    NewsSnapshot,
    RISK_SELECTION_THRESHOLDS,
    SectorSnapshot,
    StockCandidate,
    build_macro_snapshot,
    build_synthetic_price_frame,
    build_thesis_monitor,
    dedupe_layer_penalties,
    default_model_calibration,
    fetch_price_frame,
    fetch_stooq_price_frame,
    fetch_twelvedata_price_frame,
    market_day_age,
    build_price_series_from_frame,
    build_news_snapshot,
    build_market_week,
    build_model_calibration,
    compute_confidence_score,
    compute_score_breakdown,
    derive_data_as_of,
    blend_weight_maps,
    normalize_history_entry,
    select_best_candidate,
    select_best_per_risk,
    stooq_symbol,
    stooq_symbol_variants,
    update_history,
)


def build_candidate(
    symbol: str,
    risk_level: str,
    total_score: float,
    confidence_score: float,
) -> StockCandidate:
    return StockCandidate(
        symbol=symbol,
        company_name=f"{symbol} Corp",
        reasons=["reason"],
        risk_level=risk_level,
        total_score=total_score,
        confidence_score=confidence_score,
        confidence_label=(
            "high"
            if confidence_score >= 0.75
            else "medium"
            if confidence_score >= MIN_CONFIDENCE_THRESHOLD
            else "low"
        ),
        price_as_of="2026-03-13",
        news_as_of="2026-03-14",
        article_count=4,
        effective_article_count=2.8,
        source_count=3,
        average_relevance=0.82,
        momentum_5d=0.04,
        momentum_20d=0.09,
        volatility=0.01,
        downside_volatility=0.008,
        max_drawdown=0.03,
        trend_gap=0.02,
        positive_day_ratio=0.7,
        volume_trend=0.18,
        market_relative_5d=0.01,
        market_relative_20d=0.02,
        sector_relative_5d=0.01,
        sector_relative_20d=0.02,
        news_score=0.7,
        news_confidence=0.76,
        macro_score=0.58,
        macro_confidence=0.64,
        raw_sentiment=0.2,
        calibrated_sentiment=0.16,
        dominant_signal="bullish",
        score_breakdown={
            "weighted_short_momentum": 0.10,
            "weighted_medium_momentum": 0.09,
            "weighted_trend_quality": 0.06,
            "weighted_volume_confirmation": 0.02,
            "weighted_market_relative": 0.02,
            "weighted_sector_relative": 0.01,
            "weighted_volatility_penalty": -0.03,
            "weighted_downside_penalty": -0.02,
            "weighted_drawdown_penalty": -0.01,
            "momentum_total": 0.27,
            "volatility_penalty_total": -0.06,
            "technical_total": 0.21,
            "weighted_news": total_score - 0.21,
            "weighted_macro": 0.01,
            "weighted_sector": 0.0,
            "weighted_signal_alignment": 0.0,
            "total": total_score,
        },
        news_evidence=[],
        macro_evidence=[],
        macro_as_of="2026-03-14",
    )


class GeneratePickTests(unittest.TestCase):
    def test_market_week_uses_current_week_on_trading_day(self) -> None:
        market_week = build_market_week(datetime(2026, 3, 18, 15, 0, tzinfo=timezone.utc))
        self.assertEqual("2026-03-16", market_week.week_start.isoformat())
        self.assertEqual("2026-03-20", market_week.week_end.isoformat())
        self.assertEqual("2026-W12", market_week.week_id)

    def test_market_week_rolls_forward_on_weekend(self) -> None:
        market_week = build_market_week(datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc))
        self.assertEqual("2026-03-23", market_week.week_start.isoformat())
        self.assertEqual("2026-03-27", market_week.week_end.isoformat())
        self.assertEqual("2026-W13", market_week.week_id)

    def test_score_breakdown_adds_positive_news_instead_of_punishing_negative_base(self) -> None:
        optimistic = compute_score_breakdown(momentum=-0.01, volatility=0.02, news_score=0.9)
        pessimistic = compute_score_breakdown(momentum=-0.01, volatility=0.02, news_score=0.1)
        self.assertGreater(optimistic["weighted_news"], 0)
        self.assertLess(pessimistic["weighted_news"], 0)
        self.assertGreater(optimistic["total"], pessimistic["total"])

    def test_score_breakdown_rewards_better_trend_quality(self) -> None:
        stronger = compute_score_breakdown(
            momentum=0.03,
            momentum_20d=0.11,
            volatility=0.012,
            trend_gap=0.04,
            positive_day_ratio=0.80,
            volume_trend=0.25,
            downside_volatility=0.006,
            max_drawdown=0.03,
            news_score=0.65,
            news_confidence=0.70,
        )
        weaker = compute_score_breakdown(
            momentum=0.03,
            momentum_20d=0.02,
            volatility=0.018,
            trend_gap=-0.01,
            positive_day_ratio=0.50,
            volume_trend=-0.20,
            downside_volatility=0.014,
            max_drawdown=0.09,
            news_score=0.65,
            news_confidence=0.70,
        )
        self.assertGreater(stronger["momentum_total"], weaker["momentum_total"])
        self.assertGreater(stronger["total"], weaker["total"])

    def test_default_model_calibration_normalizes_technical_weights(self) -> None:
        calibration = default_model_calibration()
        self.assertAlmostEqual(1.0, sum(calibration.technical_weights.values()), places=6)
        self.assertGreater(calibration.technical_scale, 0.0)

    def test_low_ic_technical_calibration_keeps_default_priors(self) -> None:
        default_weights = default_model_calibration().technical_weights
        feature_count = len(default_weights)
        rows = []
        for index in range(200):
            features = np.zeros(feature_count, dtype=float)
            features[index % feature_count] = 1.0
            rows.append((features, 0.01))

        with patch("backend.generate_pick.load_cached_model_calibration", return_value=None):
            with patch("backend.generate_pick.write_cached_model_calibration"):
                with patch("backend.generate_pick.build_calibration_training_rows", return_value=rows):
                    with patch("backend.generate_pick.information_coefficient", return_value=0.0):
                        calibration = build_model_calibration(
                            [("MSFT", "Microsoft Corporation")],
                            {"MSFT": "technology"},
                            [],
                        )

        self.assertEqual("price_history_low_ic_default_priors+block_priors", calibration.source)
        self.assertEqual(len(rows), calibration.training_row_count)
        self.assertEqual(0.0, calibration.training_ic)
        self.assertEqual(default_weights, calibration.technical_weights)

    def test_blend_weight_maps_scales_learned_share_gradually(self) -> None:
        base = {"news": 0.14, "macro": 0.12, "sector": 0.09}
        learned = {"news": 0.10, "macro": 0.18, "sector": 0.07}

        blended, learned_share = blend_weight_maps(
            base,
            learned,
            sample_count=18,
            min_rows=12,
            full_trust_rows=48,
        )

        self.assertGreater(learned_share, 0.0)
        self.assertLess(learned_share, 1.0)
        self.assertAlmostEqual(sum(base.values()), sum(blended.values()), places=6)
        self.assertNotEqual(base["macro"], blended["macro"])

    def test_score_breakdown_rewards_aligned_news_and_technical_signal(self) -> None:
        aligned = compute_score_breakdown(
            momentum=0.04,
            momentum_20d=0.10,
            volatility=0.01,
            trend_gap=0.03,
            positive_day_ratio=0.70,
            volume_trend=0.12,
            downside_volatility=0.006,
            max_drawdown=0.03,
            news_score=0.78,
            news_confidence=0.80,
        )
        conflicted = compute_score_breakdown(
            momentum=0.04,
            momentum_20d=0.10,
            volatility=0.01,
            trend_gap=0.03,
            positive_day_ratio=0.70,
            volume_trend=0.12,
            downside_volatility=0.006,
            max_drawdown=0.03,
            news_score=0.22,
            news_confidence=0.80,
        )
        self.assertGreater(aligned["weighted_signal_alignment"], 0.0)
        self.assertLess(conflicted["weighted_signal_alignment"], 0.0)
        self.assertGreater(aligned["total"], conflicted["total"])

    def test_score_breakdown_rewards_positive_sector_overlay(self) -> None:
        bullish_sector = compute_score_breakdown(
            momentum=0.03,
            momentum_20d=0.09,
            volatility=0.012,
            trend_gap=0.02,
            positive_day_ratio=0.65,
            volume_trend=0.10,
            downside_volatility=0.008,
            max_drawdown=0.03,
            news_score=0.50,
            news_confidence=0.50,
            sector_score=0.74,
            sector_confidence=0.82,
        )
        bearish_sector = compute_score_breakdown(
            momentum=0.03,
            momentum_20d=0.09,
            volatility=0.012,
            trend_gap=0.02,
            positive_day_ratio=0.65,
            volume_trend=0.10,
            downside_volatility=0.008,
            max_drawdown=0.03,
            news_score=0.50,
            news_confidence=0.50,
            sector_score=0.26,
            sector_confidence=0.82,
        )

        self.assertGreater(bullish_sector["weighted_sector"], 0.0)
        self.assertLess(bearish_sector["weighted_sector"], 0.0)
        self.assertGreater(bullish_sector["total"], bearish_sector["total"])

    def test_score_breakdown_dedupes_overlapping_macro_and_sector_themes(self) -> None:
        undeduped = compute_score_breakdown(
            momentum=0.03,
            momentum_20d=0.09,
            volatility=0.012,
            trend_gap=0.02,
            positive_day_ratio=0.65,
            volume_trend=0.10,
            downside_volatility=0.008,
            max_drawdown=0.03,
            news_score=0.72,
            news_confidence=0.76,
            macro_score=0.68,
            macro_confidence=0.74,
            sector_score=0.66,
            sector_confidence=0.72,
        )
        deduped = compute_score_breakdown(
            momentum=0.03,
            momentum_20d=0.09,
            volatility=0.012,
            trend_gap=0.02,
            positive_day_ratio=0.65,
            volume_trend=0.10,
            downside_volatility=0.008,
            max_drawdown=0.03,
            news_score=0.72,
            news_confidence=0.76,
            macro_score=0.68,
            macro_confidence=0.74,
            sector_score=0.66,
            sector_confidence=0.72,
            layer_penalties={
                "macro_penalty": 0.55,
                "sector_penalty": 0.45,
                "news_macro_overlap": 0.6,
                "news_sector_overlap": 0.5,
                "macro_sector_overlap": 0.5,
            },
        )

        self.assertLess(abs(deduped["weighted_macro"]), abs(undeduped["weighted_macro"]))
        self.assertLess(abs(deduped["weighted_sector"]), abs(undeduped["weighted_sector"]))

    def test_score_breakdown_rewards_relative_strength(self) -> None:
        stronger = compute_score_breakdown(
            momentum=0.03,
            momentum_20d=0.09,
            volatility=0.012,
            trend_gap=0.02,
            positive_day_ratio=0.65,
            volume_trend=0.10,
            downside_volatility=0.008,
            max_drawdown=0.03,
            news_score=0.50,
            news_confidence=0.50,
            market_relative_5d=0.04,
            market_relative_20d=0.06,
            sector_relative_5d=0.03,
            sector_relative_20d=0.05,
        )
        weaker = compute_score_breakdown(
            momentum=0.03,
            momentum_20d=0.09,
            volatility=0.012,
            trend_gap=0.02,
            positive_day_ratio=0.65,
            volume_trend=0.10,
            downside_volatility=0.008,
            max_drawdown=0.03,
            news_score=0.50,
            news_confidence=0.50,
            market_relative_5d=-0.04,
            market_relative_20d=-0.06,
            sector_relative_5d=-0.03,
            sector_relative_20d=-0.05,
        )

        self.assertGreater(stronger["weighted_market_relative"], 0.0)
        self.assertGreater(stronger["weighted_sector_relative"], 0.0)
        self.assertLess(weaker["weighted_market_relative"], 0.0)
        self.assertLess(weaker["weighted_sector_relative"], 0.0)
        self.assertGreater(stronger["total"], weaker["total"])

    def test_confidence_score_rewards_better_evidence(self) -> None:
        higher = compute_confidence_score(
            total_score=0.22,
            article_count=5,
            price_observations=40,
            news_score=0.68,
            positive_day_ratio=0.70,
            max_drawdown=0.04,
            news_confidence=0.75,
            effective_article_count=3.5,
        )
        lower = compute_confidence_score(
            total_score=0.22,
            article_count=0,
            price_observations=22,
            news_score=0.50,
            positive_day_ratio=0.50,
            max_drawdown=0.12,
            news_confidence=0.20,
            effective_article_count=0.0,
        )
        self.assertGreater(higher, lower)

    def test_stale_news_snapshot_reduces_sentiment_strength(self) -> None:
        snapshot = build_news_snapshot(
            {
                "news_score": 0.90,
                "news_confidence": 0.82,
                "raw_sentiment": 0.70,
                "calibrated_sentiment": 0.52,
                "article_count": 4,
                "effective_article_count": 2.8,
                "source_count": 3,
                "average_relevance": 0.80,
                "dominant_signal": "bullish",
                "last_updated": "2026-03-10",
                "news_reasons": [],
            }
        )
        self.assertLess(snapshot.news_score, 0.90)
        self.assertLess(snapshot.news_confidence, 0.82)
        self.assertLess(snapshot.calibrated_sentiment, 0.52)
        self.assertTrue(any("days old" in reason for reason in snapshot.reasons))

    def test_news_snapshot_falls_back_to_top_headlines_when_articles_are_missing(self) -> None:
        snapshot = build_news_snapshot(
            {
                "news_score": 0.62,
                "news_confidence": 0.64,
                "raw_sentiment": 0.20,
                "calibrated_sentiment": 0.15,
                "article_count": 2,
                "top_headlines": [
                    "Microsoft signs new enterprise AI agreement",
                    "Azure demand remains strong in Europe",
                ],
            }
        )
        self.assertEqual(2, len(snapshot.news_evidence))
        self.assertEqual("Microsoft signs new enterprise AI agreement", snapshot.news_evidence[0]["title"])

    def test_news_snapshot_uses_supporting_article_dates_when_last_updated_missing(self) -> None:
        snapshot = build_news_snapshot(
            {
                "news_score": 0.62,
                "news_confidence": 0.64,
                "raw_sentiment": 0.20,
                "calibrated_sentiment": 0.15,
                "article_count": 2,
                "top_articles": [
                    {
                        "title": "Microsoft signs new enterprise AI agreement",
                        "published_at": "2026-03-13T15:00:00Z",
                    }
                ],
            }
        )

        self.assertEqual("2026-03-13", snapshot.last_updated)

    def test_build_macro_snapshot_uses_symbol_specific_last_updated(self) -> None:
        snapshot = build_macro_snapshot(
            symbol="MSFT",
            company_name="Microsoft Corporation",
            sector="technology",
            sector_scores_payload={
                "last_updated": "2026-03-27",
                "symbol_scores": {
                    "MSFT": {
                        "symbol": "MSFT",
                        "company_name": "Microsoft Corporation",
                        "sector": "technology",
                        "score": 0.66,
                        "confidence": 0.72,
                        "direction": "bullish",
                        "last_updated": "2026-03-26",
                        "reasons": ["AI demand stayed strong."],
                        "supporting_articles": [
                            {
                                "title": "AI demand improved",
                                "published_at": "2026-03-26T10:00:00Z",
                                "impact": "positive",
                                "weight": 0.42,
                                "reason": "AI demand improved",
                            }
                        ],
                    }
                },
            },
        )

        self.assertEqual("2026-03-26", snapshot.last_updated)

    def test_build_macro_snapshot_does_not_inherit_payload_freshness_without_symbol_support(self) -> None:
        snapshot = build_macro_snapshot(
            symbol="MSFT",
            company_name="Microsoft Corporation",
            sector="technology",
            sector_scores_payload={
                "last_updated": "2026-03-27",
                "symbol_scores": {
                    "MSFT": {
                        "symbol": "MSFT",
                        "company_name": "Microsoft Corporation",
                        "sector": "technology",
                        "score": 0.50,
                        "confidence": 0.20,
                        "direction": "neutral",
                        "reasons": ["No strong catalyst."],
                        "supporting_articles": [],
                    }
                },
            },
        )

        self.assertIsNone(snapshot.last_updated)

    def test_fetch_price_frame_skips_stooq_when_disabled(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TWELVEDATA_API_KEY": "td",
                "ALPHA_VANTAGE_API_KEY": "av",
                "ALLOW_STOOQ_FALLBACK": "false",
                "ALLOW_PRICE_FALLBACK": "false",
            },
            clear=False,
        ):
            with patch("backend.generate_pick.fetch_twelvedata_price_frame", side_effect=RuntimeError("td down")):
                with patch("backend.generate_pick.fetch_alpha_vantage_price_frame", side_effect=RuntimeError("av down")):
                    with patch("backend.generate_pick.fetch_stooq_price_frame") as stooq_mock:
                        with patch("backend.generate_pick.time_module.sleep"):
                            with self.assertRaises(RuntimeError):
                                fetch_price_frame("MSFT")

        stooq_mock.assert_not_called()

    def test_derive_data_as_of_includes_macro_and_sector_dates(self) -> None:
        first = build_candidate("MSFT", "low", total_score=0.21, confidence_score=0.82)
        first.news_as_of = "2026-03-14"
        first.macro_as_of = "2026-03-19"
        first.sector_as_of = "2026-03-18"

        second = build_candidate("NVDA", "medium", total_score=0.19, confidence_score=0.74)
        second.price_as_of = "2026-03-17"
        second.news_as_of = "2026-03-16"
        second.macro_as_of = None
        second.sector_as_of = None

        self.assertEqual("2026-03-19", derive_data_as_of([first, second]))

    def test_dedupe_layer_penalties_detect_theme_overlap(self) -> None:
        penalties = dedupe_layer_penalties(
            NewsSnapshot(
                news_score=0.64,
                reasons=["Semiconductor demand improved and chip shortages eased."],
                raw_sentiment=0.20,
                calibrated_sentiment=0.14,
                article_count=2,
                effective_article_count=1.5,
                source_count=2,
                average_relevance=0.8,
                news_confidence=0.72,
                dominant_signal="bullish",
                last_updated="2026-03-14",
                news_evidence=[{"title": "Chip demand rebounds", "llm_reason": "Semiconductor demand improved."}],
            ),
            MacroSnapshot(
                symbol="NVDA",
                company_name="NVIDIA Corporation",
                sector="technology",
                macro_score=0.66,
                macro_confidence=0.74,
                direction="bullish",
                reasons=["Semiconductor supply chain conditions improved."],
                supporting_articles=[{"title": "Chip supply improves", "reason": "Semiconductor supply chain improved."}],
                last_updated="2026-03-14",
            ),
            SectorSnapshot(
                sector="technology",
                sector_score=0.63,
                sector_confidence=0.70,
                direction="bullish",
                reasons=["Technology benefited from stronger semiconductor demand."],
                supporting_articles=[{"title": "Technology lifted by chip cycle", "reason": "chip cycle stronger"}],
                last_updated="2026-03-14",
            ),
        )

        self.assertGreater(penalties["news_macro_overlap"], 0.0)
        self.assertGreater(penalties["news_sector_overlap"], 0.0)
        self.assertLess(penalties["macro_penalty"], 1.0)
        self.assertLess(penalties["sector_penalty"], 1.0)

    def test_thesis_monitor_marks_clean_setup_as_healthy(self) -> None:
        candidate = build_candidate("MSFT", "low", total_score=0.21, confidence_score=0.82)
        monitor = build_thesis_monitor(
            candidate,
            threshold_score=0.10,
            threshold_confidence=0.55,
            reference_date=date(2026, 3, 14),
        )

        self.assertEqual("healthy", monitor["status"])
        self.assertEqual("Support is intact", monitor["headline"])
        self.assertEqual(6, len(monitor["signals"]))

    def test_thesis_monitor_flags_narrow_margin_and_stale_data(self) -> None:
        candidate = build_candidate("NVDA", "high", total_score=0.125, confidence_score=0.57)
        candidate.price_as_of = "2026-03-06"
        candidate.news_as_of = "2026-03-06"
        candidate.article_count = 1
        candidate.effective_article_count = 0.6
        candidate.news_score = 0.48
        candidate.news_confidence = 0.38

        monitor = build_thesis_monitor(
            candidate,
            threshold_score=0.12,
            threshold_confidence=0.55,
            reference_date=date(2026, 3, 14),
        )

        self.assertEqual("risk", monitor["status"])
        self.assertTrue(any("barely cleared the release bar" in alert for alert in monitor["alerts"]))
        self.assertTrue(any(signal["label"] == "Freshness" and signal["state"] == "risk" for signal in monitor["signals"]))

    def test_thesis_monitor_flags_stale_macro_overlay(self) -> None:
        candidate = build_candidate("MSFT", "low", total_score=0.21, confidence_score=0.82)
        candidate.macro_as_of = "2026-03-09"

        monitor = build_thesis_monitor(
            candidate,
            threshold_score=0.10,
            threshold_confidence=0.55,
            reference_date=date(2026, 3, 14),
        )

        freshness_signal = next(signal for signal in monitor["signals"] if signal["label"] == "Freshness")
        self.assertEqual("risk", freshness_signal["state"])
        self.assertIn("macro overlay", freshness_signal["detail"])

    def test_market_day_age_ignores_weekend_gap(self) -> None:
        self.assertEqual(1, market_day_age(date(2026, 3, 20), date(2026, 3, 23)))

    def test_stooq_symbol_maps_us_equities(self) -> None:
        self.assertEqual("AAPL.US", stooq_symbol("AAPL"))
        self.assertEqual("BRK-B.US", stooq_symbol("BRK.B"))

    def test_stooq_symbol_variants_include_common_us_forms(self) -> None:
        self.assertEqual(["AAPL.US", "AAPL"], stooq_symbol_variants("AAPL"))
        self.assertEqual(
            ["BRK-B.US", "BRK-B", "BRK.B.US", "BRK.B", "BRKB.US", "BRKB"],
            stooq_symbol_variants("BRK.B"),
        )

    def test_fetch_stooq_price_frame_sends_optional_api_key(self) -> None:
        captured = {}

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"Date,Open,High,Low,Close,Volume\n2026-03-27,1,1,1,100,1000\n"

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return DummyResponse()

        with patch.dict("os.environ", {"STOOQ_API_KEY": "stooq-test"}, clear=False):
            with patch("backend.generate_pick.urlopen", side_effect=fake_urlopen):
                frame = fetch_stooq_price_frame("SPY")

        self.assertFalse(frame.empty)
        self.assertIn("apikey=stooq-test", captured["url"])

    def test_fetch_stooq_price_frame_rejects_api_key_instruction_page(self) -> None:
        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"Get your apikey:\n1. Open https://stooq.com/q/d/?s=spy.us&get_apikey"

        with patch("backend.generate_pick.urlopen", return_value=DummyResponse()):
            with self.assertRaises(RuntimeError):
                fetch_stooq_price_frame("SPY")

    def test_build_price_series_from_frame_uses_recent_sorted_data(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2026-01-01", periods=30, freq="D")[::-1],
                "Close": list(range(30, 0, -1)),
                "Volume": [1_000_000 + index for index in range(30)],
            }
        )

        series = build_price_series_from_frame(frame, "TEST", max_days=25)

        self.assertEqual(25, series.observations)
        self.assertEqual("2026-01-30", series.latest_trading_date)
        self.assertEqual(30, series.closes[0])
        self.assertEqual(6, series.closes[-1])

    def test_build_synthetic_price_frame_produces_usable_history(self) -> None:
        frame = build_synthetic_price_frame("MSFT", periods=80)

        self.assertEqual(80, len(frame))
        self.assertEqual(["Date", "Close", "Volume"], list(frame.columns))
        self.assertTrue((frame["Close"] > 0).all())
        self.assertTrue((frame["Volume"] > 0).all())

    def test_fetch_price_frame_uses_alpha_vantage_before_stooq(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2026-01-01", periods=30, freq="B"),
                "Close": [100 + index for index in range(30)],
                "Volume": [1_000_000 for _ in range(30)],
            }
        )

        with patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "alpha-test"}, clear=False):
            with patch("backend.generate_pick.twelvedata_api_key", return_value=""):
                with patch("backend.generate_pick.force_price_fallback", return_value=False):
                    with patch("backend.generate_pick.fetch_alpha_vantage_price_frame", return_value=frame) as alpha_mock:
                        with patch("backend.generate_pick.fetch_stooq_price_frame") as stooq_mock:
                            fetched = fetch_price_frame("MSFT")

        alpha_mock.assert_called_once_with("MSFT")
        stooq_mock.assert_not_called()
        self.assertFalse(fetched.empty)

    def test_fetch_twelvedata_price_frame_parses_values_payload(self) -> None:
        payload = {
            "meta": {"symbol": "MSFT", "interval": "1day"},
            "values": [
                {"datetime": "2026-03-27", "close": "389.12", "volume": "1234567"},
                {"datetime": "2026-03-26", "close": "386.45", "volume": "1122334"},
            ],
        }

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch.dict("os.environ", {"TWELVEDATA_API_KEY": "twelve-test"}, clear=False):
            with patch("backend.generate_pick.apply_twelvedata_request_spacing"):
                with patch("backend.generate_pick.urlopen", return_value=DummyResponse()):
                    frame = fetch_twelvedata_price_frame("MSFT")

        self.assertEqual(["Date", "Close", "Volume"], list(frame.columns))
        self.assertEqual(2, len(frame))
        self.assertEqual("2026-03-27", str(frame.iloc[0]["Date"]))
        self.assertEqual("389.12", str(frame.iloc[0]["Close"]))

    def test_fetch_price_frame_uses_twelvedata_before_alpha_and_stooq(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2026-01-01", periods=30, freq="B"),
                "Close": [100 + index for index in range(30)],
                "Volume": [1_000_000 for _ in range(30)],
            }
        )

        with patch.dict(
            "os.environ",
            {"TWELVEDATA_API_KEY": "twelve-test", "ALPHA_VANTAGE_API_KEY": "alpha-test"},
            clear=False,
        ):
            with patch("backend.generate_pick.force_price_fallback", return_value=False):
                with patch("backend.generate_pick.fetch_twelvedata_price_frame", return_value=frame) as twelve_mock:
                    with patch("backend.generate_pick.fetch_alpha_vantage_price_frame") as alpha_mock:
                        with patch("backend.generate_pick.fetch_stooq_price_frame") as stooq_mock:
                            fetched = fetch_price_frame("MSFT")

        twelve_mock.assert_called_once_with("MSFT")
        alpha_mock.assert_not_called()
        stooq_mock.assert_not_called()
        self.assertFalse(fetched.empty)

    def test_fetch_price_frame_falls_back_to_alpha_when_twelvedata_fails(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2026-01-01", periods=30, freq="B"),
                "Close": [100 + index for index in range(30)],
                "Volume": [1_000_000 for _ in range(30)],
            }
        )

        with patch.dict(
            "os.environ",
            {"TWELVEDATA_API_KEY": "twelve-test", "ALPHA_VANTAGE_API_KEY": "alpha-test"},
            clear=False,
        ):
            with patch("backend.generate_pick.force_price_fallback", return_value=False):
                with patch(
                    "backend.generate_pick.fetch_twelvedata_price_frame",
                    side_effect=RuntimeError("twelve rate limit"),
                ) as twelve_mock:
                    with patch("backend.generate_pick.fetch_alpha_vantage_price_frame", return_value=frame) as alpha_mock:
                        with patch("backend.generate_pick.fetch_stooq_price_frame") as stooq_mock:
                            fetched = fetch_price_frame("MSFT")

        self.assertGreaterEqual(twelve_mock.call_count, 1)
        alpha_mock.assert_called_once_with("MSFT")
        stooq_mock.assert_not_called()
        self.assertFalse(fetched.empty)

    def test_fetch_price_frame_falls_back_to_stooq_when_alpha_vantage_fails(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2026-01-01", periods=30, freq="B"),
                "Close": [100 + index for index in range(30)],
                "Volume": [1_000_000 for _ in range(30)],
            }
        )

        with patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "alpha-test"}, clear=False):
            with patch("backend.generate_pick.twelvedata_api_key", return_value=""):
                with patch("backend.generate_pick.force_price_fallback", return_value=False):
                    with patch(
                        "backend.generate_pick.fetch_alpha_vantage_price_frame",
                        side_effect=RuntimeError("rate limit"),
                    ) as alpha_mock:
                        with patch("backend.generate_pick.fetch_stooq_price_frame", return_value=frame) as stooq_mock:
                            fetched = fetch_price_frame("MSFT")

        self.assertGreaterEqual(alpha_mock.call_count, 1)
        stooq_mock.assert_called_once_with("MSFT")
        self.assertFalse(fetched.empty)

    def test_select_best_candidate_returns_low_confidence_pick_when_thresholds_fail(self) -> None:
        selection = select_best_candidate(
            [
                build_candidate("LOW1", "low", 0.03, 0.80),
                build_candidate("MID1", "medium", 0.09, 0.80),
                build_candidate("HIGH1", "high", 0.05, 0.80),
            ]
        )
        self.assertEqual("picked", selection.status)
        self.assertFalse(selection.is_qualified)
        self.assertEqual("low_confidence", selection.release_quality)
        self.assertEqual("MID1", selection.pick.symbol)
        self.assertIsNotNone(selection.best_candidate)

    def test_overall_selection_labels_candidate_below_risk_specific_score_floor(self) -> None:
        selection = select_best_candidate(
            [
                build_candidate("HIGH1", "high", RISK_SELECTION_THRESHOLDS["high"] - 0.001, 0.80),
                build_candidate("MID1", "medium", 0.09, 0.80),
            ]
        )

        self.assertEqual("picked", selection.status)
        self.assertFalse(selection.is_qualified)
        self.assertEqual("HIGH1", selection.pick.symbol)
        self.assertEqual(RISK_SELECTION_THRESHOLDS["high"], selection.threshold_score)
        self.assertIn("high-risk bar", selection.status_reason)

    def test_select_best_candidate_returns_no_pick_when_no_candidates_exist(self) -> None:
        selection = select_best_candidate([])

        self.assertEqual("no_pick", selection.status)
        self.assertIsNone(selection.pick)
        self.assertIsNone(selection.best_candidate)

    def test_select_best_candidate_uses_explicit_price_failure_policy(self) -> None:
        selection = select_best_candidate(
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

    def test_select_best_per_risk_can_mix_picks_and_no_picks(self) -> None:
        selections = select_best_per_risk(
            [
                build_candidate("LOW1", "low", RISK_SELECTION_THRESHOLDS["low"] + 0.01, 0.80),
                build_candidate("MID1", "medium", 0.02, 0.80),
                build_candidate("HIGH1", "high", RISK_SELECTION_THRESHOLDS["high"] + 0.01, 0.80),
            ]
        )
        self.assertEqual("picked", selections["low"].status)
        self.assertEqual("no_pick", selections["medium"].status)
        self.assertEqual("picked", selections["high"].status)

    def test_normalize_history_entry_migrates_legacy_shape(self) -> None:
        normalized = normalize_history_entry(
            {
                "week_start": "2026-03-16",
                "week_end": "2026-03-20",
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
                "score": 0.14,
            }
        )
        self.assertEqual("2026-W12", normalized["week_id"])
        self.assertEqual("picked", normalized["status"])
        self.assertEqual(0.14, normalized["model_score"])

    def test_update_history_replaces_existing_week_entry(self) -> None:
        temp_dir = Path(self.id().replace(".", "_"))
        temp_dir.mkdir(exist_ok=True)
        history_path = temp_dir / "history.json"

        try:
            initial_payload = {
                "schema_version": 2,
                "model_version": "v1.0",
                "generated_at": "2026-03-10T12:00:00Z",
                "entries": [
                    {
                        "week_id": "2026-W12",
                        "week_start": "2026-03-16",
                        "week_end": "2026-03-20",
                        "week_label": "Mar 16 - 20, 2026",
                        "logged_at": "2026-03-16T12:00:00Z",
                        "status": "picked",
                        "status_reason": "old",
                        "symbol": "OLD",
                        "company_name": "Old Corp",
                        "risk": "medium",
                        "model_score": 0.11,
                        "confidence_score": 0.75,
                        "confidence_label": "high",
                        "data_as_of": "2026-03-13",
                        "model_version": "v1.0",
                    }
                ],
            }
            history_path.write_text(json.dumps(initial_payload), encoding="utf-8")

            market_week = MarketWeek(
                week_id="2026-W12",
                week_start=date(2026, 3, 16),
                week_end=date(2026, 3, 20),
                week_label="Mar 16 - 20, 2026",
            )
            payload = update_history(
                market_week=market_week,
                selection=select_best_candidate([build_candidate("NEW", "medium", 0.20, 0.90)]),
                generated_at="2026-03-16T12:00:00Z",
                data_as_of="2026-03-13",
                history_path=history_path,
            )

            self.assertEqual(1, len(payload["entries"]))
            self.assertEqual("NEW", payload["entries"][0]["symbol"])
        finally:
            if history_path.exists():
                history_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
