import json
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from backend.generate_pick import (
    MIN_CONFIDENCE_THRESHOLD,
    MarketWeek,
    MacroSnapshot,
    NewsSnapshot,
    RISK_SELECTION_THRESHOLDS,
    SectorSnapshot,
    StockCandidate,
    build_macro_snapshot,
    build_thesis_monitor,
    dedupe_layer_penalties,
    default_model_calibration,
    market_day_age,
    build_price_series_from_frame,
    build_news_snapshot,
    build_market_week,
    compute_confidence_score,
    compute_score_breakdown,
    derive_data_as_of,
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

    def test_select_best_candidate_returns_no_pick_when_thresholds_fail(self) -> None:
        selection = select_best_candidate(
            [
                build_candidate("LOW1", "low", 0.03, 0.80),
                build_candidate("MID1", "medium", 0.09, 0.80),
                build_candidate("HIGH1", "high", 0.05, 0.80),
            ]
        )
        self.assertEqual("no_pick", selection.status)
        self.assertIsNone(selection.pick)
        self.assertIsNotNone(selection.best_candidate)

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
