import unittest
from datetime import datetime, timezone

from backend.generate_sector_scores import (
    GlobalNewsArticle,
    aggregate_symbol_scores,
    aggregate_sector_scores,
    build_global_article,
    compute_macro_relevance,
    normalize_review_payload,
)


class GenerateSectorScoresTests(unittest.TestCase):
    def test_compute_macro_relevance_prefers_sector_level_news(self) -> None:
        generic = compute_macro_relevance(
            "Stocks finished mixed on Friday",
            "Markets were choppy in a quiet session.",
            entity_count=1,
        )
        macro = compute_macro_relevance(
            "Oil jumps after OPEC signals tighter supply",
            "Crude prices climbed and airlines warned about fuel costs after OPEC supply headlines.",
            entity_count=4,
        )

        self.assertLess(generic, 0.20)
        self.assertGreater(macro, 0.45)

    def test_normalize_review_payload_filters_unknown_sectors(self) -> None:
        articles = [
            GlobalNewsArticle(
                article_id="news_1",
                title="Oil jumps",
                text="Crude prices moved higher.",
                published_at=datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://example.com/1",
                macro_relevance=0.9,
                recency_weight=0.95,
                source_quality=1.0,
                weight=0.85,
                cluster_size=2,
            ),
            GlobalNewsArticle(
                article_id="news_2",
                title="Retail spending weakens",
                text="Consumer demand softened this month.",
                published_at=datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://example.com/2",
                macro_relevance=0.8,
                recency_weight=0.92,
                source_quality=1.0,
                weight=0.72,
                cluster_size=1,
            ),
        ]

        normalized = normalize_review_payload(
            {
                "articles": [
                    {
                        "article_id": "news_1",
                        "market_relevance": 0.9,
                        "magnitude": 0.8,
                        "confidence": 0.7,
                        "beneficiary_sectors": ["energy", "unknown_sector"],
                        "hurt_sectors": ["consumer_discretionary"],
                        "beneficiary_symbols": ["XOM", "UNKNOWN"],
                        "hurt_symbols": ["TSLA"],
                        "reason": "Higher fuel prices support producers and hurt fuel-intensive spending.",
                    }
                ]
            },
            articles,
            ["energy", "consumer_discretionary"],
            ["XOM", "TSLA"],
        )

        self.assertEqual(["energy"], normalized[0]["beneficiary_sectors"])
        self.assertEqual(["consumer_discretionary"], normalized[0]["hurt_sectors"])
        self.assertEqual(["XOM"], normalized[0]["beneficiary_symbols"])
        self.assertEqual(["TSLA"], normalized[0]["hurt_symbols"])
        self.assertEqual([], normalized[1]["beneficiary_sectors"])
        self.assertEqual([], normalized[1]["beneficiary_symbols"])
        self.assertEqual(0.0, normalized[1]["market_relevance"])

    def test_aggregate_sector_scores_reflects_positive_and_negative_sector_effects(self) -> None:
        articles = [
            GlobalNewsArticle(
                article_id="news_1",
                title="Oil jumps after OPEC supply warning",
                text="Energy producers rallied while airlines faced higher fuel costs.",
                published_at=datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://example.com/1",
                macro_relevance=0.9,
                recency_weight=0.95,
                source_quality=1.0,
                weight=0.90,
                cluster_size=3,
            ),
            GlobalNewsArticle(
                article_id="news_2",
                title="Drug pricing relief helps managed care margins",
                text="Healthcare reimbursement pressure eased.",
                published_at=datetime(2026, 3, 27, 9, 0, tzinfo=timezone.utc),
                provider="Bloomberg",
                url="https://example.com/2",
                macro_relevance=0.7,
                recency_weight=0.90,
                source_quality=0.96,
                weight=0.62,
                cluster_size=1,
            ),
        ]
        reviews = [
            {
                "article_id": "news_1",
                "market_relevance": 0.9,
                "magnitude": 0.9,
                "confidence": 0.8,
                "beneficiary_sectors": ["energy"],
                "hurt_sectors": ["consumer_discretionary"],
                "beneficiary_symbols": ["XOM"],
                "hurt_symbols": ["TSLA"],
                "reason": "Higher crude prices improve energy revenue and pressure travel demand.",
            },
            {
                "article_id": "news_2",
                "market_relevance": 0.7,
                "magnitude": 0.6,
                "confidence": 0.7,
                "beneficiary_sectors": ["healthcare"],
                "hurt_sectors": [],
                "beneficiary_symbols": ["UNH"],
                "hurt_symbols": [],
                "reason": "Health plan and drug margin outlook improved.",
            },
        ]

        aggregated = aggregate_sector_scores(
            articles,
            reviews,
            ["energy", "consumer_discretionary", "healthcare"],
        )

        self.assertGreater(aggregated["energy"]["score"], 0.55)
        self.assertLess(aggregated["consumer_discretionary"]["score"], 0.45)
        self.assertGreater(aggregated["healthcare"]["confidence"], 0.20)

        symbol_scores = aggregate_symbol_scores(
            articles,
            reviews,
            {
                "XOM": {"company_name": "Exxon Mobil Corporation", "sector": "energy"},
                "TSLA": {"company_name": "Tesla Inc.", "sector": "consumer_discretionary"},
                "UNH": {"company_name": "UnitedHealth Group Incorporated", "sector": "healthcare"},
            },
        )

        self.assertGreater(symbol_scores["XOM"]["score"], 0.55)
        self.assertLess(symbol_scores["TSLA"]["score"], 0.45)
        self.assertGreater(symbol_scores["UNH"]["confidence"], 0.20)
        self.assertEqual("2026-03-27", aggregated["energy"]["last_updated"])
        self.assertEqual("2026-03-27", symbol_scores["XOM"]["last_updated"])

    def test_build_global_article_keeps_recent_high_quality_multi_entity_story(self) -> None:
        now = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
        article = build_global_article(
            {
                "title": "Tesla shares fall after Europe registrations slow",
                "description": "Several automakers were cited as EV demand normalized in Europe.",
                "snippet": "Tesla, BYD and Volkswagen all appeared in the registration report.",
                "source": "Reuters",
                "url": "https://example.com/tesla-europe",
                "published_at": "2026-03-27T09:00:00Z",
                "entities": [{}, {}, {}],
                "similar": [{}],
            },
            now,
            0,
        )

        self.assertIsNotNone(article)
        self.assertGreaterEqual(article.macro_relevance, 0.18)


if __name__ == "__main__":
    unittest.main()
