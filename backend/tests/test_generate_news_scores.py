import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from backend.generate_news_scores import (
    NewsArticle,
    aggregate_news_signal,
    build_marketaux_article,
    compute_finbert_sentiment,
    compute_recency_weight,
    compute_relevance_score,
    fetch_raw_news,
)


def news_item(title: str, summary: str = "", description: str = "") -> dict:
    return {
        "content": {
            "title": title,
            "summary": summary,
            "description": description,
        }
    }


class GenerateNewsScoresTests(unittest.TestCase):
    def test_relevance_prefers_company_specific_article(self) -> None:
        generic = news_item(
            title="Stocks climb as the S&P 500 recovers",
            summary="The broader market moved higher after economic data.",
        )
        company_specific = news_item(
            title="AT&T expands fiber network after strong subscriber growth",
            summary="AT&T said the network expansion will accelerate consumer broadband reach.",
        )

        generic_score = compute_relevance_score(generic, "T", "AT&T Inc.")
        company_score = compute_relevance_score(company_specific, "T", "AT&T Inc.")

        self.assertLess(generic_score, 0.20)
        self.assertGreater(company_score, 0.45)

    def test_recency_weight_decays_for_older_articles(self) -> None:
        now = datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc)
        fresh_weight = compute_recency_weight(now - timedelta(hours=3), now)
        stale_weight = compute_recency_weight(now - timedelta(days=4), now)

        self.assertGreater(fresh_weight, stale_weight)
        self.assertGreaterEqual(fresh_weight, 0.30)
        self.assertGreaterEqual(stale_weight, 0.30)

    def test_aggregate_news_signal_rewards_high_quality_positive_coverage(self) -> None:
        articles = [
            NewsArticle(
                title="Microsoft signs major AI cloud contract",
                text="Microsoft signs major AI cloud contract and expands Azure demand.",
                published_at=datetime(2026, 3, 19, 8, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://www.reuters.com/example",
                relevance_score=0.90,
                recency_weight=0.95,
                source_quality=1.00,
                weight=0.86,
            ),
            NewsArticle(
                title="Microsoft lifts revenue outlook after enterprise demand improves",
                text="Microsoft lifts revenue outlook after enterprise demand improves.",
                published_at=datetime(2026, 3, 18, 10, 0, tzinfo=timezone.utc),
                provider="Bloomberg",
                url="https://www.bloomberg.com/example",
                relevance_score=0.84,
                recency_weight=0.82,
                source_quality=0.96,
                weight=0.66,
            ),
            NewsArticle(
                title="Opinion: valuations across software remain elevated",
                text="Opinion: valuations across software remain elevated.",
                published_at=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
                provider="Blog",
                url="https://example.com/post",
                relevance_score=0.48,
                recency_weight=0.42,
                source_quality=0.75,
                weight=0.15,
            ),
        ]

        signal = aggregate_news_signal(articles, [0.92, 0.71, -0.20])

        self.assertGreater(signal["news_score"], 0.55)
        self.assertGreater(signal["news_confidence"], 0.40)
        self.assertEqual("bullish", signal["dominant_signal"])

    def test_aggregate_news_signal_penalizes_concentrated_coverage(self) -> None:
        concentrated = [
            NewsArticle(
                title="Microsoft raises guidance after strong cloud demand",
                text="Microsoft raises guidance after strong cloud demand.",
                published_at=datetime(2026, 3, 19, 8, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://www.reuters.com/example",
                relevance_score=0.92,
                recency_weight=0.95,
                source_quality=1.00,
                weight=0.90,
            ),
            NewsArticle(
                title="Microsoft wins a smaller enterprise software deal",
                text="Microsoft wins a smaller enterprise software deal.",
                published_at=datetime(2026, 3, 19, 7, 0, tzinfo=timezone.utc),
                provider="Blog",
                url="https://example.com/post",
                relevance_score=0.70,
                recency_weight=0.90,
                source_quality=0.75,
                weight=0.10,
            ),
        ]
        diversified = [
            NewsArticle(
                title="Microsoft raises guidance after strong cloud demand",
                text="Microsoft raises guidance after strong cloud demand.",
                published_at=datetime(2026, 3, 19, 8, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://www.reuters.com/example",
                relevance_score=0.92,
                recency_weight=0.95,
                source_quality=1.00,
                weight=0.50,
            ),
            NewsArticle(
                title="Microsoft wins a smaller enterprise software deal",
                text="Microsoft wins a smaller enterprise software deal.",
                published_at=datetime(2026, 3, 19, 7, 0, tzinfo=timezone.utc),
                provider="Bloomberg",
                url="https://www.bloomberg.com/example",
                relevance_score=0.84,
                recency_weight=0.90,
                source_quality=0.96,
                weight=0.50,
            ),
        ]

        concentrated_signal = aggregate_news_signal(concentrated, [0.80, 0.75])
        diversified_signal = aggregate_news_signal(diversified, [0.80, 0.75])

        self.assertLess(concentrated_signal["news_confidence"], diversified_signal["news_confidence"])
        self.assertGreater(concentrated_signal["dominant_weight_share"], diversified_signal["dominant_weight_share"])

    def test_finbert_fallback_uses_heuristic_sentiment(self) -> None:
        articles = [
            NewsArticle(
                title="Microsoft beats earnings and raises guidance",
                text="Microsoft beats earnings and raises guidance after strong demand in Azure.",
                published_at=datetime(2026, 3, 19, 8, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://www.reuters.com/example",
                relevance_score=0.90,
                recency_weight=0.95,
                source_quality=1.00,
                weight=0.85,
            )
        ]

        signal = compute_finbert_sentiment(None, articles)

        self.assertGreater(signal["news_score"], 0.50)
        self.assertGreater(signal["news_confidence"], 0.20)

    def test_build_marketaux_article_uses_entity_metadata(self) -> None:
        now = datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc)
        article = build_marketaux_article(
            {
                "title": "Microsoft raises guidance after strong Azure demand",
                "description": "Management lifted its outlook after strong enterprise demand.",
                "snippet": "Microsoft said demand remains strong across cloud and AI products.",
                "source": "Reuters",
                "url": "https://www.reuters.com/example",
                "published_at": "2026-03-19T08:00:00Z",
                "entities": [
                    {
                        "symbol": "MSFT",
                        "match_score": 16.0,
                        "sentiment_score": 0.72,
                        "highlights": [
                            {
                                "highlight": "<em>Microsoft</em> raises guidance after strong Azure demand.",
                                "sentiment": 0.8,
                            }
                        ],
                    }
                ],
                "similar": [{}, {}],
            },
            "MSFT",
            "Microsoft Corporation",
            now,
        )

        self.assertIsNotNone(article)
        self.assertGreater(article.relevance_score, 0.60)
        self.assertGreater(article.entity_sentiment or 0.0, 0.60)
        self.assertEqual(3, article.cluster_size)
        self.assertTrue(article.highlight_snippets)

    def test_provider_sentiment_improves_signal_quality(self) -> None:
        articles = [
            NewsArticle(
                title="Microsoft expands enterprise AI contracts",
                text="A fairly neutral article body.",
                published_at=datetime(2026, 3, 19, 8, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://www.reuters.com/example",
                relevance_score=0.90,
                recency_weight=0.95,
                source_quality=1.00,
                weight=0.90,
                entity_sentiment=0.70,
            )
        ]

        signal = compute_finbert_sentiment(None, articles)

        self.assertGreater(signal["provider_sentiment_coverage"], 0.50)
        self.assertGreater(signal["news_score"], 0.50)

    def test_fetch_raw_news_requires_marketaux_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                fetch_raw_news("MSFT")


if __name__ == "__main__":
    unittest.main()
