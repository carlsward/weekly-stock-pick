import io
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from urllib.error import HTTPError

import backend.generate_news_scores as generate_news_scores_module

from backend.generate_news_scores import (
    ArticleReview,
    NewsArticle,
    aggregate_news_signal,
    aggregate_llm_news_signal,
    build_signal_article_entries,
    build_marketaux_article,
    compute_finbert_sentiment,
    compute_recency_weight,
    compute_relevance_score,
    fetch_raw_news,
    latest_article_date,
    select_supporting_article_entries,
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
    def tearDown(self) -> None:
        generate_news_scores_module.reset_marketaux_fetch_state()

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

    def test_fetch_marketaux_payload_returns_neutral_after_rate_limit_when_fallback_enabled(self) -> None:
        rate_limit_error = HTTPError(
            url="https://api.marketaux.com/v1/news/all",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"code":"rate_limit_reached","message":"Too many requests"}}'),
        )

        with patch.dict(
            "os.environ",
            {
                "MARKETAUX_API_TOKEN": "token",
                "ALLOW_MARKETAUX_FALLBACK": "true",
                "MARKETAUX_REQUEST_INTERVAL_SECONDS": "0",
            },
            clear=False,
        ):
            with patch("backend.generate_news_scores.urlopen", side_effect=rate_limit_error):
                payload = generate_news_scores_module.fetch_marketaux_payload(
                    "MSFT",
                    datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
                )

        self.assertEqual([], payload)
        self.assertTrue(generate_news_scores_module.marketaux_rate_limit_exhausted())

    def test_fetch_raw_news_short_circuits_after_rate_limit_in_same_run(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MARKETAUX_API_TOKEN": "token",
                "ALLOW_MARKETAUX_FALLBACK": "true",
            },
            clear=False,
        ):
            generate_news_scores_module.mark_marketaux_rate_limit_exhausted()

            with patch("backend.generate_news_scores.load_marketaux_fixture", return_value=None):
                with patch("backend.generate_news_scores.fetch_marketaux_payload") as fetch_mock:
                    payload = fetch_raw_news("MSFT")

        self.assertEqual([], payload)
        fetch_mock.assert_not_called()

    def test_llm_news_aggregation_discounts_market_roundups(self) -> None:
        articles = [
            NewsArticle(
                title="Market Retreat: S&P 500 and Dow Slip as Inflation Concerns Dampen Rate Cut Hopes",
                text="A broad market recap mentioning several megacap stocks.",
                published_at=datetime(2026, 3, 19, 8, 0, tzinfo=timezone.utc),
                provider="Blog",
                url="https://example.com/1",
                relevance_score=0.52,
                recency_weight=0.95,
                source_quality=0.75,
                weight=0.37,
            ),
            NewsArticle(
                title="Microsoft wins multi-year enterprise cloud contract",
                text="Microsoft signed a major enterprise cloud contract that should raise Azure revenue.",
                published_at=datetime(2026, 3, 19, 7, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://example.com/2",
                relevance_score=0.88,
                recency_weight=0.94,
                source_quality=1.00,
                weight=0.83,
            ),
        ]
        reviews = [
            ArticleReview(
                article_id="article_1",
                doc_type="market_roundup",
                company_relevance=0.18,
                materiality=0.12,
                impact_score=0.60,
                confidence=0.72,
                reason="Broad roundup with only incidental mention of Microsoft.",
            ),
            ArticleReview(
                article_id="article_2",
                doc_type="company_specific",
                company_relevance=0.93,
                materiality=0.86,
                impact_score=0.74,
                confidence=0.81,
                reason="Direct contract win with clear revenue relevance.",
            ),
        ]

        signal = aggregate_llm_news_signal(
            articles,
            reviews,
            {
                "summary": "Contract win matters more than the roundup.",
                "overall_signal": "bullish",
                "overall_impact_score": 0.68,
                "overall_confidence": 0.78,
            },
        )

        self.assertGreater(signal["news_score"], 0.55)
        self.assertLess(signal["llm_low_quality_share"], 0.60)
        self.assertGreater(signal["llm_average_directness"], 0.40)

    def test_supporting_article_freshness_ignores_fresh_roundup(self) -> None:
        articles = [
            NewsArticle(
                title="Stocks rise as rate-cut hopes return",
                text="A market roundup that mentions Microsoft only in passing.",
                published_at=datetime(2026, 3, 19, 9, 0, tzinfo=timezone.utc),
                provider="Blog",
                url="https://example.com/roundup",
                relevance_score=0.50,
                recency_weight=0.98,
                source_quality=0.75,
                weight=0.34,
            ),
            NewsArticle(
                title="Microsoft lands major enterprise software renewal",
                text="Microsoft signed a large renewal that directly supports next-quarter revenue.",
                published_at=datetime(2026, 3, 18, 11, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url="https://example.com/msft",
                relevance_score=0.89,
                recency_weight=0.88,
                source_quality=1.0,
                weight=0.78,
            ),
        ]
        reviews = [
            ArticleReview(
                article_id="article_1",
                doc_type="market_roundup",
                company_relevance=0.12,
                materiality=0.10,
                impact_score=0.30,
                confidence=0.65,
                reason="Incidental company mention inside a broad roundup.",
            ),
            ArticleReview(
                article_id="article_2",
                doc_type="company_specific",
                company_relevance=0.94,
                materiality=0.86,
                impact_score=0.77,
                confidence=0.82,
                reason="Direct company catalyst with revenue relevance.",
            ),
        ]

        supporting_entries = select_supporting_article_entries(
            build_signal_article_entries(articles, reviews)
        )

        self.assertEqual(1, len(supporting_entries))
        self.assertEqual("Microsoft lands major enterprise software renewal", supporting_entries[0]["article"].title)
        self.assertEqual("2026-03-18", latest_article_date([entry["article"] for entry in supporting_entries]))

    def test_fetch_raw_news_requires_marketaux_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                fetch_raw_news("MSFT")


if __name__ == "__main__":
    unittest.main()
