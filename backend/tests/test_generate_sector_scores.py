import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from backend.generate_sector_scores import (
    GlobalNewsArticle,
    aggregate_symbol_scores,
    aggregate_sector_scores,
    build_analysis_prompt,
    build_global_article,
    compute_macro_relevance,
    fetch_recent_global_articles,
    normalize_alpha_vantage_item,
    normalize_gdelt_item,
    normalize_review_payload,
    select_articles_for_llm,
)


class GenerateSectorScoresTests(unittest.TestCase):
    def _marketaux_global_item(self, title: str) -> dict:
        return {
            "title": title,
            "description": "Oil prices and sector demand expectations moved after a global supply headline.",
            "snippet": "",
            "source": "Reuters",
            "url": f"https://example.com/{abs(hash(title))}",
            "published_at": "2026-03-27T10:00:00Z",
            "entities": [{}, {}],
            "similar": [],
        }

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

    def test_compute_macro_relevance_captures_geopolitical_supply_shocks(self) -> None:
        macro = compute_macro_relevance(
            "Iran threatens Hormuz closure after regional conflict escalates",
            "Oil markets, shipping insurers and airline operators reacted to the geopolitical risk and possible supply disruption.",
            entity_count=3,
        )

        self.assertGreater(macro, 0.45)

    def test_build_analysis_prompt_mentions_broad_world_event_categories(self) -> None:
        prompt = build_analysis_prompt(
            [
                GlobalNewsArticle(
                    article_id="news_1",
                    title="Iran threatens Hormuz closure",
                    text="Geopolitical tensions raised the risk of an oil supply shock.",
                    published_at=datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc),
                    provider="Reuters",
                    url="https://example.com/1",
                    macro_relevance=0.9,
                    recency_weight=0.95,
                    source_quality=1.0,
                    weight=0.90,
                    cluster_size=2,
                )
            ],
            ["energy"],
            {"XOM": {"company_name": "Exxon Mobil Corporation", "sector": "energy"}},
        )

        lowered = prompt.lower()
        self.assertIn("geopolitical", lowered)
        self.assertIn("elections", lowered)
        self.assertIn("cyberattacks", lowered)
        self.assertIn("natural disasters", lowered)

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
                        "event_type": "oil_supply_shock",
                        "transmission_channel": "Tighter crude supply lifts fuel input costs and producer realizations.",
                        "affected_inputs": ["oil_price", "fuel_costs"],
                        "horizon": "1-2w",
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
        self.assertEqual("oil_supply_shock", normalized[0]["event_type"])
        self.assertEqual("1-2w", normalized[0]["horizon"])
        self.assertEqual(["oil_price", "fuel_costs"], normalized[0]["affected_inputs"])
        self.assertEqual([], normalized[1]["beneficiary_sectors"])
        self.assertEqual([], normalized[1]["beneficiary_symbols"])
        self.assertEqual(0.0, normalized[1]["market_relevance"])
        self.assertEqual("unclear", normalized[1]["event_type"])

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
                "event_type": "oil_supply_shock",
                "transmission_channel": "Higher crude prices boost upstream revenue and pressure fuel-intensive demand.",
                "affected_inputs": ["oil_price", "fuel_costs"],
                "horizon": "1-2w",
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
                "event_type": "reimbursement_tailwind",
                "transmission_channel": "Lower reimbursement pressure supports healthcare margins.",
                "affected_inputs": ["reimbursement", "medical_margins"],
                "horizon": "1w",
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
        self.assertEqual("oil_supply_shock", aggregated["energy"]["supporting_articles"][0]["event_type"])
        self.assertEqual("1-2w", symbol_scores["XOM"]["supporting_articles"][0]["horizon"])

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

    def test_fetch_recent_global_articles_stops_after_marketaux_when_sufficient(self) -> None:
        marketaux_payload = [
            self._marketaux_global_item(f"Oil jumps after OPEC supply warning {index}")
            for index in range(10)
        ]

        with patch.dict("os.environ", {"MARKETAUX_API_TOKEN": "token"}, clear=False):
            with patch(
                "backend.generate_sector_scores.fetch_marketaux_global_payload",
                return_value=marketaux_payload,
            ):
                with patch("backend.generate_sector_scores.fetch_gdelt_payload") as gdelt_mock:
                    with patch("backend.generate_sector_scores.fetch_alpha_vantage_payload") as alpha_mock:
                        articles = fetch_recent_global_articles()

        self.assertGreaterEqual(len(articles), 8)
        gdelt_mock.assert_not_called()
        alpha_mock.assert_not_called()

    def test_fetch_recent_global_articles_escalates_to_gdelt_before_alpha(self) -> None:
        marketaux_payload = [
            self._marketaux_global_item("Oil rises after supply risk returns")
        ]
        gdelt_payload = [
            self._marketaux_global_item(f"Global supply shock pressures sectors {index}")
            for index in range(9)
        ]

        with patch.dict("os.environ", {"MARKETAUX_API_TOKEN": "token"}, clear=False):
            with patch(
                "backend.generate_sector_scores.fetch_marketaux_global_payload",
                return_value=marketaux_payload,
            ):
                with patch(
                    "backend.generate_sector_scores.fetch_gdelt_payload",
                    return_value=gdelt_payload,
                ) as gdelt_mock:
                    with patch(
                        "backend.generate_sector_scores.normalize_gdelt_item",
                        side_effect=lambda item: item,
                    ):
                        with patch("backend.generate_sector_scores.fetch_alpha_vantage_payload") as alpha_mock:
                            articles = fetch_recent_global_articles()

        self.assertGreaterEqual(len(articles), 6)
        gdelt_mock.assert_called_once()
        alpha_mock.assert_not_called()

    def test_normalize_alpha_vantage_item_builds_marketaux_like_shape(self) -> None:
        normalized = normalize_alpha_vantage_item(
            {
                "title": "Oil prices jump as shipping disruptions threaten supply routes",
                "summary": "Energy shares gained as crude prices rose on renewed transport risk.",
                "url": "https://example.com/alpha-oil",
                "time_published": "20260327T1130",
                "source": "Reuters",
                "topics": [
                    {"topic": "energy_transportation"},
                    {"topic": "economy_macro"},
                ],
                "ticker_sentiment": [
                    {"ticker": "XOM"},
                    {"ticker": "CVX"},
                ],
                "overall_sentiment_label": "Neutral",
            }
        )

        self.assertIsNotNone(normalized)
        self.assertEqual("Reuters", normalized["source"])
        self.assertEqual(2, len(normalized["entities"]))
        self.assertIn("energy_transportation", normalized["snippet"])
        self.assertEqual("2026-03-27T11:30:00Z", normalized["published_at"])

    def test_normalize_gdelt_item_preserves_title_and_domain(self) -> None:
        normalized = normalize_gdelt_item(
            {
                "title": "Semiconductor shortage fears return after new export restrictions",
                "url": "https://example.com/gdelt-chip",
                "domain": "ft.com",
                "seendate": "20260327T084500Z",
                "language": "English",
                "sourcecountry": "UK",
            }
        )

        self.assertIsNotNone(normalized)
        self.assertEqual("ft.com", normalized["source"])
        self.assertEqual("2026-03-27T08:45:00Z", normalized["published_at"])
        self.assertIn("Source country: UK.", normalized["description"])

    def test_select_articles_for_llm_preserves_feed_mix(self) -> None:
        articles = [
            GlobalNewsArticle(
                article_id=f"marketaux_{index}",
                title=f"Market story {index}",
                text="Broad market story",
                published_at=datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url=f"https://example.com/m{index}",
                macro_relevance=0.9,
                recency_weight=0.9,
                source_quality=1.0,
                weight=1.0 - index * 0.01,
                cluster_size=1,
                feed="marketaux",
            )
            for index in range(12)
        ] + [
            GlobalNewsArticle(
                article_id=f"gdelt_{index}",
                title=f"GDELT story {index}",
                text="Oil supply disruption story",
                published_at=datetime(2026, 3, 27, 11, 0, tzinfo=timezone.utc),
                provider="ft.com",
                url=f"https://example.com/g{index}",
                macro_relevance=0.85,
                recency_weight=0.88,
                source_quality=0.95,
                weight=0.95 - index * 0.01,
                cluster_size=1,
                feed="gdelt",
            )
            for index in range(6)
        ] + [
            GlobalNewsArticle(
                article_id=f"alpha_{index}",
                title=f"Alpha story {index}",
                text="Macro and rates story",
                published_at=datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc),
                provider="Reuters",
                url=f"https://example.com/a{index}",
                macro_relevance=0.82,
                recency_weight=0.87,
                source_quality=0.95,
                weight=0.92 - index * 0.01,
                cluster_size=1,
                feed="alpha_vantage",
            )
            for index in range(5)
        ]

        selected = select_articles_for_llm(articles, 12)
        selected_feeds = {article.feed for article in selected}

        self.assertIn("marketaux", selected_feeds)
        self.assertIn("gdelt", selected_feeds)
        self.assertIn("alpha_vantage", selected_feeds)


if __name__ == "__main__":
    unittest.main()
