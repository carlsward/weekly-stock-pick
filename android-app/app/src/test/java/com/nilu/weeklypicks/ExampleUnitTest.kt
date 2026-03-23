package com.nilu.weeklypicks

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class ExampleUnitTest {
    private val parser = ContractParser()

    @Test
    fun parser_maps_dashboard_contract() {
        val dashboard = parser.parseDashboard(sampleDashboardJson)

        assertEquals("2026-W12", dashboard.marketContext.weekId)
        assertEquals(SelectionStatus.PICKED, dashboard.overallSelection.status)
        assertEquals(SelectionStatus.NO_PICK, dashboard.riskSelections.getValue("high").status)
        assertEquals("MSFT", dashboard.overallSelection.pick?.symbol)
        assertEquals(0.2, dashboard.overallSelection.pick?.scoreBreakdown?.total)
        assertEquals(1, dashboard.overallSelection.pick?.newsEvidence?.size)
        assertEquals("Reuters", dashboard.overallSelection.pick?.newsEvidence?.firstOrNull()?.provider)
        assertEquals(0.02, dashboard.overallSelection.pick?.scoreBreakdown?.signalAlignment)
        assertEquals(ThesisMonitorStatus.WATCH, dashboard.overallSelection.pick?.thesisMonitor?.status)
        assertEquals(5, dashboard.overallSelection.pick?.thesisMonitor?.signals?.size)
    }

    @Test
    fun parser_maps_history_contract() {
        val history = parser.parseHistory(sampleHistoryJson)

        assertEquals(1, history.entries.size)
        assertEquals("2026-W12", history.entries.first().weekId)
        assertEquals(SelectionStatus.PICKED, history.entries.first().status)
        assertNotNull(history.entries.first().loggedAt)
    }

    @Test
    fun parser_maps_legacy_dashboard_contract() {
        val dashboard = parser.parseDashboard(legacyDashboardJson)

        assertEquals(1, dashboard.schemaVersion)
        assertEquals(SelectionStatus.PICKED, dashboard.overallSelection.status)
        assertEquals("GOOGL", dashboard.overallSelection.pick?.symbol)
        assertEquals(SelectionStatus.PICKED, dashboard.riskSelections.getValue("low").status)
    }

    @Test
    fun parser_maps_legacy_history_contract() {
        val history = parser.parseHistory(legacyHistoryJson)

        assertEquals(1, history.schemaVersion)
        assertEquals(2, history.entries.size)
        assertEquals(SelectionStatus.PICKED, history.entries.first().status)
    }

    @Test
    fun weekly_change_summary_compares_against_previous_week() {
        val dashboard = parser.parseDashboard(sampleDashboardJson)
        val history = parser.parseHistory(sampleHistoryWithPreviousWeekJson)

        val summary = buildWeeklyChangeSummary(dashboard, history.entries)

        assertNotNull(summary)
        assertEquals("MSFT replaced AAPL", summary?.title)
        assertEquals(0.05, summary?.scoreDelta ?: 0.0, 0.0001)
        assertEquals(0.12, summary?.confidenceDelta ?: 0.0, 0.0001)
    }
}

internal const val sampleDashboardJson = """
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
    "skipped_symbols": 2
  },
  "selection_thresholds": {
    "overall_score": 0.1,
    "minimum_confidence": 0.55,
    "risk_scores": {
      "low": 0.08,
      "medium": 0.1,
      "high": 0.12
    }
  },
  "overall_selection": {
    "status": "picked",
    "status_reason": "MSFT cleared the thresholds.",
    "threshold_score": 0.1,
    "threshold_confidence": 0.55,
    "pick": {
      "symbol": "MSFT",
      "company_name": "Microsoft Corporation",
      "risk": "low",
      "model_score": 0.2,
      "confidence_score": 0.82,
      "confidence_label": "high",
      "price_as_of": "2026-03-14",
      "news_as_of": "2026-03-14",
      "article_count": 4,
      "metrics": {
        "momentum_5d": 0.04,
        "daily_volatility": 0.01,
        "news_sentiment": 0.72,
        "raw_news_sentiment": 0.3
      },
      "score_breakdown": {
        "momentum": 0.25,
        "short_momentum": 0.08,
        "medium_momentum": 0.09,
        "trend_quality": 0.06,
        "volume_confirmation": 0.02,
        "volatility_penalty": -0.08,
        "daily_volatility_penalty": -0.03,
        "downside_penalty": -0.02,
        "drawdown_penalty": -0.01,
        "news_adjustment": 0.03,
        "signal_alignment": 0.02,
        "technical_total": 0.17,
        "total": 0.2
      },
      "reasons": [
        "reason"
      ],
      "news_evidence": [
        {
          "title": "Microsoft signs a major enterprise AI agreement",
          "provider": "Reuters",
          "url": "https://www.reuters.com/example",
          "published_at": "2026-03-14T10:00:00Z",
          "relevance_score": 0.91,
          "sentiment": 0.42
        }
      ],
      "thesis_monitor": {
        "status": "watch",
        "headline": "The thesis needs watching",
        "summary": "The pick still qualified, but the margin of safety is not wide.",
        "alerts": [
          "The pick still qualified, but the margin of safety is not wide."
        ],
        "signals": [
          {
            "label": "Momentum",
            "state": "positive",
            "value": "+4.0% 5D • +9.0% 20D",
            "detail": "Price action is still supporting the release thesis."
          },
          {
            "label": "Volatility",
            "state": "positive",
            "value": "1.0% daily • 2.0% drawdown",
            "detail": "Volatility and drawdown remain controlled for this release."
          },
          {
            "label": "News",
            "state": "watch",
            "value": "0.72 sentiment • 4 articles",
            "detail": "News flow is mostly neutral, so the thesis needs technical confirmation to stay intact."
          },
          {
            "label": "Margin",
            "state": "watch",
            "value": "+0.10 score • +0.27 conf",
            "detail": "The pick still qualified, but the margin of safety is not wide."
          },
          {
            "label": "Freshness",
            "state": "positive",
            "value": "price 0d • news 0d",
            "detail": "Price and news timestamps are recent enough for a live monitoring read."
          }
        ]
      }
    }
  },
  "risk_selections": {
    "low": {
      "status": "picked",
      "status_reason": "MSFT qualified.",
      "threshold_score": 0.08,
      "threshold_confidence": 0.55,
      "pick": {
        "symbol": "MSFT",
        "company_name": "Microsoft Corporation",
        "risk": "low",
        "model_score": 0.2,
        "confidence_score": 0.82,
        "confidence_label": "high",
        "price_as_of": "2026-03-14",
        "news_as_of": "2026-03-14",
        "article_count": 4,
        "metrics": {
          "momentum_5d": 0.04,
          "daily_volatility": 0.01,
          "news_sentiment": 0.72,
          "raw_news_sentiment": 0.3
        },
        "score_breakdown": {
          "momentum": 0.25,
          "volatility_penalty": -0.08,
          "news_adjustment": 0.03,
          "total": 0.2
        },
        "reasons": [
          "reason"
        ]
      }
    },
    "medium": {
      "status": "picked",
      "status_reason": "CSCO qualified.",
      "threshold_score": 0.1,
      "threshold_confidence": 0.55,
      "pick": {
        "symbol": "CSCO",
        "company_name": "Cisco Systems Inc.",
        "risk": "medium",
        "model_score": 0.19,
        "confidence_score": 0.78,
        "confidence_label": "high",
        "price_as_of": "2026-03-14",
        "news_as_of": "2026-03-14",
        "article_count": 5,
        "metrics": {
          "momentum_5d": 0.05,
          "daily_volatility": 0.015,
          "news_sentiment": 0.7,
          "raw_news_sentiment": 0.25
        },
        "score_breakdown": {
          "momentum": 0.27,
          "volatility_penalty": -0.12,
          "news_adjustment": 0.04,
          "total": 0.19
        },
        "reasons": [
          "reason"
        ]
      }
    },
    "high": {
      "status": "no_pick",
      "status_reason": "No high-risk release this week.",
      "threshold_score": 0.12,
      "threshold_confidence": 0.55,
      "pick": null,
      "best_candidate": {
        "symbol": "NVDA",
        "company_name": "NVIDIA Corporation",
        "risk": "high",
        "model_score": 0.11,
        "confidence_score": 0.49,
        "confidence_label": "low"
      }
    }
  }
}
"""

internal const val sampleHistoryJson = """
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
      "status_reason": "MSFT cleared the thresholds.",
      "symbol": "MSFT",
      "company_name": "Microsoft Corporation",
      "risk": "low",
      "model_score": 0.2,
      "confidence_score": 0.82,
      "confidence_label": "high",
      "data_as_of": "2026-03-14"
    }
  ]
}
"""

internal const val sampleHistoryWithPreviousWeekJson = """
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
      "status_reason": "MSFT cleared the thresholds.",
      "symbol": "MSFT",
      "company_name": "Microsoft Corporation",
      "risk": "low",
      "model_score": 0.2,
      "confidence_score": 0.82,
      "confidence_label": "high",
      "data_as_of": "2026-03-14"
    },
    {
      "week_id": "2026-W11",
      "week_start": "2026-03-09",
      "week_end": "2026-03-13",
      "week_label": "Mar 9 - 13, 2026",
      "logged_at": "2026-03-09T12:00:00Z",
      "status": "picked",
      "status_reason": "AAPL cleared the thresholds.",
      "symbol": "AAPL",
      "company_name": "Apple Inc.",
      "risk": "medium",
      "model_score": 0.15,
      "confidence_score": 0.70,
      "confidence_label": "medium",
      "data_as_of": "2026-03-13"
    }
  ]
}
"""

internal const val legacyDashboardJson = """
{
  "low": {
    "symbol": "MSFT",
    "company_name": "Microsoft Corporation",
    "week_start": "2025-11-26",
    "week_end": "2025-12-03",
    "reasons": [
      "Legacy low risk pick."
    ],
    "score": 0.04,
    "risk": "low",
    "model_version": "v1.1"
  },
  "medium": {
    "symbol": "AAPL",
    "company_name": "Apple Inc.",
    "week_start": "2025-11-26",
    "week_end": "2025-12-03",
    "reasons": [
      "Legacy medium risk pick."
    ],
    "score": 0.06,
    "risk": "medium",
    "model_version": "v1.1"
  },
  "high": {
    "symbol": "GOOGL",
    "company_name": "Alphabet Inc.",
    "week_start": "2025-11-26",
    "week_end": "2025-12-03",
    "reasons": [
      "Legacy high risk pick."
    ],
    "score": 0.09,
    "risk": "high",
    "model_version": "v1.1"
  }
}
"""

internal const val legacyHistoryJson = """
[
  {
    "logged_at": "2025-11-25",
    "symbol": "GOOGL",
    "company_name": "Alphabet Inc.",
    "week_start": "2025-11-25",
    "week_end": "2025-12-02",
    "score": 0.085,
    "risk": "high",
    "model_version": "v1.0"
  },
  {
    "logged_at": "2025-11-26",
    "symbol": "MRK",
    "company_name": "Merck & Co. Inc.",
    "week_start": "2025-11-26",
    "week_end": "2025-12-03",
    "score": 0.103,
    "risk": "high",
    "model_version": "v1.0"
  }
]
"""
