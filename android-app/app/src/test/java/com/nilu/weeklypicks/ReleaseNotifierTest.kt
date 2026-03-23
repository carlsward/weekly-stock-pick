package com.nilu.weeklypicks

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ReleaseNotifierTest {
    private val parser = ContractParser()

    @Test
    fun notification_plan_emits_new_release_once_for_fresh_network_content() {
        val content = buildContent(
            source = DataSource.NETWORK,
            freshness = Freshness.FRESH
        )

        val plan = buildReleaseNotificationPlan(
            content = content,
            lastReleaseWeek = "2026-W11",
            lastThesisAlertKey = null,
            lastStaleAlertWeek = null
        )

        assertEquals(1, plan.events.size)
        assertTrue(plan.events.first().title.startsWith("New weekly release"))
        assertEquals("2026-W12", plan.releaseWeekToRecord)
    }

    @Test
    fun notification_plan_emits_thesis_alert_when_same_week_monitor_worsens() {
        val content = buildContent(
            source = DataSource.NETWORK,
            freshness = Freshness.FRESH
        )

        val plan = buildReleaseNotificationPlan(
            content = content,
            lastReleaseWeek = "2026-W12",
            lastThesisAlertKey = null,
            lastStaleAlertWeek = null
        )

        assertEquals(1, plan.events.size)
        assertTrue(plan.events.first().title.startsWith("Thesis watch"))
        assertEquals("2026-W12:MSFT:WATCH", plan.thesisAlertKeyToRecord)
    }

    @Test
    fun notification_plan_skips_alerts_for_cached_content() {
        val content = buildContent(
            source = DataSource.CACHE,
            freshness = Freshness.STALE
        )

        val plan = buildReleaseNotificationPlan(
            content = content,
            lastReleaseWeek = null,
            lastThesisAlertKey = null,
            lastStaleAlertWeek = null
        )

        assertTrue(plan.events.isEmpty())
    }

    @Test
    fun notification_plan_emits_stale_warning_once_per_week() {
        val content = buildContent(
            source = DataSource.NETWORK,
            freshness = Freshness.STALE
        )

        val plan = buildReleaseNotificationPlan(
            content = content,
            lastReleaseWeek = "2026-W12",
            lastThesisAlertKey = "2026-W12:MSFT:WATCH",
            lastStaleAlertWeek = null
        )

        assertEquals(1, plan.events.size)
        assertEquals("Dashboard data is stale", plan.events.first().title)
        assertEquals("2026-W12", plan.staleWeekToRecord)
    }

    private fun buildContent(
        source: DataSource,
        freshness: Freshness
    ): DashboardContent {
        val dashboard = parser.parseDashboard(sampleDashboardJson)
        val history = parser.parseHistory(sampleHistoryWithPreviousWeekJson)
        return DashboardContent(
            dashboard = dashboard,
            historyEntries = history.entries,
            historyMessage = null,
            selectedRisk = "low",
            isRefreshing = false,
            source = source,
            warningMessage = null,
            freshness = freshness,
            weeklyChange = buildWeeklyChangeSummary(dashboard, history.entries)
        )
    }
}
