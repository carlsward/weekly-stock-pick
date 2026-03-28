package com.nilu.weeklypicks

import kotlinx.coroutines.test.runTest
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WeeklyStockRepositoryTest {
    @Test
    fun refresh_returns_network_snapshot_and_caches_payloads() = runTest {
        val store = InMemoryLocalStore()
        val server = MockWebServer()
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleDashboardJson))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleThesisMonitorJson))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleHistoryJson))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleTrackRecordJson))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleMonthlyPickJson))
        server.start()

        try {
            val repository = WeeklyStockRepository(
                dashboardUrl = server.url("/risk_picks.json").toString(),
                historyUrl = server.url("/history.json").toString(),
                thesisMonitorUrl = server.url("/thesis_monitor.json").toString(),
                trackRecordUrl = server.url("/track_record.json").toString(),
                monthlyPickUrl = server.url("/monthly_pick.json").toString(),
                localStore = store,
                client = OkHttpClient()
            )

            val result = repository.refresh()

            assertTrue(result is RepositoryResult.Success)
            val snapshot = (result as RepositoryResult.Success).snapshot
            assertEquals(DataSource.NETWORK, snapshot.source)
            assertEquals("MSFT", snapshot.dashboard.overallSelection.pick?.symbol)
            assertEquals(0.19, snapshot.dashboard.overallSelection.pick?.modelScore ?: 0.0, 0.0001)
            assertEquals(1, snapshot.trackRecord?.summary?.closedPicks)
            assertEquals("MSFT", snapshot.monthlyPick?.selection?.pick?.symbol)
            assertNotNull(store.readDashboardCache())
            assertNotNull(store.readHistoryCache())
            assertNotNull(store.readThesisMonitorCache())
            assertNotNull(store.readTrackRecordCache())
            assertNotNull(store.readMonthlyPickCache())
        } finally {
            server.shutdown()
        }
    }

    @Test
    fun refresh_falls_back_to_cache_when_dashboard_request_fails() = runTest {
        val store = InMemoryLocalStore().apply {
            writeDashboardCache(sampleDashboardJson)
            writeHistoryCache(sampleHistoryJson)
        }

        val server = MockWebServer()
        server.enqueue(MockResponse().setResponseCode(500))
        server.enqueue(MockResponse().setResponseCode(500))
        server.enqueue(MockResponse().setResponseCode(500))
        server.enqueue(MockResponse().setResponseCode(500))
        server.start()

        try {
            val repository = WeeklyStockRepository(
                dashboardUrl = server.url("/risk_picks.json").toString(),
                historyUrl = server.url("/history.json").toString(),
                thesisMonitorUrl = server.url("/thesis_monitor.json").toString(),
                trackRecordUrl = server.url("/track_record.json").toString(),
                localStore = store,
                client = OkHttpClient()
            )

            val result = repository.refresh()

            assertTrue(result is RepositoryResult.Success)
            val snapshot = (result as RepositoryResult.Success).snapshot
            assertEquals(DataSource.CACHE, snapshot.source)
            assertEquals("MSFT", snapshot.dashboard.overallSelection.pick?.symbol)
            assertNotNull(snapshot.warningMessage)
        } finally {
            server.shutdown()
        }
    }

    @Test
    fun refresh_does_not_overwrite_dashboard_cache_when_payload_is_invalid() = runTest {
        val store = InMemoryLocalStore().apply {
            writeDashboardCache(sampleDashboardJson)
            writeHistoryCache(sampleHistoryJson)
        }

        val server = MockWebServer()
        server.enqueue(MockResponse().setResponseCode(200).setBody("{\"schema_version\":2"))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleThesisMonitorJson))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleHistoryJson))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleTrackRecordJson))
        server.start()

        try {
            val repository = WeeklyStockRepository(
                dashboardUrl = server.url("/risk_picks.json").toString(),
                historyUrl = server.url("/history.json").toString(),
                thesisMonitorUrl = server.url("/thesis_monitor.json").toString(),
                trackRecordUrl = server.url("/track_record.json").toString(),
                localStore = store,
                client = OkHttpClient()
            )

            val result = repository.refresh()

            assertTrue(result is RepositoryResult.Success)
            val snapshot = (result as RepositoryResult.Success).snapshot
            assertEquals(DataSource.CACHE, snapshot.source)
            assertEquals(sampleDashboardJson, store.readDashboardCache())
            assertEquals("MSFT", snapshot.dashboard.overallSelection.pick?.symbol)
        } finally {
            server.shutdown()
        }
    }

    @Test
    fun refresh_does_not_overwrite_history_cache_when_payload_is_invalid() = runTest {
        val store = InMemoryLocalStore().apply {
            writeDashboardCache(sampleDashboardJson)
            writeHistoryCache(sampleHistoryJson)
        }

        val server = MockWebServer()
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleDashboardJson))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleThesisMonitorJson))
        server.enqueue(MockResponse().setResponseCode(200).setBody("{\"entries\":"))
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleTrackRecordJson))
        server.start()

        try {
            val repository = WeeklyStockRepository(
                dashboardUrl = server.url("/risk_picks.json").toString(),
                historyUrl = server.url("/history.json").toString(),
                thesisMonitorUrl = server.url("/thesis_monitor.json").toString(),
                trackRecordUrl = server.url("/track_record.json").toString(),
                localStore = store,
                client = OkHttpClient()
            )

            val result = repository.refresh()

            assertTrue(result is RepositoryResult.Success)
            val snapshot = (result as RepositoryResult.Success).snapshot
            assertEquals(DataSource.NETWORK, snapshot.source)
            assertEquals(sampleHistoryJson, store.readHistoryCache())
            assertEquals(1, snapshot.history?.entries?.size)
        } finally {
            server.shutdown()
        }
    }
}
