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
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleHistoryJson))
        server.start()

        try {
            val repository = WeeklyStockRepository(
                dashboardUrl = server.url("/risk_picks.json").toString(),
                historyUrl = server.url("/history.json").toString(),
                localStore = store,
                client = OkHttpClient()
            )

            val result = repository.refresh()

            assertTrue(result is RepositoryResult.Success)
            val snapshot = (result as RepositoryResult.Success).snapshot
            assertEquals(DataSource.NETWORK, snapshot.source)
            assertEquals("MSFT", snapshot.dashboard.overallSelection.pick?.symbol)
            assertNotNull(store.readDashboardCache())
            assertNotNull(store.readHistoryCache())
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
        server.start()

        try {
            val repository = WeeklyStockRepository(
                dashboardUrl = server.url("/risk_picks.json").toString(),
                historyUrl = server.url("/history.json").toString(),
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
        server.enqueue(MockResponse().setResponseCode(200).setBody(sampleHistoryJson))
        server.start()

        try {
            val repository = WeeklyStockRepository(
                dashboardUrl = server.url("/risk_picks.json").toString(),
                historyUrl = server.url("/history.json").toString(),
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
        server.enqueue(MockResponse().setResponseCode(200).setBody("{\"entries\":"))
        server.start()

        try {
            val repository = WeeklyStockRepository(
                dashboardUrl = server.url("/risk_picks.json").toString(),
                historyUrl = server.url("/history.json").toString(),
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
