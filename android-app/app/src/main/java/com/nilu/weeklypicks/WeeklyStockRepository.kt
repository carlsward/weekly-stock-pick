package com.nilu.weeklypicks

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.util.concurrent.TimeUnit

sealed class RepositoryResult {
    data class Success(val snapshot: DashboardSnapshot) : RepositoryResult()
    data class Failure(val message: String) : RepositoryResult()
}

class WeeklyStockRepository(
    private val dashboardUrl: String,
    private val historyUrl: String,
    private val localStore: LocalStore,
    private val parser: ContractParser = ContractParser(),
    private val client: OkHttpClient = defaultHttpClient()
) {
    fun loadCachedSnapshot(): DashboardSnapshot? {
        val cachedDashboard = localStore.readDashboardCache() ?: return null
        val dashboard = runCatching { parser.parseDashboard(cachedDashboard) }.getOrNull() ?: return null
        val history = localStore.readHistoryCache()
            ?.let { raw -> runCatching { parser.parseHistory(raw) }.getOrNull() }

        return DashboardSnapshot(
            dashboard = dashboard,
            history = history,
            source = DataSource.CACHE,
            warningMessage = "Showing saved data while the app checks for a fresh update.",
            historyMessage = if (history == null) {
                "History will load when a network refresh succeeds."
            } else {
                null
            }
        )
    }

    suspend fun refresh(): RepositoryResult = withContext(Dispatchers.IO) {
        val dashboardNetwork = runCatching {
            val rawDashboard = fetchJson(dashboardUrl)
            val parsedDashboard = parser.parseDashboard(rawDashboard)
            localStore.writeDashboardCache(rawDashboard)
            parsedDashboard
        }

        val dashboard = dashboardNetwork.getOrNull()
        val dashboardSource = if (dashboard != null) {
            DataSource.NETWORK
        } else {
            DataSource.CACHE
        }

        val resolvedDashboard = dashboard ?: loadCachedSnapshot()?.dashboard
        if (resolvedDashboard == null) {
            val message = dashboardNetwork.exceptionOrNull()?.message
                ?: "Could not load weekly picks."
            return@withContext RepositoryResult.Failure(
                "$message Check your connection and try again."
            )
        }

        val historyNetwork = runCatching {
            val rawHistory = fetchJson(historyUrl)
            val parsedHistory = parser.parseHistory(rawHistory)
            localStore.writeHistoryCache(rawHistory)
            parsedHistory
        }

        val history = historyNetwork.getOrNull()
            ?: localStore.readHistoryCache()?.let { raw ->
                runCatching { parser.parseHistory(raw) }.getOrNull()
            }

        val warningMessage = if (dashboardSource == DataSource.CACHE) {
            "Could not refresh from the network. Showing the last saved dashboard."
        } else {
            null
        }

        val historyMessage = when {
            historyNetwork.isSuccess -> null
            history != null -> "History could not be refreshed. Showing the last saved history."
            else -> "History is unavailable right now."
        }

        RepositoryResult.Success(
            DashboardSnapshot(
                dashboard = resolvedDashboard,
                history = history,
                source = dashboardSource,
                warningMessage = warningMessage,
                historyMessage = historyMessage
            )
        )
    }

    private fun fetchJson(url: String): String {
        val request = Request.Builder()
            .url(url)
            .header("Accept", "application/json")
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("Server returned ${response.code}")
            }

            val body = response.body?.string()
            if (body.isNullOrBlank()) {
                throw IOException("Server returned an empty response")
            }
            return body
        }
    }
}

private fun defaultHttpClient(): OkHttpClient {
    return OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .callTimeout(15, TimeUnit.SECONDS)
        .build()
}
