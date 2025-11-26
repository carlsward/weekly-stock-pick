package com.example.weeklystockapp2

import android.app.Application
import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val jsonUrl =
        "https://raw.githubusercontent.com/carlsward/weekly-stock-pick/main/backend/current_pick.json"

    private val riskPicksUrl =
        "https://raw.githubusercontent.com/carlsward/weekly-stock-pick/main/backend/risk_picks.json"

    private val historyUrl =
        "https://raw.githubusercontent.com/carlsward/weekly-stock-pick/main/backend/history.json"

    private val prefsName = "weekly_picks_prefs"
    private val keyDefaultRisk = "default_risk"
    private val keyCachedRiskPicks = "cached_risk_picks"
    private val keyCachedSinglePick = "cached_current_pick"
    private val keyCachedHistory = "cached_history"

    private val prefs =
        application.getSharedPreferences(prefsName, Context.MODE_PRIVATE)

    var uiState: UiState by mutableStateOf(UiState.Loading)
        private set

    var historyState: HistoryUiState by mutableStateOf(HistoryUiState.Loading)
        private set

    private var riskMap: Map<String, WeeklyPick> = emptyMap()

    init {
        loadData()
        loadHistory()
    }

    fun refreshAll() {
        loadData()
        loadHistory()
    }

    private fun fetchUrlWithTimeout(urlStr: String, timeoutMs: Int = 7000, retries: Int = 2): String {
        var lastError: Exception? = null
        repeat(retries) { attempt ->
            try {
                val conn = URL(urlStr).openConnection() as HttpURLConnection
                conn.connectTimeout = timeoutMs
                conn.readTimeout = timeoutMs
                conn.requestMethod = "GET"
                conn.instanceFollowRedirects = true
                conn.connect()
                conn.inputStream.bufferedReader().use { reader ->
                    return reader.readText()
                }
            } catch (e: Exception) {
                lastError = e
            }
        }
        throw lastError ?: RuntimeException("Failed to fetch $urlStr")
    }

    fun onRiskSelected(riskKey: String) {
        val candidate = riskMap[riskKey] ?: return
        prefs.edit {
            putString(keyDefaultRisk, riskKey)
        }
        uiState = UiState.Content(
            pick = candidate,
            availableRisks = riskMap.keys.toList()
        )
    }

    private fun loadData() {
        viewModelScope.launch(Dispatchers.IO) {
            val savedRisk = prefs.getString(keyDefaultRisk, null)

            // 1) Försök hämta risk_picks.json från nätet
            try {
                val jsonText = fetchUrlWithTimeout(riskPicksUrl)

                // Cacha rå JSON
                prefs.edit {
                    putString(keyCachedRiskPicks, jsonText)
                }

                val loadedRiskMap = parseRiskPicks(jsonText)
                riskMap = loadedRiskMap

                val defaultPick =
                    if (savedRisk != null && loadedRiskMap.containsKey(savedRisk)) {
                        loadedRiskMap[savedRisk]
                    } else {
                        loadedRiskMap["medium"]
                            ?: loadedRiskMap["low"]
                            ?: loadedRiskMap.values.firstOrNull()
                    }

                if (defaultPick != null) {
                    withContext(Dispatchers.Main) {
                        uiState = UiState.Content(
                            pick = defaultPick,
                            availableRisks = loadedRiskMap.keys.toList()
                        )
                    }
                    return@launch
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }

            // 2) Försök cached risk_picks.json
            val cachedRisk = prefs.getString(keyCachedRiskPicks, null)
            if (cachedRisk != null) {
                try {
                    val loadedRiskMap = parseRiskPicks(cachedRisk)
                    riskMap = loadedRiskMap

                    val defaultPick =
                        if (savedRisk != null && loadedRiskMap.containsKey(savedRisk)) {
                            loadedRiskMap[savedRisk]
                        } else {
                            loadedRiskMap["medium"]
                                ?: loadedRiskMap["low"]
                                ?: loadedRiskMap.values.firstOrNull()
                        }

                    if (defaultPick != null) {
                        withContext(Dispatchers.Main) {
                            uiState = UiState.Content(
                                pick = defaultPick,
                                availableRisks = loadedRiskMap.keys.toList()
                            )
                        }
                        return@launch
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            // 3) Försök nätverks-hämta current_pick.json
            try {
                val jsonText = fetchUrlWithTimeout(jsonUrl)
                prefs.edit {
                    putString(keyCachedSinglePick, jsonText)
                }
                val pick = parseWeeklyPick(jsonText)

                withContext(Dispatchers.Main) {
                    uiState = UiState.Content(
                        pick = pick,
                        availableRisks = emptyList()
                    )
                }
                return@launch
            } catch (e: Exception) {
                e.printStackTrace()
            }

            // 4) Sista fallback: cached current_pick.json
            val cachedSingle = prefs.getString(keyCachedSinglePick, null)
            if (cachedSingle != null) {
                try {
                    val pick = parseWeeklyPick(cachedSingle)
                    withContext(Dispatchers.Main) {
                        uiState = UiState.Content(
                            pick = pick,
                            availableRisks = emptyList()
                        )
                    }
                    return@launch
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            // Om allt ovan misslyckas
            withContext(Dispatchers.Main) {
                uiState = UiState.Error("Could not load weekly pick.")
            }
        }
    }

    private fun loadHistory() {
        viewModelScope.launch(Dispatchers.IO) {
            // 1) Försök hämta från nätet
            try {
                val jsonText = fetchUrlWithTimeout(historyUrl)
                prefs.edit {
                    putString(keyCachedHistory, jsonText)
                }

                val entries = parseHistory(jsonText)
                withContext(Dispatchers.Main) {
                    historyState = HistoryUiState.Content(entries)
                }
                return@launch
            } catch (e: Exception) {
                e.printStackTrace()
            }

            // 2) Fallback: cached history
            val cached = prefs.getString(keyCachedHistory, null)
            if (cached != null) {
                try {
                    val entries = parseHistory(cached)
                    withContext(Dispatchers.Main) {
                        historyState = HistoryUiState.Content(entries)
                    }
                    return@launch
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            // Om allt misslyckas
            withContext(Dispatchers.Main) {
                historyState = HistoryUiState.Error("Could not load history.")
            }
        }
    }

    private fun parseWeeklyPick(jsonText: String): WeeklyPick {
        val obj = JSONObject(jsonText)

        val reasonsJson = obj.optJSONArray("reasons") ?: JSONArray()
        val reasons = mutableListOf<String>()
        for (i in 0 until reasonsJson.length()) {
            reasons.add(reasonsJson.getString(i))
        }

        return WeeklyPick(
            symbol = obj.getString("symbol"),
            companyName = obj.getString("company_name"),
            weekStart = obj.getString("week_start"),
            weekEnd = obj.getString("week_end"),
            reasons = reasons,
            score = obj.optDouble("score", Double.NaN),
            risk = obj.optString("risk", "unknown")
        )
    }

    private fun parseRiskPicks(jsonText: String): Map<String, WeeklyPick> {
        val obj = JSONObject(jsonText)
        val result = mutableMapOf<String, WeeklyPick>()

        val keys = obj.keys()
        while (keys.hasNext()) {
            val riskKey = keys.next()
            val pickObj = obj.getJSONObject(riskKey)

            val reasonsJson = pickObj.optJSONArray("reasons") ?: JSONArray()
            val reasons = mutableListOf<String>()
            for (i in 0 until reasonsJson.length()) {
                reasons.add(reasonsJson.getString(i))
            }

            val pick = WeeklyPick(
                symbol = pickObj.getString("symbol"),
                companyName = pickObj.getString("company_name"),
                weekStart = pickObj.getString("week_start"),
                weekEnd = pickObj.getString("week_end"),
                reasons = reasons,
                score = pickObj.optDouble("score", Double.NaN),
                risk = pickObj.optString("risk", riskKey)
            )

            result[riskKey] = pick
        }

        return result
    }

    private fun parseHistory(jsonText: String): List<HistoryEntry> {
        val arr = JSONArray(jsonText)
        val list = mutableListOf<HistoryEntry>()
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            val entry = HistoryEntry(
                loggedAt = obj.optString("logged_at", ""),
                symbol = obj.optString("symbol", ""),
                companyName = obj.optString("company_name", ""),
                weekStart = obj.optString("week_start", ""),
                weekEnd = obj.optString("week_end", ""),
                score = if (obj.has("score")) obj.optDouble("score") else null,
                risk = obj.optString("risk", "unknown")
            )
            list.add(entry)
        }
        return list
    }
}

sealed class UiState {
    object Loading : UiState()
    data class Content(
        val pick: WeeklyPick,
        val availableRisks: List<String>
    ) : UiState()

    data class Error(val message: String) : UiState()
}

sealed class HistoryUiState {
    object Loading : HistoryUiState()
    data class Content(val entries: List<HistoryEntry>) : HistoryUiState()
    data class Error(val message: String) : HistoryUiState()
}
