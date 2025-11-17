package com.example.weeklystockapp2

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.lifecycle.lifecycleScope
import com.example.weeklystockapp2.ui.theme.WeeklyStockApp2Theme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.URL

class MainActivity : ComponentActivity() {

    private val jsonUrl =
        "https://raw.githubusercontent.com/carlsward/weekly-stock-pick/main/backend/current_pick.json"

    private val riskPicksUrl =
        "https://raw.githubusercontent.com/carlsward/weekly-stock-pick/main/backend/risk_picks.json"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        var pickState: MutableState<WeeklyPick?>? = null
        var riskMapState: MutableState<Map<String, WeeklyPick>>? = null

        setContent {
            WeeklyStockApp2Theme {
                val state = remember { mutableStateOf<WeeklyPick?>(null) }
                val riskMap = remember { mutableStateOf<Map<String, WeeklyPick>>(emptyMap()) }

                pickState = state
                riskMapState = riskMap

                Surface(color = MaterialTheme.colorScheme.background) {
                    val currentPick = state.value
                    if (currentPick != null) {
                        WeeklyPickScreen(
                            pick = currentPick,
                            availableRisks = riskMap.value.keys.toList()
                        ) { riskKey ->
                            val candidate = riskMap.value[riskKey]
                            if (candidate != null) {
                                state.value = candidate
                            }
                        }
                    } else {
                        LoadingView()
                    }
                }
            }
        }

        // Hämta i bakgrunden
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Försök använda risk_picks.json först
                val jsonText = URL(riskPicksUrl).readText()
                val riskMap = parseRiskPicks(jsonText)

                withContext(Dispatchers.Main) {
                    riskMapState?.value = riskMap

                    val defaultPick =
                        riskMap["medium"] ?: riskMap["low"] ?: riskMap.values.firstOrNull()

                    if (defaultPick != null) {
                        pickState?.value = defaultPick
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()

                // Fallback: gammal current_pick.json
                try {
                    val jsonText = URL(jsonUrl).readText()
                    val pick = parseWeeklyPick(jsonText)
                    withContext(Dispatchers.Main) {
                        pickState?.value = pick
                    }
                } catch (e2: Exception) {
                    e2.printStackTrace()
                }
            }
        }
    }

    private fun parseWeeklyPick(jsonText: String): WeeklyPick {
        val obj = JSONObject(jsonText)

        val reasonsJson = obj.getJSONArray("reasons")
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

            val reasonsJson = pickObj.getJSONArray("reasons")
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
}

@Composable
fun LoadingView() {
    Text(
        text = "Loading weekly pick...",
        style = MaterialTheme.typography.bodyLarge
    )
}
