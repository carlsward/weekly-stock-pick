package com.example.weeklystockapp2

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
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


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        var pickState: MutableState<WeeklyPick?>? = null

        setContent {
            WeeklyStockApp2Theme {
                val state = remember { mutableStateOf<WeeklyPick?>(null) }
                // Spara referensen så vi kan uppdatera den från coroutine
                pickState = state

                Surface(color = MaterialTheme.colorScheme.background) {
                    val pick = state.value
                    if (pick != null) {
                        WeeklyPickScreen(pick = pick)
                    } else {
                        LoadingView()
                    }
                }
            }
        }

        // Hämta JSON i bakgrundstråd
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val jsonText = URL(jsonUrl).readText()
                val pick = parseWeeklyPick(jsonText)

                withContext(Dispatchers.Main) {
                    pickState?.value = pick
                }
            } catch (e: Exception) {
                e.printStackTrace()
                // Här kan du senare lägga till felhantering i UI
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
            reasons = reasons
        )
    }
}

@Composable
fun LoadingView() {
    androidx.compose.material3.Text(
        text = "Loading weekly pick...",
        style = MaterialTheme.typography.bodyLarge
    )
}
