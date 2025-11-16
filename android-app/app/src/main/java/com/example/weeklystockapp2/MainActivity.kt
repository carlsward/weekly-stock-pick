package com.example.weeklystockapp2


import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.example.weeklystockapp2.ui.theme.WeeklyStockApp2Theme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)


        val mockPick = WeeklyPick(
            symbol = "AAPL",
            companyName = "Apple Inc.",
            weekStart = "2025-11-17",
            weekEnd = "2025-11-24",
            reasons = listOf(
                "Starkt nyhetsflöde efter senaste kvartalsrapporten med positiv analytikerreaktion.",
                "Stabil intjäning och kassaflöde med historiskt lägre volatilitet än många tech-peers.",
                "Teknisk bild visar positiv kortsiktig trend över sitt 20-dagars glidande medelvärde."
            )
        )

        setContent {
            WeeklyStockApp2Theme {           // eller det namn som står i Theme.kt
                Surface(color = MaterialTheme.colorScheme.background) {
                    WeeklyPickScreen(pick = mockPick)
                }
            }
        }

    }
}