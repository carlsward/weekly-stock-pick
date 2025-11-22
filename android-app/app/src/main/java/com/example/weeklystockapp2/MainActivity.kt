package com.example.weeklystockapp2

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.weeklystockapp2.ui.theme.WeeklyStockApp2Theme
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import com.example.weeklystockapp2.ui.theme.RiskHighColor
import com.example.weeklystockapp2.ui.theme.RiskLowColor
import com.example.weeklystockapp2.ui.theme.RiskMediumColor
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.example.weeklystockapp2.SplashScreen



class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            WeeklyStockApp2Theme {
                val vm: MainViewModel = viewModel()
                var showSplash by remember { mutableStateOf(true) }

                if (showSplash) {
                    SplashScreen(
                        onFinished = { showSplash = false }
                    )
                } else {
                    Surface(color = MaterialTheme.colorScheme.background) {
                        when (val state = vm.uiState) {
                            is UiState.Loading -> LoadingView()
                            is UiState.Error -> ErrorView(message = state.message)
                            is UiState.Content -> {
                                LazyColumn(
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .padding(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(16.dp)
                                ) {
                                    item {
                                        WeeklyPickScreen(
                                            pick = state.pick,
                                            availableRisks = state.availableRisks,
                                            onRiskSelected = { riskKey ->
                                                vm.onRiskSelected(riskKey)
                                            }
                                        )
                                    }
                                    item {
                                        HistorySection(historyState = vm.historyState)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }
}

@Composable
fun LoadingView() {
    Text(
        text = "Loading weekly pick...",
        style = MaterialTheme.typography.bodyLarge
    )
}

@Composable
fun ErrorView(message: String) {
    Text(
        text = message,
        style = MaterialTheme.typography.bodyLarge,
        color = MaterialTheme.colorScheme.error
    )
}

@Composable
fun HistorySection(historyState: HistoryUiState) {
    when (historyState) {
        is HistoryUiState.Loading -> {
            Text(
                text = "Loading history...",
                style = MaterialTheme.typography.bodySmall
            )
        }

        is HistoryUiState.Error -> {
            Text(
                text = historyState.message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error
            )
        }

        is HistoryUiState.Content -> {
            val entries = historyState.entries
                .sortedBy { it.weekStart }
                .takeLast(5)
                .reversed() // senaste överst

            if (entries.isNotEmpty()) {
                Text(
                    text = "Recent picks",
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(modifier = Modifier.height(4.dp))

                Column(
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    entries.forEach { entry ->
                        HistoryRow(entry)
                        HorizontalDivider()
                    }
                }
            }
        }
    }
}

@Composable
fun HistoryRow(entry: HistoryEntry) {
    val riskColor = when (entry.risk.lowercase()) {
        "low" -> RiskLowColor
        "medium" -> RiskMediumColor
        "high" -> RiskHighColor
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Box(
                modifier = Modifier
                    .width(4.dp)
                    .height(40.dp)
                    .background(
                        color = riskColor,
                        shape = RoundedCornerShape(999.dp)
                    )
            )

            Column {
                Text(
                    text = "${entry.symbol} – ${entry.companyName}",
                    style = MaterialTheme.typography.bodyMedium
                )
                Text(
                    text = "Week: ${entry.weekStart} to ${entry.weekEnd}",
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = "Risk: ${entry.risk}  •  Score: ${entry.score ?: Double.NaN}",
                    style = MaterialTheme.typography.bodySmall,
                    color = riskColor
                )
            }
        }
    }
}

