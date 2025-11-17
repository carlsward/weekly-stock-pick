package com.example.weeklystockapp2

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun WeeklyPickScreen(
    pick: WeeklyPick,
    availableRisks: List<String>,
    onRiskSelected: (String) -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        contentAlignment = Alignment.TopCenter
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {

                // Risk-väljare om vi har några risknivåer
                if (availableRisks.isNotEmpty()) {
                    val ordered = listOf("low", "medium", "high")
                    val displayOrder = ordered.filter { it in availableRisks } +
                            availableRisks.filter { it !in ordered }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        displayOrder.forEach { riskKey ->
                            Text(
                                text = riskKey.replaceFirstChar { it.uppercase() },
                                modifier = Modifier
                                    .clickable { onRiskSelected(riskKey) }
                                    .padding(vertical = 4.dp),
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                }

                Text(
                    text = "Best stock to hold this week",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = "${pick.symbol} – ${pick.companyName}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Period: ${pick.weekStart} till ${pick.weekEnd}",
                    style = MaterialTheme.typography.bodyMedium
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = "Risknivå: ${pick.risk}",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold
                )

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "Why this stock:",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )

                Spacer(modifier = Modifier.height(8.dp))

                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(pick.reasons) { reason ->
                        Text(
                            text = "• $reason",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }
        }
    }
}
