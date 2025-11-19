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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

@Composable
fun WeeklyPickScreen(
    pick: WeeklyPick,
    availableRisks: List<String>,
    onRiskSelected: (String) -> Unit
) {
    var showInfoDialog by remember { mutableStateOf(false) }
    var showFullSummary by remember { mutableStateOf(false) }

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

                // Risk selector
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

                // Title + info icon
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Best stock to hold this week",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.weight(1f)
                    )

                    IconButton(onClick = { showInfoDialog = true }) {
                        Icon(
                            imageVector = Icons.Filled.Info,
                            contentDescription = "About the model"
                        )
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = "${pick.symbol} – ${pick.companyName}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Period: ${pick.weekStart} to ${pick.weekEnd}",
                    style = MaterialTheme.typography.bodyMedium
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = "Risk level: ${pick.risk}",
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

                // Base reasons + summary reason
                val reasons = pick.reasons
                val baseReasons = if (reasons.size >= 2) reasons.dropLast(1) else reasons
                val summaryReason = if (reasons.size >= 2) reasons.last() else null

                Column(
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    // Normal bullet points
                    baseReasons.forEach { reason ->
                        Text(
                            text = "• $reason",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }

                    // Summary with "Show more"
                    summaryReason?.let { summary ->
                        Text(
                            text = "• $summary",
                            style = MaterialTheme.typography.bodyMedium,
                            maxLines = if (showFullSummary) Int.MAX_VALUE else 4,
                            overflow = TextOverflow.Ellipsis
                        )

                        Text(
                            text = if (showFullSummary) "Show less" else "Show more",
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier
                                .padding(top = 2.dp)
                                .clickable { showFullSummary = !showFullSummary }
                        )
                    }
                }
            }
        }
    }

    if (showInfoDialog) {
        AlertDialog(
            onDismissRequest = { showInfoDialog = false },
            title = {
                Text(
                    text = "How the weekly stock is selected",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("The model currently evaluates three key aspects for each stock:")
                    Text("• Momentum: price change over the last 5 trading days.")
                    Text("• Volatility: how much the price moves day to day.")
                    Text("• News sentiment: AI analysis of relevant news articles.")
                    Text(
                        text = "Score = momentum − volatility, adjusted with news sentiment. " +
                                "A higher score is better, favoring stocks that rise steadily, " +
                                "do not fluctuate too much, and have a positive news outlook."
                    )
                    Text(
                        text = "Risk levels:\n" +
                                "• Low: low volatility\n" +
                                "• Medium: medium volatility\n" +
                                "• High: high volatility\n\n" +
                                "This model is simplified and should not be considered financial advice."
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showInfoDialog = false }) {
                    Text("Close")
                }
            }
        )
    }
}
