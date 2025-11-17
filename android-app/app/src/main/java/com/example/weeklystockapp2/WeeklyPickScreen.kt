package com.example.weeklystockapp2

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
fun WeeklyPickScreen(pick: WeeklyPick) {
    Box(
        modifier = Modifier.Companion
            .fillMaxSize()
            .padding(16.dp),
        contentAlignment = Alignment.Companion.TopCenter
    ) {
        Card(
            modifier = Modifier.Companion.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(modifier = Modifier.Companion.padding(16.dp)) {
                Text(
                    text = "Best stock to hold this week",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Companion.Bold
                )

                Spacer(modifier = Modifier.Companion.height(12.dp))

                Text(
                    text = "${pick.symbol} – ${pick.companyName}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Companion.SemiBold
                )

                Spacer(modifier = Modifier.Companion.height(8.dp))

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
                    fontWeight = FontWeight.Companion.SemiBold
                )

                Spacer(modifier = Modifier.Companion.height(8.dp))

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