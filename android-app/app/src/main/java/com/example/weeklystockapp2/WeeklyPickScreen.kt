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
import androidx.compose.ui.unit.dp

@Composable
fun WeeklyPickScreen(
    pick: WeeklyPick,
    availableRisks: List<String>,
    onRiskSelected: (String) -> Unit
) {
    var showInfoDialog by remember { mutableStateOf(false) }
    var showModelDetailsDialog by remember { mutableStateOf(false) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        contentAlignment = Alignment.TopCenter
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp)
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

                    // Titelrad + info-ikon
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
                                contentDescription = "Om modellen"
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

                    Column(
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        pick.reasons.forEach { reason ->
                            Text(
                                text = "• $reason",
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
            }

            // Kort disclaimer under kortet
            Text(
                text = "Informationen i appen är generell och är inte personlig investeringsrådgivning.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
                modifier = Modifier.padding(horizontal = 4.dp)
            )
        }
    }

    // Kort info-dialog
    if (showInfoDialog) {
        AlertDialog(
            onDismissRequest = { showInfoDialog = false },
            title = {
                Text(
                    text = "Hur veckans aktie väljs",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = "Appen väljer varje dag ut en ”bästa aktie att hålla denna vecka” baserat på två typer av signaler:"
                    )
                    Text(
                        text = "1. Prisdata (teknisk modell)\n" +
                                "   • Momentum: prisförändring de senaste 5 handelsdagarna.\n" +
                                "   • Volatilitet: hur mycket kursen svänger dag till dag.\n" +
                                "   Modellen gillar aktier som gått bra, men inte svänger för mycket."
                    )
                    Text(
                        text = "2. Nyhetsdata (AI-modell)\n" +
                                "   • Appen hämtar rubriker och korta texter om varje bolag.\n" +
                                "   • En AI-modell (FinBERT) uppskattar om nyheterna är övervägande positiva eller negativa.\n" +
                                "   • En summariseringsmodell kondenserar de viktigaste punkterna till en kort text som visas i appen."
                    )
                    Text(
                        text = "Dessa delar vägs ihop till en samlad score för varje aktie.\n\n" +
                                "Modellen är förenklad och ska inte ses som personlig investeringsrådgivning."
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showInfoDialog = false }) {
                    Text("Stäng")
                }
            },
            dismissButton = {
                TextButton(
                    onClick = {
                        showInfoDialog = false
                        showModelDetailsDialog = true
                    }
                ) {
                    Text("Läs mer om modellen")
                }
            }
        )
    }

    // Längre dialog: modell + disclaimer
    if (showModelDetailsDialog) {
        AlertDialog(
            onDismissRequest = { showModelDetailsDialog = false },
            title = {
                Text(
                    text = "Om modellen och disclaimer",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = "Modellen bakom appen är byggd för att ge en snabb, automatiserad bedömning " +
                                "av vilka aktier som ser mest attraktiva ut att hålla ungefär en vecka framåt, " +
                                "givet den information som finns just nu."
                    )
                    Text(
                        text = "I dagsläget tittar modellen på:\n\n" +
                                "1. Prisdata (teknisk del)\n" +
                                "   • Momentum: hur mycket aktiekursen förändrats de senaste 5 handelsdagarna.\n" +
                                "   • Volatilitet: hur mycket kursen svänger från dag till dag.\n" +
                                "   En högre momentum och lägre volatilitet ger en bättre teknisk del-score.\n\n" +
                                "2. Nyhetsdata (AI-del)\n" +
                                "   • Appen hämtar de senaste nyheterna om varje bolag från öppna källor.\n" +
                                "   • En språklig AI-modell (FinBERT) räknar ut ett sentimentvärde: negativt, neutralt eller positivt.\n" +
                                "   • En summariseringsmodell kondenserar de viktigaste punkterna till en kort text.\n\n" +
                                "3. Sammanslagen score per aktie\n" +
                                "   • Den tekniska delen (momentum/volatilitet) viktas ihop med nyhetsbetyget.\n" +
                                "   • För varje risknivå (Low/Medium/High) väljs den aktie som har högst score inom sin riskkorg.\n\n" +
                                "Resultatet loggas också i en historik så att du kan se tidigare veckor i appen."
                    )
                    Text(
                        text = "Viktig information:\n\n" +
                                "Denna applikation tillhandahåller automatiskt genererade analyser av en begränsad uppsättning aktier, " +
                                "baserat på historiska kursdata och nyhetsinformation från externa källor.\n\n" +
                                "Algoritmen är förenklad och bygger på antaganden som kan vara ofullständiga eller felaktiga. " +
                                "Det finns inga garantier för att de aktier som lyfts fram kommer att prestera bättre än marknaden " +
                                "eller andra alternativ.\n\n" +
                                "Informationen i appen är inte anpassad efter din ekonomiska situation, riskprofil eller investeringshorisont " +
                                "och utgör därför inte personlig finansiell rådgivning. Historisk avkastning är ingen garanti för framtida avkastning.\n\n" +
                                "Du bör alltid göra din egen bedömning och vid behov tala med en licensierad rådgivare innan du fattar investeringsbeslut. " +
                                "Utvecklaren av appen tar inget ansvar för förluster eller skador som kan uppstå till följd av användning av informationen i appen."
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showModelDetailsDialog = false }) {
                    Text("Stäng")
                }
            }
        )
    }
}
