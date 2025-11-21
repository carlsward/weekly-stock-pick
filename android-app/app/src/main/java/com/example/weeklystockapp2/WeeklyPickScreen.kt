package com.example.weeklystockapp2

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.weeklystockapp2.ui.theme.RiskHighColor
import com.example.weeklystockapp2.ui.theme.RiskLowColor
import com.example.weeklystockapp2.ui.theme.RiskMediumColor
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.layout.size



@Composable
fun WeeklyPickScreen(
    pick: WeeklyPick,
    availableRisks: List<String>,
    onRiskSelected: (String) -> Unit
) {
    var showInfoDialog by remember { mutableStateOf(false) }
    var showFullSummary by remember { mutableStateOf(false) }

    // Card in-animation
    var cardVisible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        cardVisible = true
    }
    val cardScale by animateFloatAsState(
        targetValue = if (cardVisible) 1f else 0.95f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "cardScale"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    listOf(
                        MaterialTheme.colorScheme.background,
                        MaterialTheme.colorScheme.surfaceVariant
                    )
                )
            )
            .padding(16.dp),
        contentAlignment = Alignment.TopCenter
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .scale(cardScale),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                // Risk selector
                if (availableRisks.isNotEmpty()) {
                    val ordered = listOf("low", "medium", "high")
                    val displayOrder = ordered.filter { it in availableRisks } +
                            availableRisks.filter { it !in ordered }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        displayOrder.forEach { riskKey ->
                            val isSelected = riskKey.equals(pick.risk, ignoreCase = true)

                            val targetBg = when {
                                isSelected && riskKey.equals("low", ignoreCase = true) ->
                                    RiskLowColor.copy(alpha = 0.18f)
                                isSelected && riskKey.equals("medium", ignoreCase = true) ->
                                    RiskMediumColor.copy(alpha = 0.22f)
                                isSelected && riskKey.equals("high", ignoreCase = true) ->
                                    RiskHighColor.copy(alpha = 0.20f)
                                else -> Color.Transparent
                            }

                            val targetBorder = when {
                                isSelected && riskKey.equals("low", ignoreCase = true) ->
                                    RiskLowColor
                                isSelected && riskKey.equals("medium", ignoreCase = true) ->
                                    RiskMediumColor
                                isSelected && riskKey.equals("high", ignoreCase = true) ->
                                    RiskHighColor
                                else -> MaterialTheme.colorScheme.outline.copy(alpha = 0.4f)
                            }

                            val targetTextColor = when {
                                isSelected && riskKey.equals("low", ignoreCase = true) ->
                                    RiskLowColor
                                isSelected && riskKey.equals("medium", ignoreCase = true) ->
                                    RiskMediumColor
                                isSelected && riskKey.equals("high", ignoreCase = true) ->
                                    RiskHighColor
                                isSelected -> MaterialTheme.colorScheme.primary
                                else -> MaterialTheme.colorScheme.onSurfaceVariant
                            }

                            val bgColor by animateColorAsState(
                                targetValue = targetBg,
                                label = "chipBg"
                            )
                            val borderColor by animateColorAsState(
                                targetValue = targetBorder,
                                label = "chipBorder"
                            )
                            val textColor by animateColorAsState(
                                targetValue = targetTextColor,
                                label = "chipText"
                            )

                            val chipScale by animateFloatAsState(
                                targetValue = if (isSelected) 1.05f else 1f,
                                animationSpec = spring(
                                    dampingRatio = Spring.DampingRatioMediumBouncy,
                                    stiffness = Spring.StiffnessMedium
                                ),
                                label = "chipScale"
                            )

                            Text(
                                text = riskKey.replaceFirstChar { it.uppercase() },
                                modifier = Modifier
                                    .scale(chipScale)
                                    .border(
                                        width = 1.dp,
                                        color = borderColor,
                                        shape = RoundedCornerShape(999.dp)
                                    )
                                    .background(
                                        color = bgColor,
                                        shape = RoundedCornerShape(999.dp)
                                    )
                                    .clickable { onRiskSelected(riskKey) }
                                    .padding(horizontal = 12.dp, vertical = 6.dp),
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
                                color = textColor
                            )
                        }
                    }
                }

                // Titel + info-ikon
                Box(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "Best stock to hold this week",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .align(Alignment.CenterStart)
                            .padding(end = 40.dp),
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                        color = MaterialTheme.colorScheme.onSurface
                    )

                    IconButton(
                        onClick = { showInfoDialog = true },
                        modifier = Modifier.align(Alignment.CenterEnd)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Info,
                            contentDescription = "About the model",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                Spacer(modifier = Modifier.height(4.dp))

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    StockLogoBadge(
                        symbol = pick.symbol,
                        risk = pick.risk
                    )

                    Text(
                        text = pick.companyName,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }



                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = "Period: ${pick.weekStart} to ${pick.weekEnd}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(2.dp))

                val riskColor = when (pick.risk.lowercase()) {
                    "low" -> RiskLowColor
                    "medium" -> RiskMediumColor
                    "high" -> RiskHighColor
                    else -> MaterialTheme.colorScheme.onSurface
                }

                Text(
                    text = "Risk level: ${pick.risk}",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = riskColor
                )

                Spacer(modifier = Modifier.height(8.dp))

                // NYTT: animerad model-score-mätare
                ScoreMeter(score = pick.score)

                Spacer(modifier = Modifier.height(12.dp))
                HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Why this stock:",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface
                )

                // Dela upp orsaker + lång nyhetssammanfattning
                val reasons = pick.reasons
                val baseReasons = if (reasons.size >= 2) reasons.dropLast(1) else reasons
                val summaryReason = if (reasons.size >= 2) reasons.last() else null

                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {

                    baseReasons.forEach { reason ->
                        Text(
                            text = "• $reason",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }

                    summaryReason?.let { summary ->
                        Text(
                            text = "• $summary",
                            style = MaterialTheme.typography.bodyMedium,
                            maxLines = if (showFullSummary) Int.MAX_VALUE else 4,
                            overflow = TextOverflow.Ellipsis,
                            color = MaterialTheme.colorScheme.onSurface
                        )

                        Text(
                            text = if (showFullSummary) "Show less" else "Show more",
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier
                                .padding(top = 2.dp)
                                .clickable { showFullSummary = !showFullSummary },
                            color = MaterialTheme.colorScheme.primary
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


@Composable
private fun StockLogoBadge(
    symbol: String,
    risk: String
) {
    val baseColor = when (risk.lowercase()) {
        "low" -> RiskLowColor
        "medium" -> RiskMediumColor
        "high" -> RiskHighColor
        else -> MaterialTheme.colorScheme.primary
    }

    Box(
        modifier = Modifier
            .size(54.dp) // lite större för längre tickers
            .background(
                color = baseColor.copy(alpha = 0.12f),
                shape = CircleShape
            ),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .size(46.dp)
                .background(
                    color = baseColor,
                    shape = CircleShape
                ),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = symbol.uppercase(),   // hela tickern
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
    }
}


@Composable
private fun ScoreMeter(score: Double) {
    // Hantera NaN och klampa till ett rimligt intervall för visualisering
    val safeScore = if (score.isNaN()) 0.0 else score
    val clamped = safeScore.coerceIn(-0.10, 0.10) // -10% .. +10% ungefär
    // Normalisera till [0,1]
    val normalized = ((clamped + 0.10) / 0.20).toFloat().coerceIn(0f, 1f)

    val animatedFill by animateFloatAsState(
        targetValue = normalized,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioNoBouncy,
            stiffness = Spring.StiffnessMedium
        ),
        label = "scoreFill"
    )

    val gradient = Brush.horizontalGradient(
        listOf(
            RiskLowColor,
            RiskMediumColor,
            RiskHighColor
        )
    )

    val sentimentLabel = when {
        safeScore < -0.02 -> "Bearish"
        safeScore > 0.02 -> "Bullish"
        else -> "Neutral"
    }

    Column(
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Text(
            text = "Model score",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(10.dp)
                .background(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(999.dp)
                )
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(fraction = animatedFill)
                    .background(
                        brush = gradient,
                        shape = RoundedCornerShape(999.dp)
                    )
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = String.format("%.3f", safeScore),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = sentimentLabel,
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
