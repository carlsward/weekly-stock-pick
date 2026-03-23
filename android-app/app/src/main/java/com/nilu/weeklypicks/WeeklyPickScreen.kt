@file:OptIn(ExperimentalLayoutApi::class)

package com.nilu.weeklypicks

import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.nilu.weeklypicks.ui.theme.RiskHighColor
import com.nilu.weeklypicks.ui.theme.RiskLowColor
import com.nilu.weeklypicks.ui.theme.RiskMediumColor
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale
import kotlin.math.abs

@Composable
fun WeeklyPickScreen(
    content: DashboardContent,
    onRiskSelected: (String) -> Unit,
    onRefresh: () -> Unit
) {
    var showInfoDialog by remember { mutableStateOf(false) }
    var showFullSummary by rememberSaveable(content.selectedRisk) { mutableStateOf(false) }
    val selectedSelection = content.selectedSelection
    val overallPick = content.dashboard.overallSelection.pick

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(28.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        listOf(
                            MaterialTheme.colorScheme.surface,
                            MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
                        )
                    )
                )
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            HeaderRow(
                isRefreshing = content.isRefreshing,
                onRefresh = onRefresh,
                onShowInfo = { showInfoDialog = true }
            )

            Text(
                text = content.dashboard.marketContext.weekLabel,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = "Last updated ${formatInstant(content.dashboard.generatedAt)} • Data as of ${formatDate(content.dashboard.dataAsOf)}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "Source: ${if (content.source == DataSource.NETWORK) "live network" else "saved cache"}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            if (content.freshness == Freshness.STALE) {
                StatusBanner(
                    title = "This dashboard is stale",
                    message = "The saved data is older than the expected refresh window. Treat it as reference only.",
                    accent = RiskHighColor
                )
            }

            content.warningMessage?.let { warning ->
                StatusBanner(
                    title = "Using saved data",
                    message = warning,
                    accent = RiskMediumColor
                )
            }

            overallPick?.let { pick ->
                Text(
                    text = "Release pick: ${pick.symbol} • ${pick.companyName} • ${pick.risk.replaceFirstChar { it.uppercase() }} risk",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }

            content.weeklyChange?.let { WeeklyChangeCard(it) }

            RiskSelector(
                selections = content.dashboard.riskSelections,
                selectedRisk = content.selectedRisk,
                onRiskSelected = onRiskSelected
            )

            if (selectedSelection.status == SelectionStatus.PICKED && selectedSelection.pick != null) {
                PickDetails(
                    pick = selectedSelection.pick,
                    showFullSummary = showFullSummary,
                    onToggleSummary = { showFullSummary = !showFullSummary }
                )
            } else {
                NoPickDetails(selection = selectedSelection)
            }

            Text(
                text = "This is a model-driven research tool, not financial advice. Always verify the thesis independently before investing.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }

    if (showInfoDialog) {
        ModelInfoDialog(onDismiss = { showInfoDialog = false })
    }
}

@Composable
private fun HeaderRow(
    isRefreshing: Boolean,
    onRefresh: () -> Unit,
    onShowInfo: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Column {
            Text(
                text = "Weekly stock release",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = if (isRefreshing) "Refreshing..." else "Commercial release mode",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)
        ) {
            HeaderActionChip(
                label = "How it works",
                onClick = onShowInfo
            )
            HeaderActionChip(
                label = if (isRefreshing) "Refreshing" else "Refresh",
                onClick = onRefresh
            )
        }
    }
}

@Composable
private fun HeaderActionChip(
    label: String,
    onClick: () -> Unit
) {
    AssistChip(
        onClick = onClick,
        label = {
            Text(
                text = label,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        },
        colors = AssistChipDefaults.assistChipColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f),
            labelColor = MaterialTheme.colorScheme.onSurface
        ),
        border = AssistChipDefaults.assistChipBorder(
            enabled = true,
            borderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
        )
    )
}

@Composable
private fun StatusBanner(
    title: String,
    message: String,
    accent: Color
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = accent.copy(alpha = 0.12f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = accent
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
private fun WeeklyChangeCard(change: WeeklyChangeSummary) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "What changed this week",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = change.title,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = change.summary,
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "${change.previousWeekLabel} → ${change.currentWeekLabel}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                change.scoreDelta?.let {
                    DeltaChip(
                        label = "Score",
                        value = formatDelta(it),
                        accent = if (it >= 0) RiskLowColor else RiskHighColor
                    )
                }
                change.confidenceDelta?.let {
                    DeltaChip(
                        label = "Confidence",
                        value = formatDelta(it),
                        accent = if (it >= 0) RiskLowColor else RiskHighColor
                    )
                }
                if (change.currentRisk != null && change.previousRisk != null && change.currentRisk != change.previousRisk) {
                    DeltaChip(
                        label = "Risk",
                        value = "${change.previousRisk} -> ${change.currentRisk}",
                        accent = RiskMediumColor
                    )
                }
            }
        }
    }
}

@Composable
private fun DeltaChip(
    label: String,
    value: String,
    accent: Color
) {
    AssistChip(
        onClick = {},
        enabled = false,
        label = {
            Text("$label $value")
        },
        colors = AssistChipDefaults.assistChipColors(
            disabledContainerColor = accent.copy(alpha = 0.12f),
            disabledLabelColor = accent
        ),
        border = AssistChipDefaults.assistChipBorder(
            enabled = false,
            borderColor = accent.copy(alpha = 0.35f)
        )
    )
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun RiskSelector(
    selections: Map<String, Selection>,
    selectedRisk: String,
    onRiskSelected: (String) -> Unit
) {
    val orderedKeys = listOf("low", "medium", "high")
        .filter { selections.containsKey(it) }

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = "Risk profiles",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold
        )
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            orderedKeys.forEach { risk ->
                val selection = selections.getValue(risk)
                RiskChip(
                    risk = risk,
                    isSelected = selectedRisk == risk,
                    hasPick = selection.status == SelectionStatus.PICKED,
                    onClick = { onRiskSelected(risk) }
                )
            }
        }
    }
}

@Composable
private fun RiskChip(
    risk: String,
    isSelected: Boolean,
    hasPick: Boolean,
    onClick: () -> Unit
) {
    val baseColor = riskColor(risk)
    val targetBackground = when {
        isSelected -> baseColor.copy(alpha = 0.18f)
        !hasPick -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
        else -> Color.Transparent
    }
    val targetBorder = when {
        isSelected -> baseColor
        !hasPick -> MaterialTheme.colorScheme.outline.copy(alpha = 0.5f)
        else -> MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
    }
    val targetText = when {
        isSelected -> baseColor
        !hasPick -> MaterialTheme.colorScheme.onSurfaceVariant
        else -> MaterialTheme.colorScheme.onSurface
    }

    val background by animateColorAsState(targetValue = targetBackground, label = "riskBg")
    val border by animateColorAsState(targetValue = targetBorder, label = "riskBorder")
    val textColor by animateColorAsState(targetValue = targetText, label = "riskText")

    Text(
        text = buildString {
            append(risk.replaceFirstChar { it.uppercase() })
            if (!hasPick) append(" · No pick")
        },
        modifier = Modifier
            .border(1.dp, border, RoundedCornerShape(999.dp))
            .background(background, RoundedCornerShape(999.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        style = MaterialTheme.typography.bodyMedium,
        fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
        color = textColor
    )
}

@Composable
private fun PickDetails(
    pick: WeeklyPick,
    showFullSummary: Boolean,
    onToggleSummary: () -> Unit
) {
    val summaryReason = pick.reasons.firstOrNull { it.startsWith("News summary", ignoreCase = true) }
    val primaryReasons = pick.reasons.filterNot { it == summaryReason }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            StockLogoBadge(symbol = pick.symbol, risk = pick.risk)
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = pick.companyName,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = "${pick.risk.replaceFirstChar { it.uppercase() }} risk • ${pick.confidenceLabel.replaceFirstChar { it.uppercase() }} confidence",
                    style = MaterialTheme.typography.bodyMedium,
                    color = riskColor(pick.risk)
                )
            }
        }

        ScoreMeter(score = pick.modelScore)

        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MetricCard(
                label = "5D Momentum",
                value = percentText(pick.metrics.momentum5d)
            )
            MetricCard(
                label = "Volatility",
                value = percentText(pick.metrics.dailyVolatility)
            )
            MetricCard(
                label = "Confidence",
                value = pick.confidenceLabel.replaceFirstChar { it.uppercase() }
            )
            MetricCard(
                label = "Articles",
                value = pick.articleCount.toString()
            )
        }

        pick.thesisMonitor?.let { monitor ->
            ThesisMonitorSection(monitor)
        }

        ScoreBreakdownSection(pick.scoreBreakdown)

        if (pick.newsEvidence.isNotEmpty()) {
            NewsEvidenceSection(pick.newsEvidence)
        }

        HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)

        Text(
            text = "Why it qualified",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold
        )

        primaryReasons.forEach { reason ->
            Text(
                text = "• $reason",
                style = MaterialTheme.typography.bodyMedium
            )
        }

        summaryReason?.let { summary ->
            Text(
                text = "• $summary",
                style = MaterialTheme.typography.bodyMedium,
                maxLines = if (showFullSummary) Int.MAX_VALUE else 4,
                overflow = TextOverflow.Ellipsis
            )
            TextButton(
                onClick = onToggleSummary,
                modifier = Modifier.align(Alignment.Start)
            ) {
                Text(if (showFullSummary) "Show less" else "Show more")
            }
        }
    }
}

@Composable
private fun ThesisMonitorSection(monitor: ThesisMonitor) {
    val accent = thesisStatusColor(monitor.status)

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = accent.copy(alpha = 0.10f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Thesis monitor",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )
                DeltaChip(
                    label = "Status",
                    value = monitor.status.name.lowercase().replaceFirstChar { it.uppercase() },
                    accent = accent
                )
            }

            Text(
                text = monitor.headline,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = monitor.summary,
                style = MaterialTheme.typography.bodyMedium
            )

            if (monitor.alerts.isNotEmpty()) {
                monitor.alerts.forEach { alert ->
                    Text(
                        text = "• $alert",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                monitor.signals.forEach { signal ->
                    SignalPill(signal)
                }
            }
        }
    }
}

@Composable
private fun SignalPill(signal: ThesisSignal) {
    val accent = thesisSignalColor(signal.state)
    AssistChip(
        onClick = {},
        enabled = false,
        label = {
            Text(
                text = "${signal.label}: ${signal.value}",
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        },
        colors = AssistChipDefaults.assistChipColors(
            disabledContainerColor = accent.copy(alpha = 0.12f),
            disabledLabelColor = accent
        ),
        border = AssistChipDefaults.assistChipBorder(
            enabled = false,
            borderColor = accent.copy(alpha = 0.35f)
        )
    )
}

@Composable
private fun ScoreBreakdownSection(scoreBreakdown: ScoreBreakdown) {
    val contributions = listOf(
        "Momentum" to scoreBreakdown.momentum,
        "Volatility" to scoreBreakdown.volatilityPenalty,
        "News" to scoreBreakdown.newsAdjustment,
        "Alignment" to scoreBreakdown.signalAlignment,
    )
    val maxMagnitude = contributions.maxOfOrNull { abs(it.second) }?.coerceAtLeast(0.01) ?: 0.01

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = "Score breakdown",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            contributions.forEach { (label, value) ->
                ContributionRow(
                    label = label,
                    value = value,
                    maxMagnitude = maxMagnitude
                )
            }
            Text(
                text = "Short ${formatSigned(scoreBreakdown.shortMomentum)} • Medium ${formatSigned(scoreBreakdown.mediumMomentum)} • Trend ${formatSigned(scoreBreakdown.trendQuality)} • Volume ${formatSigned(scoreBreakdown.volumeConfirmation)}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun ContributionRow(
    label: String,
    value: Double,
    maxMagnitude: Double
) {
    val accent = if (value >= 0) RiskLowColor else RiskHighColor
    val normalized = (abs(value) / maxMagnitude).toFloat().coerceIn(0f, 1f)

    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = formatSigned(value),
                style = MaterialTheme.typography.bodyMedium,
                color = accent,
                fontWeight = FontWeight.SemiBold
            )
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .background(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(999.dp)
                )
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(normalized)
                    .height(8.dp)
                    .background(
                        color = accent,
                        shape = RoundedCornerShape(999.dp)
                    )
            )
        }
    }
}

@Composable
private fun NewsEvidenceSection(newsEvidence: List<NewsEvidence>) {
    val uriHandler = LocalUriHandler.current

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = "News evidence",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            newsEvidence.forEachIndexed { index, article ->
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = article.title,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium
                    )
                    formatEvidenceMetadata(article)?.let { metadata ->
                        Text(
                            text = metadata,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    if (!article.url.isNullOrBlank()) {
                        TextButton(
                            onClick = { uriHandler.openUri(article.url) },
                            modifier = Modifier.padding(start = 0.dp)
                        ) {
                            Text("Open source")
                        }
                    }
                }
                if (index != newsEvidence.lastIndex) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun NoPickDetails(selection: Selection) {
    val bestCandidate = selection.bestCandidate
    val scoreGap = bestCandidate
        ?.takeIf { it.modelScore < selection.thresholdScore }
        ?.let { selection.thresholdScore - it.modelScore }
    val confidenceGap = bestCandidate
        ?.takeIf { it.confidenceScore < selection.thresholdConfidence }
        ?.let { selection.thresholdConfidence - it.confidenceScore }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "No release pick for this profile",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = selection.statusReason,
                style = MaterialTheme.typography.bodyMedium
            )

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                DeltaChip(
                    label = "Required score",
                    value = formatSigned(selection.thresholdScore),
                    accent = RiskMediumColor
                )
                DeltaChip(
                    label = "Required conf.",
                    value = formatSigned(selection.thresholdConfidence),
                    accent = RiskMediumColor
                )
            }

            bestCandidate?.let { candidate ->
                HorizontalDivider()
                Text(
                    text = "Closest candidate",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = "${candidate.symbol} • ${candidate.companyName}",
                    style = MaterialTheme.typography.bodyLarge
                )
                Text(
                    text = "Score ${formatSigned(candidate.modelScore)} • ${candidate.confidenceLabel.replaceFirstChar { it.uppercase() }} confidence",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                scoreGap?.let {
                    Text(
                        text = "The candidate missed the score bar by ${formatSigned(it)}.",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                confidenceGap?.let {
                    Text(
                        text = "Confidence came in ${formatSigned(it)} below the minimum threshold.",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
    }
}

@Composable
private fun MetricCard(
    label: String,
    value: String
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun ModelInfoDialog(onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = "How the release works",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("The app publishes one release-quality pick per market week only when a candidate clears both score and confidence thresholds.")
                Text("The score is additive: momentum helps, volatility subtracts, and recent news sentiment nudges the result up or down.")
                Text("The release screen compares this week with last week, shows the score contributions, and surfaces the headlines that informed the signal.")
                Text("If a risk profile does not clear the minimum bar, the app shows no pick instead of forcing a weak idea into the UI.")
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        }
    )
}

@Composable
private fun StockLogoBadge(
    symbol: String,
    risk: String
) {
    val baseColor = riskColor(risk)

    Box(
        modifier = Modifier
            .size(56.dp)
            .background(
                color = baseColor.copy(alpha = 0.12f),
                shape = CircleShape
            ),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .size(48.dp)
                .background(baseColor, CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = symbol.uppercase(Locale.getDefault()),
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
    }
}

@Composable
private fun ScoreMeter(score: Double) {
    val clamped = score.coerceIn(-0.4, 0.4)
    val normalized = ((clamped + 0.4) / 0.8).toFloat()

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            text = "Model score ${formatSigned(score)}",
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold
        )
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(12.dp)
                .background(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(999.dp)
                )
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(normalized)
                    .height(12.dp)
                    .background(
                        brush = Brush.horizontalGradient(
                            listOf(RiskHighColor, RiskMediumColor, RiskLowColor)
                        ),
                        shape = RoundedCornerShape(999.dp)
                    )
            )
        }
        Text(
            text = "Momentum, volatility, and news are blended into one release score.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private fun riskColor(risk: String): Color {
    return when (risk.lowercase()) {
        "low" -> RiskLowColor
        "medium" -> RiskMediumColor
        "high" -> RiskHighColor
        else -> Color.Gray
    }
}

private fun thesisStatusColor(status: ThesisMonitorStatus): Color {
    return when (status) {
        ThesisMonitorStatus.HEALTHY -> RiskLowColor
        ThesisMonitorStatus.WATCH -> RiskMediumColor
        ThesisMonitorStatus.RISK -> RiskHighColor
    }
}

private fun thesisSignalColor(state: ThesisSignalState): Color {
    return when (state) {
        ThesisSignalState.POSITIVE -> RiskLowColor
        ThesisSignalState.WATCH -> RiskMediumColor
        ThesisSignalState.RISK -> RiskHighColor
    }
}

private fun percentText(value: Double): String = String.format(Locale.US, "%.1f%%", value * 100)

private fun formatSigned(value: Double): String = String.format(Locale.US, "%+.3f", value)

private fun formatDelta(value: Double): String = String.format(Locale.US, "%+.2f", value)

private fun formatDate(date: LocalDate): String {
    val formatter = DateTimeFormatter.ofPattern("MMM d, yyyy", Locale.getDefault())
    return date.format(formatter)
}

private fun formatInstant(instant: Instant): String {
    val formatter = DateTimeFormatter.ofPattern("MMM d, yyyy HH:mm", Locale.getDefault())
    return instant.atZone(ZoneId.systemDefault()).format(formatter)
}

private fun formatEvidenceMetadata(article: NewsEvidence): String? {
    val parts = buildList {
        article.provider?.let { add(it) }
        article.publishedAt?.let { add(formatInstant(it)) }
        article.relevanceScore?.let { add("relevance ${String.format(Locale.US, "%.2f", it)}") }
        article.sentiment?.let { add("sentiment ${formatDelta(it)}") }
    }
    return parts.takeIf { it.isNotEmpty() }?.joinToString(" • ")
}
