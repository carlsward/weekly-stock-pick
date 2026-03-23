@file:OptIn(ExperimentalLayoutApi::class)

package com.nilu.weeklypicks

import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.background
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
    content: DashboardContent
) {
    val displaySelection = content.dashboard.overallSelection
    val detailStateKey = listOf(
        displaySelection.pick?.symbol ?: displaySelection.status.name,
        content.dashboard.marketContext.weekLabel
    ).joinToString(":")
    var showChangeDetails by rememberSaveable(detailStateKey) { mutableStateOf(false) }
    var showThesisMonitor by rememberSaveable(detailStateKey) { mutableStateOf(false) }
    var showScoreBreakdown by rememberSaveable(detailStateKey) { mutableStateOf(false) }
    var showNewsEvidence by rememberSaveable(detailStateKey) { mutableStateOf(false) }
    var showQualificationDetails by rememberSaveable(detailStateKey) { mutableStateOf(false) }
    var showNoPickDetails by rememberSaveable(detailStateKey) { mutableStateOf(false) }

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
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            HeaderRow()

            Text(
                text = content.dashboard.marketContext.weekLabel,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = buildString {
                    append("Updated ${formatInstant(content.dashboard.generatedAt)}")
                    append(" • Data ${formatDate(content.dashboard.dataAsOf)}")
                    if (content.source == DataSource.CACHE) {
                        append(" • Cached")
                    }
                },
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

            if (displaySelection.status == SelectionStatus.PICKED && displaySelection.pick != null) {
                PickDetails(
                    pick = displaySelection.pick,
                    showThesisMonitor = showThesisMonitor,
                    onToggleThesisMonitor = { showThesisMonitor = !showThesisMonitor },
                    showScoreBreakdown = showScoreBreakdown,
                    onToggleScoreBreakdown = { showScoreBreakdown = !showScoreBreakdown },
                    showNewsEvidence = showNewsEvidence,
                    onToggleNewsEvidence = { showNewsEvidence = !showNewsEvidence },
                    showQualificationDetails = showQualificationDetails,
                    onToggleQualificationDetails = { showQualificationDetails = !showQualificationDetails }
                )
            } else {
                NoPickDetails(
                    selection = displaySelection,
                    expanded = showNoPickDetails,
                    onToggle = { showNoPickDetails = !showNoPickDetails }
                )
            }

            content.weeklyChange?.let { change ->
                WeeklyChangeCard(
                    change = change,
                    expanded = showChangeDetails,
                    onToggle = { showChangeDetails = !showChangeDetails }
                )
            }
        }
    }
}

@Composable
private fun HeaderRow() {
    Text(
        text = "Weekly stock release",
        style = MaterialTheme.typography.headlineSmall,
        fontWeight = FontWeight.Bold
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
private fun WeeklyChangeCard(
    change: WeeklyChangeSummary,
    expanded: Boolean,
    onToggle: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SectionHeader(
                title = "This week",
                expanded = expanded,
                onToggle = onToggle
            )
            Text(
                text = change.title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = "${change.previousWeekLabel} -> ${change.currentWeekLabel}",
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

            if (expanded) {
                Text(
                    text = change.summary,
                    style = MaterialTheme.typography.bodyMedium
                )
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
private fun PickDetails(
    pick: WeeklyPick,
    showThesisMonitor: Boolean,
    onToggleThesisMonitor: () -> Unit,
    showScoreBreakdown: Boolean,
    onToggleScoreBreakdown: () -> Unit,
    showNewsEvidence: Boolean,
    onToggleNewsEvidence: () -> Unit,
    showQualificationDetails: Boolean,
    onToggleQualificationDetails: () -> Unit
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
            ThesisMonitorSection(
                monitor = monitor,
                expanded = showThesisMonitor,
                onToggle = onToggleThesisMonitor
            )
        }

        ScoreBreakdownSection(
            scoreBreakdown = pick.scoreBreakdown,
            expanded = showScoreBreakdown,
            onToggle = onToggleScoreBreakdown
        )

        if (pick.newsEvidence.isNotEmpty()) {
            NewsEvidenceSection(
                newsEvidence = pick.newsEvidence,
                expanded = showNewsEvidence,
                onToggle = onToggleNewsEvidence
            )
        }

        QualificationSection(
            primaryReasons = primaryReasons,
            summaryReason = summaryReason,
            expanded = showQualificationDetails,
            onToggle = onToggleQualificationDetails
        )
    }
}

@Composable
private fun ThesisMonitorSection(
    monitor: ThesisMonitor,
    expanded: Boolean,
    onToggle: () -> Unit
) {
    val accent = thesisStatusColor(monitor.status)

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
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
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.Top
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = "Thesis health",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = monitor.headline,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                        maxLines = if (expanded) Int.MAX_VALUE else 2,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                DeltaChip(
                    label = "Health",
                    value = thesisStatusLabel(monitor.status),
                    accent = accent
                )
            }

            Text(
                text = monitor.summary,
                style = MaterialTheme.typography.bodySmall,
                maxLines = if (expanded) Int.MAX_VALUE else 2,
                overflow = TextOverflow.Ellipsis
            )

            if (expanded && monitor.alerts.isNotEmpty()) {
                monitor.alerts.forEach { alert ->
                    Text(
                        text = "• $alert",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }

            if (expanded) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    monitor.signals.forEach { signal ->
                        SignalPill(signal)
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                TextButton(onClick = onToggle) {
                    Text(if (expanded) "Hide details" else "Open details")
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
private fun ScoreBreakdownSection(
    scoreBreakdown: ScoreBreakdown,
    expanded: Boolean,
    onToggle: () -> Unit
) {
    val contributions = listOf(
        "Momentum" to scoreBreakdown.momentum,
        "Volatility" to scoreBreakdown.volatilityPenalty,
        "News" to scoreBreakdown.newsAdjustment,
        "Alignment" to scoreBreakdown.signalAlignment,
    )
    val maxMagnitude = contributions.maxOfOrNull { abs(it.second) }?.coerceAtLeast(0.01) ?: 0.01

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            SectionHeader(
                title = "Score breakdown",
                expanded = expanded,
                onToggle = onToggle
            )
            Text(
                text = scoreBreakdownSummary(contributions),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (expanded) {
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
private fun NewsEvidenceSection(
    newsEvidence: List<NewsEvidence>,
    expanded: Boolean,
    onToggle: () -> Unit
) {
    val uriHandler = LocalUriHandler.current

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            SectionHeader(
                title = "News evidence",
                expanded = expanded,
                onToggle = onToggle
            )
            Text(
                text = "${newsEvidence.size} recent headline${if (newsEvidence.size == 1) "" else "s"} contributed to the signal.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (expanded) {
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
            } else {
                newsEvidence.firstOrNull()?.let { article ->
                    Text(
                        text = article.title,
                        style = MaterialTheme.typography.bodyMedium,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                    formatEvidenceMetadata(article)?.let { metadata ->
                        Text(
                            text = metadata,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun QualificationSection(
    primaryReasons: List<String>,
    summaryReason: String?,
    expanded: Boolean,
    onToggle: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            SectionHeader(
                title = "Why it qualified",
                expanded = expanded,
                onToggle = onToggle
            )

            val previewReasons = if (expanded) primaryReasons else primaryReasons.take(2)
            previewReasons.forEach { reason ->
                Text(
                    text = "• $reason",
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = if (expanded) Int.MAX_VALUE else 3,
                    overflow = TextOverflow.Ellipsis
                )
            }

            if (!expanded && primaryReasons.size > previewReasons.size) {
                Text(
                    text = "${primaryReasons.size - previewReasons.size} more notes hidden",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (expanded && summaryReason != null) {
                HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                Text(
                    text = summaryReason.removePrefix("News summary: ").removePrefix("News summary"),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun NoPickDetails(
    selection: Selection,
    expanded: Boolean,
    onToggle: () -> Unit
) {
    val bestCandidate = selection.bestCandidate
    val scoreGap = bestCandidate
        ?.takeIf { it.modelScore < selection.thresholdScore }
        ?.let { selection.thresholdScore - it.modelScore }
    val confidenceGap = bestCandidate
        ?.takeIf { it.confidenceScore < selection.thresholdConfidence }
        ?.let { selection.thresholdConfidence - it.confidenceScore }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SectionHeader(
                title = "No release pick for this profile",
                expanded = expanded,
                onToggle = onToggle,
                collapsedLabel = "Why"
            )
            Text(
                text = selection.statusReason,
                style = MaterialTheme.typography.bodyMedium
            )

            if (expanded) {
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
fun ModelInfoDialog(onDismiss: () -> Unit) {
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
                Text("The homepage keeps the pick compact and pushes the supporting analysis into expandable sections.")
                Text("This is a model-driven research tool, not financial advice. Verify the thesis independently before investing.")
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
                text = symbol.uppercase(Locale.US),
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
    }
}

@Composable
private fun ScoreMeter(score: Double) {
    val normalized = ((score.coerceIn(-0.4, 0.4) + 0.4) / 0.8).toFloat()

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
    }
}

@Composable
private fun SectionHeader(
    title: String,
    expanded: Boolean,
    onToggle: () -> Unit,
    collapsedLabel: String = "Details"
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold
        )
        TextButton(onClick = onToggle) {
            Text(if (expanded) "Hide" else collapsedLabel)
        }
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

private fun thesisStatusLabel(status: ThesisMonitorStatus): String {
    return when (status) {
        ThesisMonitorStatus.HEALTHY -> "Stable"
        ThesisMonitorStatus.WATCH -> "Watch"
        ThesisMonitorStatus.RISK -> "Review"
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
    val formatter = DateTimeFormatter.ofPattern("MMM d, yyyy", Locale.US)
    return date.format(formatter)
}

private fun formatInstant(instant: Instant): String {
    val formatter = DateTimeFormatter.ofPattern("MMM d, yyyy HH:mm", Locale.US)
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

private fun scoreBreakdownSummary(contributions: List<Pair<String, Double>>): String {
    val positive = contributions.maxByOrNull { it.second }?.takeIf { it.second > 0.01 }
    val negative = contributions.minByOrNull { it.second }?.takeIf { it.second < -0.01 }

    return when {
        positive != null && negative != null ->
            "${positive.first} is doing most of the lifting while ${negative.first.lowercase(Locale.US)} is the main drag."

        positive != null ->
            "${positive.first} is doing most of the lifting in the model."

        negative != null ->
            "${negative.first} is the main drag on the score."

        else ->
            "The score is broadly balanced across momentum, volatility, news, and alignment."
    }
}
