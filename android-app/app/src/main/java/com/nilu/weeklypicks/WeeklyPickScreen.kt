@file:OptIn(ExperimentalLayoutApi::class)

package com.nilu.weeklypicks

import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.nilu.weeklypicks.ui.theme.BullGreen
import com.nilu.weeklypicks.ui.theme.Copper
import com.nilu.weeklypicks.ui.theme.CopperBright
import com.nilu.weeklypicks.ui.theme.Ember
import com.nilu.weeklypicks.ui.theme.Graphite700
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
        shape = RoundedCornerShape(34.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.28f),
                    shape = RoundedCornerShape(34.dp)
                )
                .background(
                    Brush.verticalGradient(
                        listOf(
                            MaterialTheme.colorScheme.surface,
                            MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.92f),
                            MaterialTheme.colorScheme.background.copy(alpha = 0.96f)
                        )
                    )
                )
                .padding(22.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(0.34f)
                    .height(6.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(
                        Brush.horizontalGradient(
                            listOf(Copper, CopperBright, BullGreen)
                        )
                    )
            )

            HeaderRow()

            Text(
                text = content.dashboard.marketContext.weekLabel,
                style = MaterialTheme.typography.displayMedium,
                fontWeight = FontWeight.Bold
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

            content.thesisMonitorMessage?.let { thesisMessage ->
                StatusBanner(
                    title = "Live thesis monitor",
                    message = thesisMessage,
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
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            HyraxWordmark()
            Text(
                text = "Weekly conviction",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
        }
        HyraxBrandMark(size = 64.dp)
    }
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
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.62f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .border(
                    width = 1.dp,
                    color = accent.copy(alpha = 0.35f),
                    shape = RoundedCornerShape(24.dp)
                )
                .padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.Top
        ) {
            Box(
                modifier = Modifier
                    .width(10.dp)
                    .height(52.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(
                        Brush.verticalGradient(
                            listOf(accent, accent.copy(alpha = 0.35f))
                        )
                    )
            )

            Column(
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = title.uppercase(Locale.US),
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = accent
                )
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}

@Composable
private fun MetaChip(
    text: String,
    accent: Color = MaterialTheme.colorScheme.onSurfaceVariant
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(accent.copy(alpha = 0.12f))
            .border(
                width = 1.dp,
                color = accent.copy(alpha = 0.28f),
                shape = RoundedCornerShape(999.dp)
            )
            .padding(horizontal = 10.dp, vertical = 7.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelMedium,
            color = accent
        )
    }
}

@Composable
private fun SignalTag(
    text: String,
    accent: Color
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(accent.copy(alpha = 0.12f))
            .border(
                width = 1.dp,
                color = accent.copy(alpha = 0.25f),
                shape = RoundedCornerShape(999.dp)
            )
            .padding(horizontal = 10.dp, vertical = 7.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelMedium,
            color = accent,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
private fun HeroSurface(
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.68f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.22f),
                    shape = RoundedCornerShape(30.dp)
                )
                .background(
                    Brush.verticalGradient(
                        listOf(
                            MaterialTheme.colorScheme.surface.copy(alpha = 0.98f),
                            MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.76f)
                        )
                    )
                )
                .padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
            content = content
        )
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
        HeroSurface {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                StockLogoBadge(symbol = pick.symbol, risk = pick.risk)
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = pick.companyName,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold
                    )
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        SignalTag(
                            text = "${pick.risk.replaceFirstChar { it.uppercase() }} risk",
                            accent = riskColor(pick.risk)
                        )
                        SignalTag(
                            text = "${pick.confidenceLabel.replaceFirstChar { it.uppercase() }} confidence",
                            accent = if (pick.confidenceScore >= 0.7) BullGreen else CopperBright
                        )
                    }
                }
            }

            ScoreMeter(score = pick.modelScore)

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
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
    val transition = rememberInfiniteTransition(label = "thesisPulse")
    val pulseAlpha by transition.animateFloat(
        initialValue = if (monitor.status == ThesisMonitorStatus.HEALTHY) 0.16f else 0.24f,
        targetValue = if (monitor.status == ThesisMonitorStatus.HEALTHY) 0.24f else 0.42f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1600, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "thesisPulseAlpha"
    )

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = accent.copy(alpha = 0.08f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .border(
                    width = 1.dp,
                    color = accent.copy(alpha = pulseAlpha),
                    shape = RoundedCornerShape(24.dp)
                )
                .padding(14.dp),
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
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(24.dp)
                )
                .padding(14.dp),
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
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(24.dp)
                )
                .padding(14.dp),
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
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(24.dp)
                )
                .padding(14.dp),
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
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.62f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(24.dp)
                )
                .padding(16.dp),
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
        modifier = Modifier.border(
            width = 1.dp,
            color = MaterialTheme.colorScheme.outline.copy(alpha = 0.16f),
            shape = RoundedCornerShape(22.dp)
        ),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.78f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = label.uppercase(Locale.US),
                style = MaterialTheme.typography.labelMedium,
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
                text = "How Hyrax Alpha works",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Hyrax Alpha only publishes a release when a candidate clears both score and confidence thresholds.")
                Text("The score blends technical strength, company news, world-news overlays, and risk penalties into one conviction signal.")
                Text("The dashboard is intentionally compact at the top and deeper analysis sits behind expandable panels.")
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
    val transition = rememberInfiniteTransition(label = "logoPulse")
    val haloScale by transition.animateFloat(
        initialValue = 0.92f,
        targetValue = 1.08f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 2200),
            repeatMode = RepeatMode.Reverse
        ),
        label = "logoHalo"
    )

    Box(
        modifier = Modifier
            .size(66.dp),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .size(66.dp * haloScale)
                .graphicsLayer(alpha = 0.18f)
                .background(
                    brush = Brush.radialGradient(
                        listOf(baseColor.copy(alpha = 0.95f), Color.Transparent)
                    ),
                    shape = CircleShape
                )
        )
        Box(
            modifier = Modifier
                .size(60.dp)
                .background(
                    brush = Brush.linearGradient(
                        listOf(baseColor, CopperBright)
                    ),
                    shape = CircleShape
                )
                .border(
                    width = 2.dp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f),
                    shape = CircleShape
                ),
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
    val target = ((score.coerceIn(-0.4, 0.4) + 0.4) / 0.8).toFloat()
    val normalized by animateFloatAsState(
        targetValue = target,
        animationSpec = tween(durationMillis = 900),
        label = "scoreMeter"
    )
    val transition = rememberInfiniteTransition(label = "scoreSheen")
    val sheenOffset by transition.animateFloat(
        initialValue = -0.35f,
        targetValue = 1.35f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 2600, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "scoreSheenOffset"
    )

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            text = "Conviction score",
            style = MaterialTheme.typography.labelLarge,
            color = CopperBright
        )
        Text(
            text = formatSigned(score),
            style = MaterialTheme.typography.displayLarge,
            fontWeight = FontWeight.Bold
        )
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(14.dp)
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
                            listOf(Ember, CopperBright, BullGreen)
                        ),
                        shape = RoundedCornerShape(999.dp)
                    )
            )
            Box(
                modifier = Modifier
                    .fillMaxWidth(0.22f)
                    .height(14.dp)
                    .graphicsLayer(translationX = 220f * sheenOffset, alpha = 0.34f)
                    .background(
                        Brush.horizontalGradient(
                            listOf(Color.Transparent, Color.White.copy(alpha = 0.65f), Color.Transparent)
                        ),
                        RoundedCornerShape(999.dp)
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
            style = MaterialTheme.typography.labelLarge,
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
