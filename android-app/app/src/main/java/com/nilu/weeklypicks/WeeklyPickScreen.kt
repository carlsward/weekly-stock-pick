@file:OptIn(ExperimentalLayoutApi::class)

package com.nilu.weeklypicks

import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
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
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.nilu.weeklypicks.ui.theme.BullGreen
import com.nilu.weeklypicks.ui.theme.CopperBright
import com.nilu.weeklypicks.ui.theme.HyraxNavy
import com.nilu.weeklypicks.ui.theme.HyraxSky
import com.nilu.weeklypicks.ui.theme.RiskHighColor
import com.nilu.weeklypicks.ui.theme.RiskLowColor
import com.nilu.weeklypicks.ui.theme.RiskMediumColor
import java.time.Instant
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
    var showResearchDetails by rememberSaveable(detailStateKey) { mutableStateOf(false) }
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
            SpotlightHeader(
                weekLabel = content.dashboard.marketContext.weekLabel
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
                    expanded = showResearchDetails,
                    onToggle = { showResearchDetails = !showResearchDetails }
                )
            } else {
                NoPickDetails(
                    selection = displaySelection,
                    weekLabel = content.dashboard.marketContext.weekLabel,
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
private fun SpotlightHeader(
    weekLabel: String
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        BrandPlaque()
        Column(
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            HyraxWordmark()
            Text(
                text = weekLabel,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun BrandPlaque() {
    Card(
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = HyraxNavy),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .border(
                    width = 1.dp,
                    color = CopperBright.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(22.dp)
                )
                .padding(6.dp)
        ) {
            Image(
                painter = painterResource(id = R.drawable.hyrax_alpha_logo),
                contentDescription = "Hyrax Alpha logo",
                modifier = Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(18.dp)),
                contentScale = ContentScale.Crop
            )
        }
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
private fun HeroSurface(
    accent: Color,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.94f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .border(
                    width = 1.dp,
                    color = accent.copy(alpha = 0.22f),
                    shape = RoundedCornerShape(30.dp)
                )
                .background(
                    Brush.verticalGradient(
                        listOf(
                            MaterialTheme.colorScheme.surface.copy(alpha = 0.99f),
                            MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.78f),
                            MaterialTheme.colorScheme.surface.copy(alpha = 0.95f)
                        )
                    )
                )
        ) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .size(220.dp)
                    .graphicsLayer(alpha = 0.92f)
                    .background(
                        Brush.radialGradient(
                            listOf(accent.copy(alpha = 0.22f), Color.Transparent)
                        ),
                        shape = CircleShape
                    )
            )
            Box(
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .size(180.dp)
                    .graphicsLayer(alpha = 0.85f)
                    .background(
                        Brush.radialGradient(
                            listOf(HyraxSky.copy(alpha = 0.16f), Color.Transparent)
                        ),
                        shape = CircleShape
                    )
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                content = content
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

            if (expanded) {
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
    expanded: Boolean,
    onToggle: () -> Unit
) {
    val summaryReason = pick.reasons.firstOrNull { it.startsWith("News summary", ignoreCase = true) }
    val primaryReasons = pick.reasons.filterNot { it == summaryReason }
    val accent = convictionAccent(pick.modelScore)

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        HeroSurface(accent = accent) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = pick.symbol,
                        style = MaterialTheme.typography.displayLarge,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = pick.companyName,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                Text(
                    text = "Model score",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = formatSigned(pick.modelScore),
                    style = MaterialTheme.typography.displayMedium,
                    fontWeight = FontWeight.Bold,
                    color = HyraxNavy
                )
                Text(
                    text = "Confidence ${(pick.confidenceScore * 100).toInt()}% • ${
                        pick.confidenceLabel.replaceFirstChar { it.uppercase() }
                    }",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    TextButton(onClick = onToggle) {
                        Text(if (expanded) "Hide research" else "Open research")
                    }
                }
            }
            if (expanded) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.14f))

                pick.thesisMonitor?.let { monitor ->
                    ResearchBlock(title = "Thesis health") {
                        val accentColor = thesisStatusColor(monitor.status)
                        DeltaChip(
                            label = "Health",
                            value = thesisStatusLabel(monitor.status),
                            accent = accentColor
                        )
                        Text(
                            text = monitor.headline,
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = monitor.summary,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        if (monitor.alerts.isNotEmpty()) {
                            monitor.alerts.forEach { alert ->
                                Text(
                                    text = "• $alert",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }
                }
                ResearchBlock(title = "Score breakdown") {
                    val contributions = listOf(
                        "Momentum" to pick.scoreBreakdown.momentum,
                        "Volatility" to pick.scoreBreakdown.volatilityPenalty,
                        "News" to pick.scoreBreakdown.newsAdjustment,
                        "Alignment" to pick.scoreBreakdown.signalAlignment
                    )
                    val maxMagnitude = contributions.maxOfOrNull { abs(it.second) }?.coerceAtLeast(0.01) ?: 0.01
                    Text(
                        text = scoreBreakdownSummary(contributions),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    contributions.forEach { (label, value) ->
                        ContributionRow(
                            label = label,
                            value = value,
                            maxMagnitude = maxMagnitude
                        )
                    }
                }

                ResearchBlock(title = "Why it qualified") {
                    primaryReasons.forEach { reason ->
                        Text(
                            text = "• $reason",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                    summaryReason?.let {
                        Text(
                            text = it.removePrefix("News summary: ").removePrefix("News summary"),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                if (pick.newsEvidence.isNotEmpty()) {
                    ResearchBlock(title = "News evidence") {
                        Text(
                            text = "${pick.newsEvidence.size} recent headline${if (pick.newsEvidence.size == 1) "" else "s"} contributed to the signal.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        pick.newsEvidence.take(4).forEachIndexed { index, article ->
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
                            }
                            if (index != pick.newsEvidence.take(4).lastIndex) {
                                HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ResearchBlock(
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold
        )
        content()
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
private fun NoPickDetails(
    selection: Selection,
    weekLabel: String,
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

    HeroSurface(accent = RiskMediumColor) {
        Text(
            text = "No pick",
            style = MaterialTheme.typography.displayMedium,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = weekLabel,
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = selection.statusReason,
            style = MaterialTheme.typography.bodyLarge
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End
        ) {
            TextButton(onClick = onToggle) {
                Text(if (expanded) "Hide details" else "Why no pick?")
            }
        }

        if (expanded) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.14f))
            ResearchBlock(title = "Release bar") {
                Text(
                    text = "Required score ${formatSigned(selection.thresholdScore)}",
                    style = MaterialTheme.typography.bodyMedium
                )
                Text(
                    text = "Required confidence ${(selection.thresholdConfidence * 100).toInt()}%",
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            bestCandidate?.let { candidate ->
                ResearchBlock(title = "Closest candidate") {
                    Text(
                        text = "${candidate.symbol} • ${candidate.companyName}",
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Score ${formatSigned(candidate.modelScore)} • ${candidate.confidenceLabel.replaceFirstChar { it.uppercase() }} confidence",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    scoreGap?.let {
                        Text(
                            text = "Missed the score bar by ${formatSigned(it)}.",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                    confidenceGap?.let {
                        Text(
                            text = "Confidence was ${formatSigned(it)} below the minimum.",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
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
                Text("Hyrax Alpha is built around one premise: publish one stock for the coming week only when the edge looks real.")
                Text("The model blends technical strength, company news, world-news overlays, and risk penalties into one conviction signal.")
                Text("If nothing clears the bar, the app stays disciplined and returns no pick instead of forcing a weak idea.")
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
private fun SectionHeader(
    title: String,
    expanded: Boolean,
    onToggle: () -> Unit,
    collapsedLabel: String = "Open"
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

private fun convictionAccent(score: Double): Color {
    return when {
        score >= 0.20 -> BullGreen
        score >= 0.10 -> CopperBright
        else -> RiskMediumColor
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

private fun formatSigned(value: Double): String = String.format(Locale.US, "%+.3f", value)

private fun formatDelta(value: Double): String = String.format(Locale.US, "%+.2f", value)

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
