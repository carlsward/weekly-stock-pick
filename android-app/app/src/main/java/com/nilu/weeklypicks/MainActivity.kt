package com.nilu.weeklypicks

import android.Manifest
import android.os.Bundle
import android.os.Build
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.core.content.ContextCompat
import android.content.pm.PackageManager
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nilu.weeklypicks.ui.theme.NiLUWeeklyPicksTheme
import com.nilu.weeklypicks.ui.theme.RiskHighColor
import com.nilu.weeklypicks.ui.theme.RiskLowColor
import com.nilu.weeklypicks.ui.theme.RiskMediumColor
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestNotificationPermissionIfNeeded()
        setContent {
            NiLUWeeklyPicksTheme {
                val vm: MainViewModel = viewModel(
                    factory = MainViewModel.factory(applicationContext)
                )
                var showSplash by remember { mutableStateOf(true) }
                var showInfoDialog by remember { mutableStateOf(false) }

                if (showSplash) {
                    SplashScreen(onFinished = { showSplash = false })
                } else {
                    Surface(color = MaterialTheme.colorScheme.background) {
                        when (val state = vm.uiState) {
                            UiState.Loading -> LoadingView()
                            is UiState.Error -> ErrorView(
                                message = state.message,
                                onRetry = vm::retry
                            )

                            is UiState.Content -> {
                                PullToRefreshBox(
                                    isRefreshing = state.content.isRefreshing,
                                    onRefresh = vm::refresh,
                                    modifier = Modifier.fillMaxSize()
                                ) {
                                    LazyColumn(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .statusBarsPadding()
                                            .navigationBarsPadding()
                                            .padding(horizontal = 16.dp),
                                        verticalArrangement = Arrangement.spacedBy(16.dp),
                                        contentPadding = PaddingValues(vertical = 16.dp)
                                    ) {
                                        item {
                                            WeeklyPickScreen(
                                                content = state.content
                                            )
                                        }
                                        item {
                                            MonthlyPickSection(
                                                feed = state.content.monthlyPick,
                                                message = state.content.monthlyPickMessage
                                            )
                                        }
                                        item {
                                            TrackRecordSection(
                                                trackRecord = state.content.trackRecord,
                                                message = state.content.trackRecordMessage
                                            )
                                        }
                                        item {
                                            HistorySection(
                                                entries = state.content.historyEntries,
                                                message = state.content.historyMessage
                                            )
                                        }
                                        item {
                                            FooterInfoButton(
                                                onClick = { showInfoDialog = true }
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (showInfoDialog) {
                    ModelInfoDialog(
                        onDismiss = { showInfoDialog = false }
                    )
                }
            }
        }
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return
        }

        val granted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.POST_NOTIFICATIONS
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }
}

@Composable
fun LoadingView() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            CircularProgressIndicator()
            Text(
                text = "Loading weekly picks...",
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}

@Composable
fun ErrorView(
    message: String,
    onRetry: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = message,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.error
            )
            Button(onClick = onRetry) {
                Text("Retry")
            }
        }
    }
}

@Composable
fun FooterInfoButton(
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center
    ) {
        TextButton(onClick = onClick) {
            Text("How it works")
        }
    }
}

@Composable
fun HistorySection(
    entries: List<HistoryEntry>,
    message: String?
) {
    if (entries.isEmpty() && message == null) {
        return
    }

    val recentEntries = entries
        .sortedBy { it.weekStart }
        .takeLast(6)
        .reversed()
    val preview = recentEntries.firstOrNull()
    var expanded by rememberSaveable { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
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
                    text = "Recent releases",
                    style = MaterialTheme.typography.titleMedium
                )
                TextButton(onClick = { expanded = !expanded }) {
                    Text(if (expanded) "Hide" else "Open")
                }
            }

            preview?.let { latest ->
                val previewText = if (latest.status == SelectionStatus.PICKED) {
                    "Latest: ${latest.symbol} for ${latest.weekLabel}"
                } else {
                    "Latest week closed without a release pick."
                }
                Text(
                    text = previewText,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (expanded) {
                message?.let {
                    Text(
                        text = it,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                recentEntries.forEachIndexed { index, entry ->
                    HistoryRow(entry)
                    if (index != recentEntries.lastIndex) {
                        HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                    }
                }
            }
        }
    }
}

@Composable
fun TrackRecordSection(
    trackRecord: TrackRecordFeed?,
    message: String?
) {
    if (trackRecord == null && message == null) {
        return
    }

    var expanded by rememberSaveable { mutableStateOf(false) }
    val recentClosedEntries = trackRecord?.entries
        ?.filter { it.status == SelectionStatus.PICKED && it.realized5dReturn != null }
        ?.take(6)
        .orEmpty()

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
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
                    text = "Performance",
                    style = MaterialTheme.typography.titleMedium
                )
                if (trackRecord != null) {
                    TextButton(onClick = { expanded = !expanded }) {
                        Text(if (expanded) "Hide" else "Open")
                    }
                }
            }

            message?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            val summary = trackRecord?.summary
            if (summary != null) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    PerformanceMetric(
                        label = "Closed",
                        value = summary.closedPicks.toString(),
                        modifier = Modifier.weight(1f)
                    )
                    PerformanceMetric(
                        label = "Win rate",
                        value = percentLabel(summary.winRate),
                        modifier = Modifier.weight(1f)
                    )
                    PerformanceMetric(
                        label = "Beat SPY",
                        value = percentLabel(summary.beatSpyRate),
                        modifier = Modifier.weight(1f)
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    PerformanceMetric(
                        label = "Avg 5D",
                        value = signedPercentLabel(summary.average5dReturn),
                        modifier = Modifier.weight(1f)
                    )
                    PerformanceMetric(
                        label = "Avg excess",
                        value = signedPercentLabel(summary.average5dExcessReturn),
                        modifier = Modifier.weight(1f)
                    )
                    PerformanceMetric(
                        label = "Compounded",
                        value = signedPercentLabel(summary.compounded5dReturn),
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            if (expanded && recentClosedEntries.isNotEmpty()) {
                HorizontalDivider()
                recentClosedEntries.forEachIndexed { index, entry ->
                    Column(
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        Text(
                            text = "${entry.weekLabel} • ${entry.symbol ?: "No pick"}",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = "Return ${signedPercentLabel(entry.realized5dReturn)} • Excess ${signedPercentLabel(entry.realized5dExcessReturn)}",
                            style = MaterialTheme.typography.bodySmall,
                            color = when (entry.outcome) {
                                "win" -> RiskLowColor
                                "loss" -> RiskHighColor
                                else -> MaterialTheme.colorScheme.onSurfaceVariant
                            }
                        )
                    }
                    if (index != recentClosedEntries.lastIndex) {
                        HorizontalDivider()
                    }
                }
            }
        }
    }
}

@Composable
fun MonthlyPickSection(
    feed: MonthlyPickFeed?,
    message: String?
) {
    if (feed == null && message == null) {
        return
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = "Monthly conviction",
                style = MaterialTheme.typography.titleMedium
            )

            message?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (feed != null) {
                Text(
                    text = feed.periodContext.monthLabel,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                val selection = feed.selection
                if (selection.status == SelectionStatus.PICKED && selection.pick != null) {
                    val pick = selection.pick
                    Text(
                        text = "${pick.symbol} - ${pick.companyName}",
                        style = MaterialTheme.typography.titleLarge
                    )
                    Text(
                        text = "${pick.sector.replaceFirstChar { it.uppercase() }} • ${pick.risk.replaceFirstChar { it.uppercase() }} risk",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        PerformanceMetric(
                            label = "Score",
                            value = String.format(Locale.US, "%+.3f", pick.modelScore),
                            modifier = Modifier.weight(1f)
                        )
                        PerformanceMetric(
                            label = "Confidence",
                            value = percentLabel(pick.confidenceScore),
                            modifier = Modifier.weight(1f)
                        )
                        PerformanceMetric(
                            label = "20D / 60D",
                            value = "${signedPercentLabel(pick.metrics.momentum20d)} / ${signedPercentLabel(pick.metrics.momentum60d)}",
                            modifier = Modifier.weight(1f)
                        )
                    }
                    pick.reasons.take(3).forEach { reason ->
                        Text(
                            text = "• $reason",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    )
                } else {
                    Text(
                        text = "No monthly pick",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Text(
                        text = selection.statusReason,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    selection.bestCandidate?.let { candidate ->
                        Text(
                            text = "Closest candidate: ${candidate.symbol} at ${String.format(Locale.US, "%+.3f", candidate.modelScore)}",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PerformanceMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(10.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = value,
                style = MaterialTheme.typography.titleSmall
            )
        }
    }
}

private fun percentLabel(value: Double?): String {
    return if (value == null) {
        "N/A"
    } else {
        String.format(Locale.US, "%.0f%%", value * 100)
    }
}

private fun signedPercentLabel(value: Double?): String {
    return if (value == null) {
        "N/A"
    } else {
        String.format(Locale.US, "%+.1f%%", value * 100)
    }
}

@Composable
fun HistoryRow(entry: HistoryEntry) {
    val riskColor = when (entry.risk?.lowercase()) {
        "low" -> RiskLowColor
        "medium" -> RiskMediumColor
        "high" -> RiskHighColor
        else -> MaterialTheme.colorScheme.outline
    }

    val title = if (entry.status == SelectionStatus.PICKED) {
        "${entry.symbol} - ${entry.companyName}"
    } else {
        "No release pick"
    }

    val details = if (entry.status == SelectionStatus.PICKED) {
        buildString {
            entry.risk?.let { append(it.replaceFirstChar { ch -> ch.uppercase() }) }
            entry.modelScore?.let {
                if (isNotEmpty()) append(" • ")
                append("Score ${String.format(Locale.US, "%+.3f", it)}")
            }
        }.ifBlank { "Release archived" }
    } else {
        entry.statusReason
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Box(
            modifier = Modifier
                .width(4.dp)
                .height(52.dp)
                .background(
                    color = riskColor,
                    shape = MaterialTheme.shapes.extraLarge
                )
        )

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = entry.weekLabel,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = title,
                style = MaterialTheme.typography.bodyLarge
            )
            Text(
                text = details,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 2
            )
        }
    }
}
