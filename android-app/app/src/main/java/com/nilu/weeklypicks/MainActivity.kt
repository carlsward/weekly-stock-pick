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
