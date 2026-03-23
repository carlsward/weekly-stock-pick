package com.nilu.weeklypicks

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import java.time.Clock

class MainViewModel(
    private val repository: WeeklyStockRepository,
    private val localStore: LocalStore,
    private val releaseNotifier: ReleaseNotifier? = null,
    private val clock: Clock = Clock.systemDefaultZone()
) : ViewModel() {

    var uiState: UiState by mutableStateOf(UiState.Loading)
        private set

    init {
        repository.loadCachedSnapshot()?.let { snapshot ->
            applySnapshot(snapshot, isRefreshing = true)
        }
        refresh()
    }

    fun onRiskSelected(riskKey: String) {
        val content = (uiState as? UiState.Content)?.content ?: return
        if (!content.dashboard.riskSelections.containsKey(riskKey)) return

        localStore.writeSelectedRisk(riskKey)
        uiState = UiState.Content(
            content = content.copy(selectedRisk = riskKey)
        )
    }

    fun refresh() {
        val currentContent = (uiState as? UiState.Content)?.content
        if (currentContent != null) {
            uiState = UiState.Content(content = currentContent.copy(isRefreshing = true))
        } else {
            uiState = UiState.Loading
        }

        viewModelScope.launch {
            when (val result = repository.refresh()) {
                is RepositoryResult.Success -> applySnapshot(result.snapshot, isRefreshing = false)
                is RepositoryResult.Failure -> {
                    uiState = UiState.Error(result.message)
                }
            }
        }
    }

    fun retry() {
        refresh()
    }

    private fun applySnapshot(snapshot: DashboardSnapshot, isRefreshing: Boolean) {
        val selectedRisk = resolveSelectedRisk(snapshot.dashboard)
        val historyEntries = snapshot.history?.entries.orEmpty()
        val content = DashboardContent(
            dashboard = snapshot.dashboard,
            historyEntries = historyEntries,
            historyMessage = snapshot.historyMessage,
            selectedRisk = selectedRisk,
            isRefreshing = isRefreshing,
            source = snapshot.source,
            warningMessage = snapshot.warningMessage,
            freshness = snapshot.dashboard.freshness(clock),
            weeklyChange = buildWeeklyChangeSummary(snapshot.dashboard, historyEntries)
        )
        uiState = UiState.Content(content = content)
        releaseNotifier?.onContentUpdated(content)
    }

    private fun resolveSelectedRisk(dashboard: Dashboard): String {
        val savedRisk = localStore.readSelectedRisk()
        if (savedRisk != null && dashboard.riskSelections.containsKey(savedRisk)) {
            return savedRisk
        }

        val preferredOrder = listOf("medium", "low", "high")
        val availableKeys = dashboard.riskSelections.keys.toSet()
        preferredOrder.firstOrNull { it in availableKeys }?.let { preferred ->
            return preferred
        }

        dashboard.overallSelection.pick?.risk?.lowercase()?.let { overallRisk ->
            if (overallRisk in availableKeys) {
                return overallRisk
            }
        }

        return availableKeys.firstOrNull() ?: "medium"
    }

    companion object {
        fun factory(appContext: Context): ViewModelProvider.Factory {
            val localStore = SharedPreferencesLocalStore(appContext.applicationContext)
            val repository = WeeklyStockRepository(
                dashboardUrl = BuildConfig.DASHBOARD_URL,
                historyUrl = BuildConfig.HISTORY_URL,
                localStore = localStore
            )
            val releaseNotifier = ReleaseNotifier(
                context = appContext.applicationContext,
                localStore = localStore
            )

            return object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T {
                    return MainViewModel(
                        repository = repository,
                        localStore = localStore,
                        releaseNotifier = releaseNotifier
                    ) as T
                }
            }
        }
    }
}

sealed class UiState {
    data object Loading : UiState()
    data class Content(val content: DashboardContent) : UiState()
    data class Error(val message: String) : UiState()
}
