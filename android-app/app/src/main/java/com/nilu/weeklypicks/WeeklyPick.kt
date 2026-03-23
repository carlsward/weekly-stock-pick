package com.nilu.weeklypicks

import java.time.Clock
import java.time.Instant
import java.time.LocalDate

enum class SelectionStatus {
    PICKED,
    NO_PICK
}

enum class DataSource {
    NETWORK,
    CACHE
}

enum class Freshness {
    FRESH,
    STALE
}

enum class ThesisMonitorStatus {
    HEALTHY,
    WATCH,
    RISK
}

enum class ThesisSignalState {
    POSITIVE,
    WATCH,
    RISK
}

data class GenerationSummary(
    val universeSize: Int,
    val evaluatedCandidates: Int,
    val skippedSymbols: Int
)

data class SelectionThresholds(
    val overallScore: Double,
    val minimumConfidence: Double,
    val riskScores: Map<String, Double>
)

data class MarketContext(
    val timezone: String,
    val weekId: String,
    val weekLabel: String,
    val weekStart: LocalDate,
    val weekEnd: LocalDate
)

data class ScoreMetrics(
    val momentum5d: Double,
    val dailyVolatility: Double,
    val newsSentiment: Double,
    val rawNewsSentiment: Double
)

data class ScoreBreakdown(
    val momentum: Double,
    val shortMomentum: Double,
    val mediumMomentum: Double,
    val trendQuality: Double,
    val volumeConfirmation: Double,
    val volatilityPenalty: Double,
    val dailyVolatilityPenalty: Double,
    val downsidePenalty: Double,
    val drawdownPenalty: Double,
    val newsAdjustment: Double,
    val signalAlignment: Double,
    val technicalTotal: Double,
    val total: Double
)

data class NewsEvidence(
    val title: String,
    val provider: String?,
    val url: String?,
    val publishedAt: Instant?,
    val relevanceScore: Double?,
    val sentiment: Double?
)

data class ThesisSignal(
    val label: String,
    val state: ThesisSignalState,
    val value: String,
    val detail: String
)

data class ThesisMonitor(
    val status: ThesisMonitorStatus,
    val headline: String,
    val summary: String,
    val alerts: List<String>,
    val signals: List<ThesisSignal>
)

data class CandidateSnapshot(
    val symbol: String,
    val companyName: String,
    val risk: String,
    val modelScore: Double,
    val confidenceScore: Double,
    val confidenceLabel: String
)

data class WeeklyPick(
    val symbol: String,
    val companyName: String,
    val risk: String,
    val modelScore: Double,
    val confidenceScore: Double,
    val confidenceLabel: String,
    val priceAsOf: LocalDate,
    val newsAsOf: LocalDate?,
    val articleCount: Int,
    val metrics: ScoreMetrics,
    val scoreBreakdown: ScoreBreakdown,
    val reasons: List<String>,
    val newsEvidence: List<NewsEvidence>,
    val thesisMonitor: ThesisMonitor?
)

data class Selection(
    val status: SelectionStatus,
    val statusReason: String,
    val thresholdScore: Double,
    val thresholdConfidence: Double,
    val pick: WeeklyPick?,
    val bestCandidate: CandidateSnapshot?
)

data class Dashboard(
    val schemaVersion: Int,
    val modelVersion: String,
    val generatedAt: Instant,
    val dataAsOf: LocalDate,
    val expectedNextRefreshAt: Instant,
    val staleAfter: Instant,
    val marketContext: MarketContext,
    val generationSummary: GenerationSummary,
    val selectionThresholds: SelectionThresholds,
    val overallSelection: Selection,
    val riskSelections: Map<String, Selection>
)

data class DashboardSnapshot(
    val dashboard: Dashboard,
    val history: HistoryFeed?,
    val source: DataSource,
    val warningMessage: String?,
    val historyMessage: String?
)

data class DashboardContent(
    val dashboard: Dashboard,
    val historyEntries: List<HistoryEntry>,
    val historyMessage: String?,
    val selectedRisk: String,
    val isRefreshing: Boolean,
    val source: DataSource,
    val warningMessage: String?,
    val freshness: Freshness,
    val weeklyChange: WeeklyChangeSummary?
) {
    val selectedSelection: Selection
        get() = dashboard.riskSelections[selectedRisk] ?: dashboard.overallSelection
}

data class WeeklyChangeSummary(
    val title: String,
    val summary: String,
    val currentWeekLabel: String,
    val previousWeekLabel: String,
    val currentStatus: SelectionStatus,
    val previousStatus: SelectionStatus,
    val currentSymbol: String?,
    val previousSymbol: String?,
    val currentRisk: String?,
    val previousRisk: String?,
    val scoreDelta: Double?,
    val confidenceDelta: Double?
)

fun Dashboard.freshness(clock: Clock): Freshness =
    if (Instant.now(clock).isAfter(staleAfter)) {
        Freshness.STALE
    } else {
        Freshness.FRESH
    }

fun buildWeeklyChangeSummary(
    dashboard: Dashboard,
    historyEntries: List<HistoryEntry>
): WeeklyChangeSummary? {
    val previousEntry = historyEntries
        .filter { it.weekId != dashboard.marketContext.weekId }
        .maxByOrNull { it.weekStart }
        ?: return null

    val currentSelection = dashboard.overallSelection
    val currentSymbol = currentSelection.pick?.symbol ?: currentSelection.bestCandidate?.symbol
    val currentRisk = currentSelection.pick?.risk ?: currentSelection.bestCandidate?.risk
    val currentScore = currentSelection.pick?.modelScore ?: currentSelection.bestCandidate?.modelScore
    val currentConfidence = currentSelection.pick?.confidenceScore ?: currentSelection.bestCandidate?.confidenceScore
    val previousSymbol = previousEntry.symbol
    val previousRisk = previousEntry.risk

    val title = when {
        currentSelection.status == SelectionStatus.PICKED && previousEntry.status == SelectionStatus.PICKED &&
            currentSymbol != null && previousSymbol != null && currentSymbol != previousSymbol -> {
            "$currentSymbol replaced $previousSymbol"
        }

        currentSelection.status == SelectionStatus.PICKED && previousEntry.status == SelectionStatus.NO_PICK &&
            currentSymbol != null -> {
            "The release is back with $currentSymbol"
        }

        currentSelection.status == SelectionStatus.NO_PICK && previousEntry.status == SelectionStatus.PICKED &&
            previousSymbol != null -> {
            "The model passed this week"
        }

        currentSelection.status == SelectionStatus.NO_PICK && previousEntry.status == SelectionStatus.NO_PICK -> {
            "The release stayed on hold"
        }

        currentSymbol != null && previousSymbol != null && currentSymbol == previousSymbol -> {
            "$currentSymbol stayed on top"
        }

        currentSymbol != null -> {
            "$currentSymbol leads this week"
        }

        else -> "Weekly release update"
    }

    val summary = when {
        currentSelection.status == SelectionStatus.PICKED && previousEntry.status == SelectionStatus.PICKED &&
            currentSymbol != null && previousSymbol != null -> {
            "This week moved from $previousSymbol to $currentSymbol, compared with ${previousEntry.weekLabel}."
        }

        currentSelection.status == SelectionStatus.PICKED && currentSymbol != null -> {
            "A release-qualified pick is available again after ${previousEntry.weekLabel}."
        }

        previousSymbol != null -> {
            "No release-qualified pick cleared the bar this week after $previousSymbol was published for ${previousEntry.weekLabel}."
        }

        else -> {
            "The model outcome changed versus ${previousEntry.weekLabel}."
        }
    }

    return WeeklyChangeSummary(
        title = title,
        summary = summary,
        currentWeekLabel = dashboard.marketContext.weekLabel,
        previousWeekLabel = previousEntry.weekLabel,
        currentStatus = currentSelection.status,
        previousStatus = previousEntry.status,
        currentSymbol = currentSymbol,
        previousSymbol = previousSymbol,
        currentRisk = currentRisk,
        previousRisk = previousRisk,
        scoreDelta = if (currentScore != null && previousEntry.modelScore != null) {
            currentScore - previousEntry.modelScore
        } else {
            null
        },
        confidenceDelta = if (currentConfidence != null && previousEntry.confidenceScore != null) {
            currentConfidence - previousEntry.confidenceScore
        } else {
            null
        }
    )
}
