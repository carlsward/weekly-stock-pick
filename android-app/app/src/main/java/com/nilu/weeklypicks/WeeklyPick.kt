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
    val sector: String? = null,
    val risk: String,
    val modelScore: Double,
    val confidenceScore: Double,
    val confidenceLabel: String,
    val priceAsOf: LocalDate,
    val newsAsOf: LocalDate?,
    val macroAsOf: LocalDate? = null,
    val sectorAsOf: LocalDate? = null,
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
    val trackRecord: TrackRecordFeed?,
    val source: DataSource,
    val warningMessage: String?,
    val historyMessage: String?,
    val thesisMonitorMessage: String?,
    val trackRecordMessage: String?,
    val monthlyPick: MonthlyPickFeed? = null,
    val monthlyPickMessage: String? = null
)

data class DashboardContent(
    val dashboard: Dashboard,
    val historyEntries: List<HistoryEntry>,
    val trackRecord: TrackRecordFeed?,
    val historyMessage: String?,
    val thesisMonitorMessage: String?,
    val trackRecordMessage: String?,
    val selectedRisk: String,
    val isRefreshing: Boolean,
    val source: DataSource,
    val warningMessage: String?,
    val freshness: Freshness,
    val weeklyChange: WeeklyChangeSummary?,
    val monthlyPick: MonthlyPickFeed? = null,
    val monthlyPickMessage: String? = null
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

data class ThesisMonitorFeed(
    val schemaVersion: Int,
    val modelVersion: String,
    val generatedAt: Instant,
    val dataAsOf: LocalDate,
    val expectedNextRefreshAt: Instant,
    val staleAfter: Instant,
    val marketContext: MarketContext,
    val sourceDashboardGeneratedAt: Instant,
    val selection: Selection,
    val activePick: WeeklyPick?
)

data class TrackRecordSummary(
    val totalWeeks: Int,
    val totalPicks: Int,
    val noPickWeeks: Int,
    val closedPicks: Int,
    val openPicks: Int,
    val winRate: Double?,
    val beatSpyRate: Double?,
    val beatSectorRate: Double?,
    val average5dReturn: Double?,
    val median5dReturn: Double?,
    val average5dExcessReturn: Double?,
    val average5dSectorExcessReturn: Double?,
    val compounded5dReturn: Double?
)

data class RiskTrackRecord(
    val pickCount: Int,
    val closedPickCount: Int,
    val winRate: Double?,
    val average5dReturn: Double?,
    val average5dExcessReturn: Double?
)

data class TrackRecordEntry(
    val weekId: String,
    val weekStart: LocalDate,
    val weekEnd: LocalDate,
    val weekLabel: String,
    val loggedAt: Instant,
    val status: SelectionStatus,
    val statusReason: String,
    val symbol: String?,
    val companyName: String?,
    val sector: String?,
    val risk: String?,
    val modelScore: Double?,
    val confidenceScore: Double?,
    val confidenceLabel: String?,
    val dataAsOf: LocalDate?,
    val realized5dReturn: Double?,
    val realized5dExcessReturn: Double?,
    val realized5dSectorReturn: Double?,
    val realized5dSectorExcessReturn: Double?,
    val outcome: String?
)

data class TrackRecordFeed(
    val schemaVersion: Int,
    val modelVersion: String,
    val generatedAt: Instant,
    val dataAsOf: LocalDate,
    val expectedNextRefreshAt: Instant,
    val staleAfter: Instant,
    val marketContext: MarketContext,
    val selectionThresholds: SelectionThresholds,
    val summary: TrackRecordSummary,
    val riskBreakdown: Map<String, RiskTrackRecord>,
    val entries: List<TrackRecordEntry>
)

data class MonthlyPeriodContext(
    val timezone: String,
    val monthId: String,
    val monthLabel: String,
    val monthStart: LocalDate,
    val monthEnd: LocalDate,
    val rebalanceDate: LocalDate,
    val horizonTradingDays: Int
)

data class MonthlySelectionThresholds(
    val overallScore: Double,
    val minimumConfidence: Double
)

data class MonthlyScoreMetrics(
    val momentum20d: Double,
    val momentum60d: Double,
    val dailyVolatility: Double,
    val marketRelative20d: Double,
    val marketRelative60d: Double,
    val sectorRelative20d: Double,
    val sectorRelative60d: Double,
    val newsSentiment: Double,
    val newsConfidence: Double,
    val macroSentiment: Double,
    val macroConfidence: Double,
    val sectorSentiment: Double,
    val sectorConfidence: Double
)

data class MonthlyScoreBreakdown(
    val trendStrength: Double,
    val relativeStrength: Double,
    val participation: Double,
    val riskControl: Double,
    val technicalTotal: Double,
    val newsAdjustment: Double,
    val macroAdjustment: Double,
    val sectorAdjustment: Double,
    val signalAlignment: Double,
    val total: Double
)

data class MonthlyPickCandidate(
    val symbol: String,
    val companyName: String,
    val sector: String,
    val risk: String,
    val modelScore: Double,
    val confidenceScore: Double,
    val confidenceLabel: String,
    val priceAsOf: LocalDate,
    val newsAsOf: LocalDate?,
    val macroAsOf: LocalDate?,
    val sectorAsOf: LocalDate?,
    val articleCount: Int,
    val metrics: MonthlyScoreMetrics,
    val scoreBreakdown: MonthlyScoreBreakdown,
    val reasons: List<String>,
    val newsEvidence: List<NewsEvidence>,
    val macroEvidence: List<NewsEvidence>
)

data class MonthlySelection(
    val status: SelectionStatus,
    val statusReason: String,
    val thresholdScore: Double,
    val thresholdConfidence: Double,
    val pick: MonthlyPickCandidate?,
    val bestCandidate: CandidateSnapshot?
)

data class MonthlyPickFeed(
    val schemaVersion: Int,
    val modelVersion: String,
    val generatedAt: Instant,
    val dataAsOf: LocalDate,
    val expectedNextRefreshAt: Instant,
    val staleAfter: Instant,
    val periodContext: MonthlyPeriodContext,
    val generationSummary: GenerationSummary,
    val selectionThresholds: MonthlySelectionThresholds,
    val selection: MonthlySelection
)

fun Dashboard.withLiveThesisMonitor(feed: ThesisMonitorFeed?): Dashboard {
    val currentPick = overallSelection.pick ?: return this
    val livePick = feed?.activePick ?: return this
    if (feed.selection.status != SelectionStatus.PICKED) {
        return this
    }
    if (feed.marketContext.weekId != marketContext.weekId) {
        return this
    }
    if (livePick.symbol != currentPick.symbol) {
        return this
    }

    val updatedOverall = overallSelection.copy(pick = livePick)
    val updatedRiskSelections = riskSelections.mapValues { (_, selection) ->
        val selectionPick = selection.pick
        if (selectionPick != null && selectionPick.symbol == livePick.symbol) {
            selection.copy(pick = livePick)
        } else {
            selection
        }
    }
    return copy(
        overallSelection = updatedOverall,
        riskSelections = updatedRiskSelections
    )
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
