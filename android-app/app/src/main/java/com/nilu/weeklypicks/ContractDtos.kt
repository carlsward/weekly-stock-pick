package com.nilu.weeklypicks

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import java.time.Instant
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

private val contractJson = Json {
    ignoreUnknownKeys = true
    explicitNulls = false
}

class ContractParser(
    private val json: Json = contractJson
) {
    fun parseDashboard(rawJson: String): Dashboard {
        return try {
            val dto = json.decodeFromString<DashboardEnvelopeDto>(rawJson)
            dto.toModel()
        } catch (_: SerializationException) {
            parseLegacyDashboard(json, rawJson)
        }
    }

    fun parseHistory(rawJson: String): HistoryFeed {
        return try {
            val dto = json.decodeFromString<HistoryEnvelopeDto>(rawJson)
            dto.toModel()
        } catch (_: SerializationException) {
            parseLegacyHistory(json, rawJson)
        }
    }

    fun parseThesisMonitor(rawJson: String): ThesisMonitorFeed {
        val dto = json.decodeFromString<ThesisMonitorEnvelopeDto>(rawJson)
        return dto.toModel()
    }

    fun parseTrackRecord(rawJson: String): TrackRecordFeed {
        val dto = json.decodeFromString<TrackRecordEnvelopeDto>(rawJson)
        return dto.toModel()
    }

    fun parseMonthlyPick(rawJson: String): MonthlyPickFeed {
        val dto = json.decodeFromString<MonthlyPickEnvelopeDto>(rawJson)
        return dto.toModel()
    }
}

private const val DEFAULT_OVERALL_SCORE_THRESHOLD = 0.10
private const val DEFAULT_MINIMUM_CONFIDENCE = 0.55
private val DEFAULT_RISK_THRESHOLDS = mapOf(
    "low" to 0.08,
    "medium" to 0.10,
    "high" to 0.12
)

@Serializable
private data class DashboardEnvelopeDto(
    @SerialName("schema_version")
    val schemaVersion: Int,
    @SerialName("model_version")
    val modelVersion: String,
    @SerialName("generated_at")
    val generatedAt: String,
    @SerialName("data_as_of")
    val dataAsOf: String,
    @SerialName("expected_next_refresh_at")
    val expectedNextRefreshAt: String,
    @SerialName("stale_after")
    val staleAfter: String,
    @SerialName("market_context")
    val marketContext: MarketContextDto,
    @SerialName("generation_summary")
    val generationSummary: GenerationSummaryDto,
    @SerialName("selection_thresholds")
    val selectionThresholds: SelectionThresholdsDto,
    @SerialName("overall_selection")
    val overallSelection: SelectionDto,
    @SerialName("risk_selections")
    val riskSelections: Map<String, SelectionDto>
)

@Serializable
private data class MarketContextDto(
    val timezone: String,
    @SerialName("week_id")
    val weekId: String,
    @SerialName("week_label")
    val weekLabel: String,
    @SerialName("week_start")
    val weekStart: String,
    @SerialName("week_end")
    val weekEnd: String
)

@Serializable
private data class GenerationSummaryDto(
    @SerialName("universe_size")
    val universeSize: Int,
    @SerialName("evaluated_candidates")
    val evaluatedCandidates: Int,
    @SerialName("skipped_symbols")
    val skippedSymbols: Int
)

@Serializable
private data class SelectionThresholdsDto(
    @SerialName("overall_score")
    val overallScore: Double,
    @SerialName("minimum_confidence")
    val minimumConfidence: Double,
    @SerialName("risk_scores")
    val riskScores: Map<String, Double>
)

@Serializable
private data class SelectionDto(
    val status: String,
    @SerialName("status_reason")
    val statusReason: String,
    @SerialName("threshold_score")
    val thresholdScore: Double,
    @SerialName("threshold_confidence")
    val thresholdConfidence: Double,
    val pick: WeeklyPickDto? = null,
    @SerialName("best_candidate")
    val bestCandidate: CandidateSnapshotDto? = null
)

@Serializable
private data class CandidateSnapshotDto(
    val symbol: String,
    @SerialName("company_name")
    val companyName: String,
    val risk: String,
    @SerialName("model_score")
    val modelScore: Double,
    @SerialName("confidence_score")
    val confidenceScore: Double,
    @SerialName("confidence_label")
    val confidenceLabel: String
)

@Serializable
private data class WeeklyPickDto(
    val symbol: String,
    @SerialName("company_name")
    val companyName: String,
    val sector: String? = null,
    val risk: String,
    @SerialName("model_score")
    val modelScore: Double,
    @SerialName("confidence_score")
    val confidenceScore: Double,
    @SerialName("confidence_label")
    val confidenceLabel: String,
    @SerialName("price_as_of")
    val priceAsOf: String,
    @SerialName("news_as_of")
    val newsAsOf: String? = null,
    @SerialName("macro_as_of")
    val macroAsOf: String? = null,
    @SerialName("sector_as_of")
    val sectorAsOf: String? = null,
    @SerialName("article_count")
    val articleCount: Int,
    val metrics: ScoreMetricsDto,
    @SerialName("score_breakdown")
    val scoreBreakdown: ScoreBreakdownDto,
    val reasons: List<String>,
    @SerialName("news_evidence")
    val newsEvidence: List<NewsEvidenceDto> = emptyList(),
    @SerialName("thesis_monitor")
    val thesisMonitor: ThesisMonitorDto? = null
)

@Serializable
private data class ScoreMetricsDto(
    @SerialName("momentum_5d")
    val momentum5d: Double,
    @SerialName("daily_volatility")
    val dailyVolatility: Double,
    @SerialName("news_sentiment")
    val newsSentiment: Double,
    @SerialName("raw_news_sentiment")
    val rawNewsSentiment: Double
)

@Serializable
private data class ScoreBreakdownDto(
    val momentum: Double,
    @SerialName("short_momentum")
    val shortMomentum: Double = 0.0,
    @SerialName("medium_momentum")
    val mediumMomentum: Double = 0.0,
    @SerialName("trend_quality")
    val trendQuality: Double = 0.0,
    @SerialName("volume_confirmation")
    val volumeConfirmation: Double = 0.0,
    @SerialName("volatility_penalty")
    val volatilityPenalty: Double,
    @SerialName("daily_volatility_penalty")
    val dailyVolatilityPenalty: Double = 0.0,
    @SerialName("downside_penalty")
    val downsidePenalty: Double = 0.0,
    @SerialName("drawdown_penalty")
    val drawdownPenalty: Double = 0.0,
    @SerialName("news_adjustment")
    val newsAdjustment: Double,
    @SerialName("signal_alignment")
    val signalAlignment: Double = 0.0,
    @SerialName("technical_total")
    val technicalTotal: Double = 0.0,
    val total: Double
)

@Serializable
private data class NewsEvidenceDto(
    val title: String,
    val provider: String? = null,
    val url: String? = null,
    @SerialName("published_at")
    val publishedAt: String? = null,
    @SerialName("relevance_score")
    val relevanceScore: Double? = null,
    val sentiment: Double? = null
)

@Serializable
private data class ThesisMonitorDto(
    val status: String,
    val headline: String,
    val summary: String,
    val alerts: List<String> = emptyList(),
    val signals: List<ThesisSignalDto> = emptyList()
)

@Serializable
private data class ThesisSignalDto(
    val label: String,
    val state: String,
    val value: String,
    val detail: String
)

@Serializable
private data class HistoryEnvelopeDto(
    @SerialName("schema_version")
    val schemaVersion: Int,
    @SerialName("model_version")
    val modelVersion: String,
    @SerialName("generated_at")
    val generatedAt: String,
    val entries: List<HistoryEntryDto>
)

@Serializable
private data class HistoryEntryDto(
    @SerialName("week_id")
    val weekId: String,
    @SerialName("week_start")
    val weekStart: String,
    @SerialName("week_end")
    val weekEnd: String,
    @SerialName("week_label")
    val weekLabel: String,
    @SerialName("logged_at")
    val loggedAt: String,
    val status: String,
    @SerialName("status_reason")
    val statusReason: String,
    val symbol: String? = null,
    @SerialName("company_name")
    val companyName: String? = null,
    val risk: String? = null,
    @SerialName("model_score")
    val modelScore: Double? = null,
    @SerialName("confidence_score")
    val confidenceScore: Double? = null,
    @SerialName("confidence_label")
    val confidenceLabel: String? = null,
    @SerialName("data_as_of")
    val dataAsOf: String? = null
)

@Serializable
private data class ThesisMonitorEnvelopeDto(
    @SerialName("schema_version")
    val schemaVersion: Int,
    @SerialName("model_version")
    val modelVersion: String,
    @SerialName("generated_at")
    val generatedAt: String,
    @SerialName("data_as_of")
    val dataAsOf: String,
    @SerialName("expected_next_refresh_at")
    val expectedNextRefreshAt: String,
    @SerialName("stale_after")
    val staleAfter: String,
    @SerialName("market_context")
    val marketContext: MarketContextDto,
    @SerialName("source_dashboard_generated_at")
    val sourceDashboardGeneratedAt: String,
    val selection: SelectionDto,
    @SerialName("active_pick")
    val activePick: WeeklyPickDto? = null
)

@Serializable
private data class TrackRecordEnvelopeDto(
    @SerialName("schema_version")
    val schemaVersion: Int,
    @SerialName("model_version")
    val modelVersion: String,
    @SerialName("generated_at")
    val generatedAt: String,
    @SerialName("data_as_of")
    val dataAsOf: String,
    @SerialName("expected_next_refresh_at")
    val expectedNextRefreshAt: String,
    @SerialName("stale_after")
    val staleAfter: String,
    @SerialName("market_context")
    val marketContext: MarketContextDto,
    @SerialName("selection_thresholds")
    val selectionThresholds: SelectionThresholdsDto,
    val summary: TrackRecordSummaryDto,
    @SerialName("risk_breakdown")
    val riskBreakdown: Map<String, RiskTrackRecordDto>,
    val entries: List<TrackRecordEntryDto>
)

@Serializable
private data class TrackRecordSummaryDto(
    @SerialName("total_weeks")
    val totalWeeks: Int,
    @SerialName("total_picks")
    val totalPicks: Int,
    @SerialName("no_pick_weeks")
    val noPickWeeks: Int,
    @SerialName("closed_picks")
    val closedPicks: Int,
    @SerialName("open_picks")
    val openPicks: Int,
    @SerialName("win_rate")
    val winRate: Double? = null,
    @SerialName("beat_spy_rate")
    val beatSpyRate: Double? = null,
    @SerialName("beat_sector_rate")
    val beatSectorRate: Double? = null,
    @SerialName("average_5d_return")
    val average5dReturn: Double? = null,
    @SerialName("median_5d_return")
    val median5dReturn: Double? = null,
    @SerialName("average_5d_excess_return")
    val average5dExcessReturn: Double? = null,
    @SerialName("average_5d_sector_excess_return")
    val average5dSectorExcessReturn: Double? = null,
    @SerialName("compounded_5d_return")
    val compounded5dReturn: Double? = null
)

@Serializable
private data class RiskTrackRecordDto(
    @SerialName("pick_count")
    val pickCount: Int,
    @SerialName("closed_pick_count")
    val closedPickCount: Int,
    @SerialName("win_rate")
    val winRate: Double? = null,
    @SerialName("average_5d_return")
    val average5dReturn: Double? = null,
    @SerialName("average_5d_excess_return")
    val average5dExcessReturn: Double? = null
)

@Serializable
private data class TrackRecordEntryDto(
    @SerialName("week_id")
    val weekId: String,
    @SerialName("week_start")
    val weekStart: String,
    @SerialName("week_end")
    val weekEnd: String,
    @SerialName("week_label")
    val weekLabel: String,
    @SerialName("logged_at")
    val loggedAt: String,
    val status: String,
    @SerialName("status_reason")
    val statusReason: String,
    val symbol: String? = null,
    @SerialName("company_name")
    val companyName: String? = null,
    val sector: String? = null,
    val risk: String? = null,
    @SerialName("model_score")
    val modelScore: Double? = null,
    @SerialName("confidence_score")
    val confidenceScore: Double? = null,
    @SerialName("confidence_label")
    val confidenceLabel: String? = null,
    @SerialName("data_as_of")
    val dataAsOf: String? = null,
    @SerialName("realized_5d_return")
    val realized5dReturn: Double? = null,
    @SerialName("realized_5d_excess_return")
    val realized5dExcessReturn: Double? = null,
    @SerialName("realized_5d_sector_return")
    val realized5dSectorReturn: Double? = null,
    @SerialName("realized_5d_sector_excess_return")
    val realized5dSectorExcessReturn: Double? = null,
    val outcome: String? = null
)

@Serializable
private data class MonthlyPickEnvelopeDto(
    @SerialName("schema_version")
    val schemaVersion: Int,
    @SerialName("model_version")
    val modelVersion: String,
    @SerialName("generated_at")
    val generatedAt: String,
    @SerialName("data_as_of")
    val dataAsOf: String,
    @SerialName("expected_next_refresh_at")
    val expectedNextRefreshAt: String,
    @SerialName("stale_after")
    val staleAfter: String,
    @SerialName("period_context")
    val periodContext: MonthlyPeriodContextDto,
    @SerialName("generation_summary")
    val generationSummary: GenerationSummaryDto,
    @SerialName("selection_thresholds")
    val selectionThresholds: MonthlySelectionThresholdsDto,
    val selection: MonthlySelectionDto
)

@Serializable
private data class MonthlyPeriodContextDto(
    val timezone: String,
    @SerialName("month_id")
    val monthId: String,
    @SerialName("month_label")
    val monthLabel: String,
    @SerialName("month_start")
    val monthStart: String,
    @SerialName("month_end")
    val monthEnd: String,
    @SerialName("rebalance_date")
    val rebalanceDate: String,
    @SerialName("horizon_trading_days")
    val horizonTradingDays: Int
)

@Serializable
private data class MonthlySelectionThresholdsDto(
    @SerialName("overall_score")
    val overallScore: Double,
    @SerialName("minimum_confidence")
    val minimumConfidence: Double
)

@Serializable
private data class MonthlySelectionDto(
    val status: String,
    @SerialName("status_reason")
    val statusReason: String,
    @SerialName("threshold_score")
    val thresholdScore: Double,
    @SerialName("threshold_confidence")
    val thresholdConfidence: Double,
    val pick: MonthlyPickCandidateDto? = null,
    @SerialName("best_candidate")
    val bestCandidate: CandidateSnapshotDto? = null
)

@Serializable
private data class MonthlyPickCandidateDto(
    val symbol: String,
    @SerialName("company_name")
    val companyName: String,
    val sector: String,
    val risk: String,
    @SerialName("model_score")
    val modelScore: Double,
    @SerialName("confidence_score")
    val confidenceScore: Double,
    @SerialName("confidence_label")
    val confidenceLabel: String,
    @SerialName("price_as_of")
    val priceAsOf: String,
    @SerialName("news_as_of")
    val newsAsOf: String? = null,
    @SerialName("macro_as_of")
    val macroAsOf: String? = null,
    @SerialName("sector_as_of")
    val sectorAsOf: String? = null,
    @SerialName("article_count")
    val articleCount: Int,
    val metrics: MonthlyScoreMetricsDto,
    @SerialName("score_breakdown")
    val scoreBreakdown: MonthlyScoreBreakdownDto,
    val reasons: List<String>,
    @SerialName("news_evidence")
    val newsEvidence: List<NewsEvidenceDto> = emptyList(),
    @SerialName("macro_evidence")
    val macroEvidence: List<NewsEvidenceDto> = emptyList()
)

@Serializable
private data class MonthlyScoreMetricsDto(
    @SerialName("momentum_20d")
    val momentum20d: Double,
    @SerialName("momentum_60d")
    val momentum60d: Double,
    @SerialName("daily_volatility")
    val dailyVolatility: Double,
    @SerialName("market_relative_20d")
    val marketRelative20d: Double,
    @SerialName("market_relative_60d")
    val marketRelative60d: Double,
    @SerialName("sector_relative_20d")
    val sectorRelative20d: Double,
    @SerialName("sector_relative_60d")
    val sectorRelative60d: Double,
    @SerialName("news_sentiment")
    val newsSentiment: Double,
    @SerialName("news_confidence")
    val newsConfidence: Double,
    @SerialName("macro_sentiment")
    val macroSentiment: Double,
    @SerialName("macro_confidence")
    val macroConfidence: Double,
    @SerialName("sector_sentiment")
    val sectorSentiment: Double,
    @SerialName("sector_confidence")
    val sectorConfidence: Double
)

@Serializable
private data class MonthlyScoreBreakdownDto(
    @SerialName("trend_strength")
    val trendStrength: Double,
    @SerialName("relative_strength")
    val relativeStrength: Double,
    val participation: Double,
    @SerialName("risk_control")
    val riskControl: Double,
    @SerialName("technical_total")
    val technicalTotal: Double,
    @SerialName("news_adjustment")
    val newsAdjustment: Double,
    @SerialName("macro_adjustment")
    val macroAdjustment: Double,
    @SerialName("sector_adjustment")
    val sectorAdjustment: Double,
    @SerialName("signal_alignment")
    val signalAlignment: Double,
    val total: Double
)

@Serializable
private data class LegacyDashboardEnvelopeDto(
    val low: LegacyRiskPickDto? = null,
    val medium: LegacyRiskPickDto? = null,
    val high: LegacyRiskPickDto? = null
)

@Serializable
private data class LegacyRiskPickDto(
    val symbol: String,
    @SerialName("company_name")
    val companyName: String,
    @SerialName("week_start")
    val weekStart: String,
    @SerialName("week_end")
    val weekEnd: String,
    val reasons: List<String> = emptyList(),
    val score: Double = 0.0,
    val risk: String = "unknown",
    @SerialName("model_version")
    val modelVersion: String? = null
)

@Serializable
private data class LegacyHistoryEntryDto(
    @SerialName("logged_at")
    val loggedAt: String,
    val symbol: String? = null,
    @SerialName("company_name")
    val companyName: String? = null,
    @SerialName("week_start")
    val weekStart: String,
    @SerialName("week_end")
    val weekEnd: String,
    val score: Double? = null,
    val risk: String? = null,
    @SerialName("model_version")
    val modelVersion: String? = null
)

private fun DashboardEnvelopeDto.toModel(): Dashboard {
    return Dashboard(
        schemaVersion = schemaVersion,
        modelVersion = modelVersion,
        generatedAt = Instant.parse(generatedAt),
        dataAsOf = LocalDate.parse(dataAsOf),
        expectedNextRefreshAt = Instant.parse(expectedNextRefreshAt),
        staleAfter = Instant.parse(staleAfter),
        marketContext = marketContext.toModel(),
        generationSummary = generationSummary.toModel(),
        selectionThresholds = selectionThresholds.toModel(),
        overallSelection = overallSelection.toModel(),
        riskSelections = riskSelections.mapValues { (_, selection) -> selection.toModel() }
    )
}

private fun MarketContextDto.toModel(): MarketContext {
    return MarketContext(
        timezone = timezone,
        weekId = weekId,
        weekLabel = weekLabel,
        weekStart = LocalDate.parse(weekStart),
        weekEnd = LocalDate.parse(weekEnd)
    )
}

private fun GenerationSummaryDto.toModel(): GenerationSummary {
    return GenerationSummary(
        universeSize = universeSize,
        evaluatedCandidates = evaluatedCandidates,
        skippedSymbols = skippedSymbols
    )
}

private fun SelectionThresholdsDto.toModel(): SelectionThresholds {
    return SelectionThresholds(
        overallScore = overallScore,
        minimumConfidence = minimumConfidence,
        riskScores = riskScores
    )
}

private fun SelectionDto.toModel(): Selection {
    return Selection(
        status = when (status.lowercase()) {
            "picked" -> SelectionStatus.PICKED
            else -> SelectionStatus.NO_PICK
        },
        statusReason = statusReason,
        thresholdScore = thresholdScore,
        thresholdConfidence = thresholdConfidence,
        pick = pick?.toModel(),
        bestCandidate = bestCandidate?.toModel()
    )
}

private fun CandidateSnapshotDto.toModel(): CandidateSnapshot {
    return CandidateSnapshot(
        symbol = symbol,
        companyName = companyName,
        risk = risk,
        modelScore = modelScore,
        confidenceScore = confidenceScore,
        confidenceLabel = confidenceLabel
    )
}

private fun WeeklyPickDto.toModel(): WeeklyPick {
    return WeeklyPick(
        symbol = symbol,
        companyName = companyName,
        sector = sector,
        risk = risk,
        modelScore = modelScore,
        confidenceScore = confidenceScore,
        confidenceLabel = confidenceLabel,
        priceAsOf = LocalDate.parse(priceAsOf),
        newsAsOf = newsAsOf?.let(LocalDate::parse),
        macroAsOf = macroAsOf?.let(LocalDate::parse),
        sectorAsOf = sectorAsOf?.let(LocalDate::parse),
        articleCount = articleCount,
        metrics = metrics.toModel(),
        scoreBreakdown = scoreBreakdown.toModel(),
        reasons = reasons,
        newsEvidence = newsEvidence.map { it.toModel() },
        thesisMonitor = thesisMonitor?.toModel()
    )
}

private fun ScoreMetricsDto.toModel(): ScoreMetrics {
    return ScoreMetrics(
        momentum5d = momentum5d,
        dailyVolatility = dailyVolatility,
        newsSentiment = newsSentiment,
        rawNewsSentiment = rawNewsSentiment
    )
}

private fun ScoreBreakdownDto.toModel(): ScoreBreakdown {
    return ScoreBreakdown(
        momentum = momentum,
        shortMomentum = shortMomentum,
        mediumMomentum = mediumMomentum,
        trendQuality = trendQuality,
        volumeConfirmation = volumeConfirmation,
        volatilityPenalty = volatilityPenalty,
        dailyVolatilityPenalty = dailyVolatilityPenalty,
        downsidePenalty = downsidePenalty,
        drawdownPenalty = drawdownPenalty,
        newsAdjustment = newsAdjustment,
        signalAlignment = signalAlignment,
        technicalTotal = technicalTotal,
        total = total
    )
}

private fun NewsEvidenceDto.toModel(): NewsEvidence {
    return NewsEvidence(
        title = title,
        provider = provider,
        url = url,
        publishedAt = publishedAt?.let { runCatching { Instant.parse(it) }.getOrNull() },
        relevanceScore = relevanceScore,
        sentiment = sentiment
    )
}

private fun ThesisMonitorDto.toModel(): ThesisMonitor {
    return ThesisMonitor(
        status = when (status.lowercase()) {
            "risk" -> ThesisMonitorStatus.RISK
            "watch" -> ThesisMonitorStatus.WATCH
            else -> ThesisMonitorStatus.HEALTHY
        },
        headline = headline,
        summary = summary,
        alerts = alerts,
        signals = signals.map { it.toModel() }
    )
}

private fun ThesisSignalDto.toModel(): ThesisSignal {
    return ThesisSignal(
        label = label,
        state = when (state.lowercase()) {
            "risk" -> ThesisSignalState.RISK
            "watch" -> ThesisSignalState.WATCH
            else -> ThesisSignalState.POSITIVE
        },
        value = value,
        detail = detail
    )
}

private fun HistoryEnvelopeDto.toModel(): HistoryFeed {
    return HistoryFeed(
        schemaVersion = schemaVersion,
        modelVersion = modelVersion,
        generatedAt = Instant.parse(generatedAt),
        entries = entries.map { it.toModel() }
    )
}

private fun HistoryEntryDto.toModel(): HistoryEntry {
    return HistoryEntry(
        weekId = weekId,
        weekStart = LocalDate.parse(weekStart),
        weekEnd = LocalDate.parse(weekEnd),
        weekLabel = weekLabel,
        loggedAt = Instant.parse(loggedAt),
        status = when (status.lowercase()) {
            "picked" -> SelectionStatus.PICKED
            else -> SelectionStatus.NO_PICK
        },
        statusReason = statusReason,
        symbol = symbol,
        companyName = companyName,
        risk = risk,
        modelScore = modelScore,
        confidenceScore = confidenceScore,
        confidenceLabel = confidenceLabel,
        dataAsOf = dataAsOf?.let(LocalDate::parse)
    )
}

private fun ThesisMonitorEnvelopeDto.toModel(): ThesisMonitorFeed {
    return ThesisMonitorFeed(
        schemaVersion = schemaVersion,
        modelVersion = modelVersion,
        generatedAt = Instant.parse(generatedAt),
        dataAsOf = LocalDate.parse(dataAsOf),
        expectedNextRefreshAt = Instant.parse(expectedNextRefreshAt),
        staleAfter = Instant.parse(staleAfter),
        marketContext = marketContext.toModel(),
        sourceDashboardGeneratedAt = Instant.parse(sourceDashboardGeneratedAt),
        selection = selection.toModel(),
        activePick = activePick?.toModel()
    )
}

private fun TrackRecordEnvelopeDto.toModel(): TrackRecordFeed {
    return TrackRecordFeed(
        schemaVersion = schemaVersion,
        modelVersion = modelVersion,
        generatedAt = Instant.parse(generatedAt),
        dataAsOf = LocalDate.parse(dataAsOf),
        expectedNextRefreshAt = Instant.parse(expectedNextRefreshAt),
        staleAfter = Instant.parse(staleAfter),
        marketContext = marketContext.toModel(),
        selectionThresholds = selectionThresholds.toModel(),
        summary = summary.toModel(),
        riskBreakdown = riskBreakdown.mapValues { (_, value) -> value.toModel() },
        entries = entries.map { it.toModel() }
    )
}

private fun TrackRecordSummaryDto.toModel(): TrackRecordSummary {
    return TrackRecordSummary(
        totalWeeks = totalWeeks,
        totalPicks = totalPicks,
        noPickWeeks = noPickWeeks,
        closedPicks = closedPicks,
        openPicks = openPicks,
        winRate = winRate,
        beatSpyRate = beatSpyRate,
        beatSectorRate = beatSectorRate,
        average5dReturn = average5dReturn,
        median5dReturn = median5dReturn,
        average5dExcessReturn = average5dExcessReturn,
        average5dSectorExcessReturn = average5dSectorExcessReturn,
        compounded5dReturn = compounded5dReturn
    )
}

private fun RiskTrackRecordDto.toModel(): RiskTrackRecord {
    return RiskTrackRecord(
        pickCount = pickCount,
        closedPickCount = closedPickCount,
        winRate = winRate,
        average5dReturn = average5dReturn,
        average5dExcessReturn = average5dExcessReturn
    )
}

private fun TrackRecordEntryDto.toModel(): TrackRecordEntry {
    return TrackRecordEntry(
        weekId = weekId,
        weekStart = LocalDate.parse(weekStart),
        weekEnd = LocalDate.parse(weekEnd),
        weekLabel = weekLabel,
        loggedAt = Instant.parse(loggedAt),
        status = when (status.lowercase()) {
            "picked" -> SelectionStatus.PICKED
            else -> SelectionStatus.NO_PICK
        },
        statusReason = statusReason,
        symbol = symbol,
        companyName = companyName,
        sector = sector,
        risk = risk,
        modelScore = modelScore,
        confidenceScore = confidenceScore,
        confidenceLabel = confidenceLabel,
        dataAsOf = dataAsOf?.let(LocalDate::parse),
        realized5dReturn = realized5dReturn,
        realized5dExcessReturn = realized5dExcessReturn,
        realized5dSectorReturn = realized5dSectorReturn,
        realized5dSectorExcessReturn = realized5dSectorExcessReturn,
        outcome = outcome
    )
}

private fun MonthlyPickEnvelopeDto.toModel(): MonthlyPickFeed {
    return MonthlyPickFeed(
        schemaVersion = schemaVersion,
        modelVersion = modelVersion,
        generatedAt = Instant.parse(generatedAt),
        dataAsOf = LocalDate.parse(dataAsOf),
        expectedNextRefreshAt = Instant.parse(expectedNextRefreshAt),
        staleAfter = Instant.parse(staleAfter),
        periodContext = periodContext.toModel(),
        generationSummary = generationSummary.toModel(),
        selectionThresholds = selectionThresholds.toModel(),
        selection = selection.toModel()
    )
}

private fun MonthlyPeriodContextDto.toModel(): MonthlyPeriodContext {
    return MonthlyPeriodContext(
        timezone = timezone,
        monthId = monthId,
        monthLabel = monthLabel,
        monthStart = LocalDate.parse(monthStart),
        monthEnd = LocalDate.parse(monthEnd),
        rebalanceDate = LocalDate.parse(rebalanceDate),
        horizonTradingDays = horizonTradingDays
    )
}

private fun MonthlySelectionThresholdsDto.toModel(): MonthlySelectionThresholds {
    return MonthlySelectionThresholds(
        overallScore = overallScore,
        minimumConfidence = minimumConfidence
    )
}

private fun MonthlySelectionDto.toModel(): MonthlySelection {
    return MonthlySelection(
        status = when (status.lowercase()) {
            "picked" -> SelectionStatus.PICKED
            else -> SelectionStatus.NO_PICK
        },
        statusReason = statusReason,
        thresholdScore = thresholdScore,
        thresholdConfidence = thresholdConfidence,
        pick = pick?.toModel(),
        bestCandidate = bestCandidate?.toModel()
    )
}

private fun MonthlyPickCandidateDto.toModel(): MonthlyPickCandidate {
    return MonthlyPickCandidate(
        symbol = symbol,
        companyName = companyName,
        sector = sector,
        risk = risk,
        modelScore = modelScore,
        confidenceScore = confidenceScore,
        confidenceLabel = confidenceLabel,
        priceAsOf = LocalDate.parse(priceAsOf),
        newsAsOf = newsAsOf?.let(LocalDate::parse),
        macroAsOf = macroAsOf?.let(LocalDate::parse),
        sectorAsOf = sectorAsOf?.let(LocalDate::parse),
        articleCount = articleCount,
        metrics = metrics.toModel(),
        scoreBreakdown = scoreBreakdown.toModel(),
        reasons = reasons,
        newsEvidence = newsEvidence.map { it.toModel() },
        macroEvidence = macroEvidence.map { it.toModel() }
    )
}

private fun MonthlyScoreMetricsDto.toModel(): MonthlyScoreMetrics {
    return MonthlyScoreMetrics(
        momentum20d = momentum20d,
        momentum60d = momentum60d,
        dailyVolatility = dailyVolatility,
        marketRelative20d = marketRelative20d,
        marketRelative60d = marketRelative60d,
        sectorRelative20d = sectorRelative20d,
        sectorRelative60d = sectorRelative60d,
        newsSentiment = newsSentiment,
        newsConfidence = newsConfidence,
        macroSentiment = macroSentiment,
        macroConfidence = macroConfidence,
        sectorSentiment = sectorSentiment,
        sectorConfidence = sectorConfidence
    )
}

private fun MonthlyScoreBreakdownDto.toModel(): MonthlyScoreBreakdown {
    return MonthlyScoreBreakdown(
        trendStrength = trendStrength,
        relativeStrength = relativeStrength,
        participation = participation,
        riskControl = riskControl,
        technicalTotal = technicalTotal,
        newsAdjustment = newsAdjustment,
        macroAdjustment = macroAdjustment,
        sectorAdjustment = sectorAdjustment,
        signalAlignment = signalAlignment,
        total = total
    )
}

private fun parseLegacyDashboard(json: Json, rawJson: String): Dashboard {
    val dto = json.decodeFromString<LegacyDashboardEnvelopeDto>(rawJson)
    return dto.toModel()
}

private fun parseLegacyHistory(json: Json, rawJson: String): HistoryFeed {
    val entries = json.decodeFromString<List<LegacyHistoryEntryDto>>(rawJson)
    return entries.toLegacyHistoryModel()
}

private fun LegacyDashboardEnvelopeDto.toModel(): Dashboard {
    val legacyByRisk = linkedMapOf<String, LegacyRiskPickDto>()
    low?.let { legacyByRisk["low"] = it }
    medium?.let { legacyByRisk["medium"] = it }
    high?.let { legacyByRisk["high"] = it }

    val anchor = legacyByRisk.values.firstOrNull()
        ?: throw SerializationException("Legacy dashboard payload was empty.")
    val weekStart = LocalDate.parse(anchor.weekStart)
    val weekEnd = LocalDate.parse(anchor.weekEnd)
    val generatedAt = isoAtUtc(weekStart)
    val expectedNextRefreshAt = isoAtUtc(weekEnd.plusDays(3))
    val staleAfter = expectedNextRefreshAt.plusSeconds(12 * 60 * 60L)
    val validPicks = legacyByRisk.values.filter { it.isValidPick() }
    val bestPick = validPicks.maxByOrNull { it.score }
    val modelVersion = bestPick?.modelVersion ?: anchor.modelVersion ?: "legacy-v1"

    return Dashboard(
        schemaVersion = 1,
        modelVersion = modelVersion,
        generatedAt = generatedAt,
        dataAsOf = weekEnd,
        expectedNextRefreshAt = expectedNextRefreshAt,
        staleAfter = staleAfter,
        marketContext = MarketContext(
            timezone = "America/New_York",
            weekId = legacyWeekId(weekStart),
            weekLabel = legacyWeekLabel(weekStart, weekEnd),
            weekStart = weekStart,
            weekEnd = weekEnd
        ),
        generationSummary = GenerationSummary(
            universeSize = 0,
            evaluatedCandidates = legacyByRisk.size,
            skippedSymbols = 0
        ),
        selectionThresholds = SelectionThresholds(
            overallScore = DEFAULT_OVERALL_SCORE_THRESHOLD,
            minimumConfidence = DEFAULT_MINIMUM_CONFIDENCE,
            riskScores = DEFAULT_RISK_THRESHOLDS
        ),
        overallSelection = bestPick.toLegacySelection(
            thresholdScore = DEFAULT_OVERALL_SCORE_THRESHOLD,
            statusReason = if (bestPick != null) {
                "Loaded from the legacy dashboard feed on GitHub main. This data format is older than the local app contract."
            } else {
                "Legacy dashboard feed did not contain a usable release pick."
            }
        ),
        riskSelections = legacyByRisk.mapValues { (riskKey, pick) ->
            pick.toLegacySelection(
                thresholdScore = DEFAULT_RISK_THRESHOLDS[riskKey] ?: DEFAULT_OVERALL_SCORE_THRESHOLD,
                statusReason = if (pick.isValidPick()) {
                    "Loaded from the legacy dashboard feed on GitHub main."
                } else {
                    "Legacy dashboard feed did not contain a usable ${riskKey}-risk pick."
                }
            )
        }
    )
}

private fun LegacyHistoryEntryDto.toModel(): HistoryEntry {
    val weekStartDate = LocalDate.parse(weekStart)
    val weekEndDate = LocalDate.parse(weekEnd)
    val hasPick = !symbol.isNullOrBlank() && symbol != "N/A"
    return HistoryEntry(
        weekId = legacyWeekId(weekStartDate),
        weekStart = weekStartDate,
        weekEnd = weekEndDate,
        weekLabel = legacyWeekLabel(weekStartDate, weekEndDate),
        loggedAt = parseLegacyInstant(loggedAt),
        status = if (hasPick) SelectionStatus.PICKED else SelectionStatus.NO_PICK,
        statusReason = if (hasPick) {
            "Migrated from the legacy history feed."
        } else {
            "Legacy history entry did not contain a usable pick."
        },
        symbol = symbol,
        companyName = companyName,
        risk = risk,
        modelScore = score,
        confidenceScore = null,
        confidenceLabel = null,
        dataAsOf = weekEndDate
    )
}

private fun List<LegacyHistoryEntryDto>.toLegacyHistoryModel(): HistoryFeed {
    val sortedEntries = map { it.toModel() }.sortedByDescending { it.weekStart }
    val generatedAt = sortedEntries.maxOfOrNull { it.loggedAt } ?: Instant.EPOCH
    val modelVersion = lastOrNull()?.modelVersion ?: "legacy-v1"
    return HistoryFeed(
        schemaVersion = 1,
        modelVersion = modelVersion,
        generatedAt = generatedAt,
        entries = sortedEntries
    )
}

private fun LegacyRiskPickDto?.toLegacySelection(
    thresholdScore: Double,
    statusReason: String
): Selection {
    val pick = this?.takeIf { it.isValidPick() }?.toLegacyWeeklyPick()
    return Selection(
        status = if (pick != null) SelectionStatus.PICKED else SelectionStatus.NO_PICK,
        statusReason = statusReason,
        thresholdScore = thresholdScore,
        thresholdConfidence = DEFAULT_MINIMUM_CONFIDENCE,
        pick = pick,
        bestCandidate = if (pick != null) {
            CandidateSnapshot(
                symbol = pick.symbol,
                companyName = pick.companyName,
                risk = pick.risk,
                modelScore = pick.modelScore,
                confidenceScore = pick.confidenceScore,
                confidenceLabel = pick.confidenceLabel
            )
        } else {
            null
        }
    )
}

private fun LegacyRiskPickDto.toLegacyWeeklyPick(): WeeklyPick {
    val weekStartDate = LocalDate.parse(weekStart)
    val weekEndDate = LocalDate.parse(weekEnd)
    return WeeklyPick(
        symbol = symbol,
        companyName = companyName,
        risk = risk,
        modelScore = score,
        confidenceScore = 0.35,
        confidenceLabel = "low",
        priceAsOf = weekEndDate,
        newsAsOf = null,
        articleCount = 0,
        metrics = ScoreMetrics(
            momentum5d = 0.0,
            dailyVolatility = 0.0,
            newsSentiment = 0.5,
            rawNewsSentiment = 0.0
        ),
        scoreBreakdown = ScoreBreakdown(
            momentum = score,
            shortMomentum = 0.0,
            mediumMomentum = 0.0,
            trendQuality = 0.0,
            volumeConfirmation = 0.0,
            volatilityPenalty = 0.0,
            dailyVolatilityPenalty = 0.0,
            downsidePenalty = 0.0,
            drawdownPenalty = 0.0,
            newsAdjustment = 0.0,
            signalAlignment = 0.0,
            technicalTotal = score,
            total = score
        ),
        reasons = if (reasons.isNotEmpty()) {
            reasons
        } else {
            listOf("Loaded from the legacy dashboard feed.")
        },
        newsEvidence = emptyList(),
        thesisMonitor = null
    )
}

private fun LegacyRiskPickDto.isValidPick(): Boolean {
    return symbol.isNotBlank() && symbol != "N/A" && companyName.isNotBlank()
}

private fun parseLegacyInstant(value: String): Instant {
    return runCatching { Instant.parse(value) }
        .getOrElse {
            runCatching {
                LocalDate.parse(value).atStartOfDay().toInstant(ZoneOffset.UTC)
            }.getOrElse {
                LocalDateTime.parse(value, DateTimeFormatter.ISO_LOCAL_DATE_TIME).toInstant(ZoneOffset.UTC)
            }
        }
}

private fun isoAtUtc(date: LocalDate, hour: Int = 12): Instant {
    return date.atTime(hour, 0).toInstant(ZoneOffset.UTC)
}

private fun legacyWeekId(weekStart: LocalDate): String {
    val isoWeek = weekStart.get(java.time.temporal.IsoFields.WEEK_OF_WEEK_BASED_YEAR)
    val isoYear = weekStart.get(java.time.temporal.IsoFields.WEEK_BASED_YEAR)
    return "%d-W%02d".format(isoYear, isoWeek)
}

private fun legacyWeekLabel(weekStart: LocalDate, weekEnd: LocalDate): String {
    return "${weekStart} - ${weekEnd}"
}
