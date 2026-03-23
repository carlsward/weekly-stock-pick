package com.nilu.weeklypicks

import java.time.Instant
import java.time.LocalDate

data class HistoryEntry(
    val weekId: String,
    val weekStart: LocalDate,
    val weekEnd: LocalDate,
    val weekLabel: String,
    val loggedAt: Instant,
    val status: SelectionStatus,
    val statusReason: String,
    val symbol: String?,
    val companyName: String?,
    val risk: String?,
    val modelScore: Double?,
    val confidenceScore: Double?,
    val confidenceLabel: String?,
    val dataAsOf: LocalDate?
)

data class HistoryFeed(
    val schemaVersion: Int,
    val modelVersion: String,
    val generatedAt: Instant,
    val entries: List<HistoryEntry>
)
