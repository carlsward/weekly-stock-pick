package com.example.weeklystockapp2

data class HistoryEntry(
    val loggedAt: String,
    val symbol: String,
    val companyName: String,
    val weekStart: String,
    val weekEnd: String,
    val score: Double?,
    val risk: String
)
