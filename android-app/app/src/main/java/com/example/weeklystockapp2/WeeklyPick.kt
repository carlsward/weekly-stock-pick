package com.example.weeklystockapp2

data class WeeklyPick(
    val symbol: String,
    val companyName: String,
    val weekStart: String,
    val weekEnd: String,
    val reasons: List<String>,
    val score: Double,
    val risk: String
)
