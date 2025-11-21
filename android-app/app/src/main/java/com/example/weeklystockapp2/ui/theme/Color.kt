package com.example.weeklystockapp2.ui.theme

import androidx.compose.ui.graphics.Color

// Bas från din palett
val RedPrimary = Color(0xFFD9042B)   // #D9042B
val RedDeep = Color(0xFFBF0413)      // #BF0413
val Navy = Color(0xFF314A59)         // #314A59
val GreenAccent = Color(0xFF1B8C57)  // #1B8C57
val Sand = Color(0xFFD9A384)         // #D9A384

// Härledda neutrala färger
val BackgroundLight = Color(0xFFFDF9F6)
val SurfaceLight = Color(0xFFFFFFFF)
val SurfaceVariantLight = Sand.copy(alpha = 0.15f)

val BackgroundDark = Navy
val SurfaceDark = Color(0xFF243442)
val SurfaceVariantDark = Color(0xFF1A2732)

// Riskfärger (används i UI)
val RiskLowColor = GreenAccent
val RiskMediumColor = Sand
val RiskHighColor = RedPrimary
