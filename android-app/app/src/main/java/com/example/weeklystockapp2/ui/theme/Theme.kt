package com.example.weeklystockapp2.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColorScheme = lightColorScheme(
    primary = RedPrimary,
    onPrimary = Color.White,
    primaryContainer = RedPrimary.copy(alpha = 0.9f),
    onPrimaryContainer = Color.White,

    secondary = GreenAccent,
    onSecondary = Color.White,

    tertiary = Sand,
    onTertiary = Navy,

    background = BackgroundLight,
    onBackground = Navy,

    surface = SurfaceLight,
    onSurface = Navy,

    surfaceVariant = SurfaceVariantLight,
    onSurfaceVariant = Navy,

    outline = Navy.copy(alpha = 0.25f)
)

private val DarkColorScheme = darkColorScheme(
    primary = RedPrimary,
    onPrimary = Color.White,
    primaryContainer = RedDeep,
    onPrimaryContainer = Color.White,

    secondary = GreenAccent,
    onSecondary = Color.White,

    tertiary = Sand,
    onTertiary = BackgroundDark,

    background = BackgroundDark,
    onBackground = Color(0xFFEFEFEF),

    surface = SurfaceDark,
    onSurface = Color(0xFFEFEFEF),

    surfaceVariant = SurfaceVariantDark,
    onSurfaceVariant = Color(0xFFDFDFDF),

    outline = Color(0xFF8FA0AF)
)

@Composable
fun WeeklyStockApp2Theme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        shapes = Shapes,
        content = content
    )
}
