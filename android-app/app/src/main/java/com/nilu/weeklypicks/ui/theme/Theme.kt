package com.nilu.weeklypicks.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColorScheme = lightColorScheme(
    primary = Copper,
    onPrimary = Graphite950,
    primaryContainer = CopperBright,
    onPrimaryContainer = Graphite950,

    secondary = Graphite700,
    onSecondary = Ivory,

    tertiary = BullGreen,
    onTertiary = Graphite950,

    background = BackgroundLight,
    onBackground = Graphite950,

    surface = SurfaceLight,
    onSurface = Graphite950,

    surfaceVariant = SurfaceVariantLight,
    onSurfaceVariant = Graphite700,

    outline = Graphite700.copy(alpha = 0.28f),
    error = BearRed
)

private val DarkColorScheme = darkColorScheme(
    primary = Copper,
    onPrimary = Graphite950,
    primaryContainer = CopperDeep,
    onPrimaryContainer = Ivory,

    secondary = Graphite700,
    onSecondary = Ivory,

    tertiary = BullGreen,
    onTertiary = Graphite950,

    background = BackgroundDark,
    onBackground = Ivory,

    surface = SurfaceDark,
    onSurface = Ivory,

    surfaceVariant = SurfaceVariantDark,
    onSurfaceVariant = Mist,

    outline = Slate,
    error = BearRed
)

@Composable
fun HyraxAlphaTheme(
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

@Composable
fun NiLUWeeklyPicksTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    HyraxAlphaTheme(
        darkTheme = darkTheme,
        content = content
    )
}
