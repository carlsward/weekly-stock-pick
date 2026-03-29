package com.nilu.weeklypicks

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.nilu.weeklypicks.ui.theme.BullGreen
import com.nilu.weeklypicks.ui.theme.Copper
import com.nilu.weeklypicks.ui.theme.CopperBright
import com.nilu.weeklypicks.ui.theme.Graphite900

@Composable
fun HyraxBrandMark(
    modifier: Modifier = Modifier,
    size: Dp = 56.dp,
    animated: Boolean = true
) {
    val transition = rememberInfiniteTransition(label = "brandMark")
    val pulse by transition.animateFloat(
        initialValue = 0.92f,
        targetValue = 1.08f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 2400),
            repeatMode = RepeatMode.Reverse
        ),
        label = "brandPulse"
    )

    Box(
        modifier = modifier.size(size),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .size(size * if (animated) pulse else 1f)
                .graphicsLayer(alpha = if (animated) 0.18f else 0.14f)
                .clip(CircleShape)
                .background(
                    Brush.radialGradient(
                        colors = listOf(CopperBright, Color.Transparent)
                    )
                )
        )

        Box(
            modifier = Modifier
                .size(size * 0.82f)
                .clip(RoundedCornerShape(size * 0.26f))
                .background(
                    brush = Brush.linearGradient(
                        colors = listOf(CopperBright, Copper, BullGreen)
                    )
                ),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "HA",
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = Graphite900
            )
        }
    }
}

@Composable
fun HyraxWordmark(
    modifier: Modifier = Modifier,
    stacked: Boolean = false
) {
    if (stacked) {
        Column(
            modifier = modifier,
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            Text(
                text = "HYRAX",
                style = MaterialTheme.typography.labelLarge,
                color = CopperBright,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "ALPHA",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onBackground,
                fontWeight = FontWeight.Bold
            )
        }
    } else {
        Text(
            modifier = modifier,
            text = "HYRAX ALPHA",
            style = MaterialTheme.typography.labelLarge,
            color = CopperBright,
            fontWeight = FontWeight.Bold
        )
    }
}
