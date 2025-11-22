package com.example.weeklystockapp2

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.Image
import kotlinx.coroutines.delay

@Composable
fun SplashScreen(
    onFinished: () -> Unit
) {
    // Animationsprogress 0..1
    val baseAlpha = remember { Animatable(0f) }
    val iFillProgress = remember { Animatable(0f) }
    val taglineAlpha = remember { Animatable(0f) }
    val taglineTranslateX = remember { Animatable(-80f) } // px från vänster

    LaunchedEffect(Unit) {
        // 1) NLU fade-in
        baseAlpha.animateTo(
            targetValue = 1f,
            animationSpec = tween(
                durationMillis = 600,
                easing = FastOutSlowInEasing
            )
        )

        // 2) I-stapeln fylls nerifrån och upp
        iFillProgress.animateTo(
            targetValue = 1f,
            animationSpec = tween(
                durationMillis = 700,
                easing = LinearOutSlowInEasing
            )
        )

        // 3) Tagline: glid in från vänster + fade-in
        taglineAlpha.animateTo(
            targetValue = 1f,
            animationSpec = tween(
                durationMillis = 450,
                easing = FastOutSlowInEasing
            )
        )
        taglineTranslateX.animateTo(
            targetValue = 0f,
            animationSpec = tween(
                durationMillis = 450,
                easing = FastOutSlowInEasing
            )
        )

        // Liten paus innan vi går vidare
        delay(300)
        onFinished()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    listOf(
                        MaterialTheme.colorScheme.background,
                        MaterialTheme.colorScheme.surfaceVariant
                    )
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        // Behåll kvadratiska proportioner – skala efter behov
        Box(
            modifier = Modifier.size(300.dp), // justera om du vill ha större/mindre logga
            contentAlignment = Alignment.Center
        ) {
            // 1) Bas: svarta NLU
            Image(
                painter = painterResource(id = R.drawable.logo_nilu_base),
                contentDescription = "NiLU logo base",
                modifier = Modifier
                    .matchParentSize()
                    .graphicsLayer(alpha = baseAlpha.value)
            )

            // 2) I-stapeln som fylls nerifrån och upp
            Image(
                painter = painterResource(id = R.drawable.logo_nilu_i),
                contentDescription = "NiLU logo I fill",
                modifier = Modifier
                    .matchParentSize()
                    .drawWithContent {
                        val progress = iFillProgress.value.coerceIn(0f, 1f)
                        if (progress <= 0f) return@drawWithContent

                        val visibleHeight = size.height * progress
                        clipRect(
                            left = 0f,
                            top = size.height - visibleHeight,
                            right = size.width,
                            bottom = size.height
                        ) {
                            this@drawWithContent.drawContent()
                        }
                    }
            )

            // 3) Tagline som glider in från vänster
            Image(
                painter = painterResource(id = R.drawable.logo_nilu_tagline),
                contentDescription = "NiLU tagline",
                modifier = Modifier
                    .matchParentSize()
                    .graphicsLayer(
                        alpha = taglineAlpha.value,
                        translationX = taglineTranslateX.value
                    )
            )
        }
    }
}
