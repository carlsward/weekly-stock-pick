package com.nilu.weeklypicks

import java.util.Locale

fun percentLabel(value: Double?): String {
    return if (value == null) {
        "N/A"
    } else {
        String.format(Locale.US, "%.0f%%", value * 100)
    }
}

fun signedPercentLabel(value: Double?): String {
    return if (value == null) {
        "N/A"
    } else {
        String.format(Locale.US, "%+.1f%%", value * 100)
    }
}
