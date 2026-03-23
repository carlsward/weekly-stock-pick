package com.nilu.weeklypicks

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat

internal data class ReleaseNotificationEvent(
    val id: Int,
    val title: String,
    val message: String
)

internal data class ReleaseNotificationPlan(
    val events: List<ReleaseNotificationEvent>,
    val releaseWeekToRecord: String? = null,
    val thesisAlertKeyToRecord: String? = null,
    val staleWeekToRecord: String? = null
)

private const val RELEASE_NOTIFICATION_CHANNEL_ID = "release_updates"
private const val NEW_RELEASE_NOTIFICATION_ID = 4101
private const val THESIS_ALERT_NOTIFICATION_ID = 4102
private const val STALE_DATA_NOTIFICATION_ID = 4103

internal fun buildReleaseNotificationPlan(
    content: DashboardContent,
    lastReleaseWeek: String?,
    lastThesisAlertKey: String?,
    lastStaleAlertWeek: String?
): ReleaseNotificationPlan {
    if (content.source != DataSource.NETWORK) {
        return ReleaseNotificationPlan(events = emptyList())
    }

    val events = mutableListOf<ReleaseNotificationEvent>()
    val weekId = content.dashboard.marketContext.weekId
    val overallSelection = content.dashboard.overallSelection
    val overallPick = overallSelection.pick
    var releaseWeekToRecord: String? = null
    var thesisAlertKeyToRecord: String? = null
    var staleWeekToRecord: String? = null

    val isNewWeek = weekId != lastReleaseWeek
    if (isNewWeek) {
        releaseWeekToRecord = weekId
        val releaseTitle: String
        val releaseMessage: String
        if (overallSelection.status == SelectionStatus.PICKED && overallPick != null) {
            releaseTitle = "New weekly release: ${overallPick.symbol}"
            val monitorText = overallPick.thesisMonitor?.headline?.lowercase()
            releaseMessage = buildString {
                append("${overallPick.companyName} leads ${content.dashboard.marketContext.weekLabel}. ")
                append("Confidence is ${overallPick.confidenceLabel}.")
                if (!monitorText.isNullOrBlank()) {
                    append(" $monitorText.")
                }
            }
        } else {
            releaseTitle = "Weekly release on hold"
            releaseMessage = "No stock cleared the release bar for ${content.dashboard.marketContext.weekLabel}."
        }
        events += ReleaseNotificationEvent(
            id = NEW_RELEASE_NOTIFICATION_ID,
            title = releaseTitle,
            message = releaseMessage
        )
    }

    if (!isNewWeek && overallSelection.status == SelectionStatus.PICKED && overallPick != null) {
        val monitor = overallPick.thesisMonitor
        if (monitor != null && monitor.status != ThesisMonitorStatus.HEALTHY) {
            val alertKey = "$weekId:${overallPick.symbol}:${monitor.status.name}"
            if (alertKey != lastThesisAlertKey) {
                thesisAlertKeyToRecord = alertKey
                events += ReleaseNotificationEvent(
                    id = THESIS_ALERT_NOTIFICATION_ID,
                    title = when (monitor.status) {
                        ThesisMonitorStatus.RISK -> "Thesis risk: ${overallPick.symbol}"
                        ThesisMonitorStatus.WATCH -> "Thesis watch: ${overallPick.symbol}"
                        ThesisMonitorStatus.HEALTHY -> "Thesis update: ${overallPick.symbol}"
                    },
                    message = monitor.summary
                )
            }
        }
    }

    if (content.freshness == Freshness.STALE && weekId != lastStaleAlertWeek) {
        staleWeekToRecord = weekId
        events += ReleaseNotificationEvent(
            id = STALE_DATA_NOTIFICATION_ID,
            title = "Dashboard data is stale",
            message = "The saved release data is older than the expected refresh window for ${content.dashboard.marketContext.weekLabel}."
        )
    }

    return ReleaseNotificationPlan(
        events = events,
        releaseWeekToRecord = releaseWeekToRecord,
        thesisAlertKeyToRecord = thesisAlertKeyToRecord,
        staleWeekToRecord = staleWeekToRecord
    )
}

class ReleaseNotifier(
    context: Context,
    private val localStore: LocalStore
) {
    private val appContext = context.applicationContext

    fun onContentUpdated(content: DashboardContent) {
        val plan = buildReleaseNotificationPlan(
            content = content,
            lastReleaseWeek = localStore.readLastReleaseNotificationWeek(),
            lastThesisAlertKey = localStore.readLastThesisAlertKey(),
            lastStaleAlertWeek = localStore.readLastStaleAlertWeek()
        )

        if (plan.events.isEmpty()) {
            return
        }

        if (!canPostNotifications()) {
            return
        }

        ensureChannel()
        val manager = NotificationManagerCompat.from(appContext)
        plan.events.forEach { event ->
            manager.notify(
                event.id,
                NotificationCompat.Builder(appContext, CHANNEL_ID)
                    .setSmallIcon(android.R.drawable.ic_dialog_info)
                    .setContentTitle(event.title)
                    .setContentText(event.message)
                    .setStyle(NotificationCompat.BigTextStyle().bigText(event.message))
                    .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                    .setAutoCancel(true)
                    .build()
            )
        }

        plan.releaseWeekToRecord?.let(localStore::writeLastReleaseNotificationWeek)
        plan.thesisAlertKeyToRecord?.let(localStore::writeLastThesisAlertKey)
        plan.staleWeekToRecord?.let(localStore::writeLastStaleAlertWeek)
    }

    private fun canPostNotifications(): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val granted = ContextCompat.checkSelfPermission(
                appContext,
                Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
            if (!granted) {
                return false
            }
        }
        return NotificationManagerCompat.from(appContext).areNotificationsEnabled()
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }

        val manager = appContext.getSystemService(NotificationManager::class.java)
        val existing = manager.getNotificationChannel(CHANNEL_ID)
        if (existing != null) {
            return
        }

        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Release updates",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Weekly release reminders and thesis monitor alerts."
            }
        )
    }

    private companion object {
        const val CHANNEL_ID = RELEASE_NOTIFICATION_CHANNEL_ID
    }
}
