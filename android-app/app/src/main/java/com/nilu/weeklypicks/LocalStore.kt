package com.nilu.weeklypicks

import android.content.Context
import androidx.core.content.edit

interface LocalStore {
    fun readSelectedRisk(): String?
    fun writeSelectedRisk(risk: String)
    fun readDashboardCache(): String?
    fun writeDashboardCache(rawJson: String)
    fun readHistoryCache(): String?
    fun writeHistoryCache(rawJson: String)
    fun readThesisMonitorCache(): String?
    fun writeThesisMonitorCache(rawJson: String)
    fun readTrackRecordCache(): String?
    fun writeTrackRecordCache(rawJson: String)
    fun readMonthlyPickCache(): String?
    fun writeMonthlyPickCache(rawJson: String)
    fun readLastReleaseNotificationWeek(): String?
    fun writeLastReleaseNotificationWeek(weekId: String)
    fun readLastThesisAlertKey(): String?
    fun writeLastThesisAlertKey(key: String)
    fun readLastStaleAlertWeek(): String?
    fun writeLastStaleAlertWeek(weekId: String)
}

class SharedPreferencesLocalStore(
    context: Context
) : LocalStore {
    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    override fun readSelectedRisk(): String? = prefs.getString(KEY_SELECTED_RISK, null)

    override fun writeSelectedRisk(risk: String) {
        prefs.edit { putString(KEY_SELECTED_RISK, risk) }
    }

    override fun readDashboardCache(): String? = prefs.getString(KEY_DASHBOARD_CACHE, null)

    override fun writeDashboardCache(rawJson: String) {
        prefs.edit { putString(KEY_DASHBOARD_CACHE, rawJson) }
    }

    override fun readHistoryCache(): String? = prefs.getString(KEY_HISTORY_CACHE, null)

    override fun writeHistoryCache(rawJson: String) {
        prefs.edit { putString(KEY_HISTORY_CACHE, rawJson) }
    }

    override fun readThesisMonitorCache(): String? = prefs.getString(KEY_THESIS_MONITOR_CACHE, null)

    override fun writeThesisMonitorCache(rawJson: String) {
        prefs.edit { putString(KEY_THESIS_MONITOR_CACHE, rawJson) }
    }

    override fun readTrackRecordCache(): String? = prefs.getString(KEY_TRACK_RECORD_CACHE, null)

    override fun writeTrackRecordCache(rawJson: String) {
        prefs.edit { putString(KEY_TRACK_RECORD_CACHE, rawJson) }
    }

    override fun readMonthlyPickCache(): String? = prefs.getString(KEY_MONTHLY_PICK_CACHE, null)

    override fun writeMonthlyPickCache(rawJson: String) {
        prefs.edit { putString(KEY_MONTHLY_PICK_CACHE, rawJson) }
    }

    override fun readLastReleaseNotificationWeek(): String? =
        prefs.getString(KEY_LAST_RELEASE_NOTIFICATION_WEEK, null)

    override fun writeLastReleaseNotificationWeek(weekId: String) {
        prefs.edit { putString(KEY_LAST_RELEASE_NOTIFICATION_WEEK, weekId) }
    }

    override fun readLastThesisAlertKey(): String? =
        prefs.getString(KEY_LAST_THESIS_ALERT_KEY, null)

    override fun writeLastThesisAlertKey(key: String) {
        prefs.edit { putString(KEY_LAST_THESIS_ALERT_KEY, key) }
    }

    override fun readLastStaleAlertWeek(): String? =
        prefs.getString(KEY_LAST_STALE_ALERT_WEEK, null)

    override fun writeLastStaleAlertWeek(weekId: String) {
        prefs.edit { putString(KEY_LAST_STALE_ALERT_WEEK, weekId) }
    }

    private companion object {
        const val PREFS_NAME = "weekly_stock_cache"
        const val KEY_SELECTED_RISK = "selected_risk"
        const val KEY_DASHBOARD_CACHE = "dashboard_cache"
        const val KEY_HISTORY_CACHE = "history_cache"
        const val KEY_THESIS_MONITOR_CACHE = "thesis_monitor_cache"
        const val KEY_TRACK_RECORD_CACHE = "track_record_cache"
        const val KEY_MONTHLY_PICK_CACHE = "monthly_pick_cache"
        const val KEY_LAST_RELEASE_NOTIFICATION_WEEK = "last_release_notification_week"
        const val KEY_LAST_THESIS_ALERT_KEY = "last_thesis_alert_key"
        const val KEY_LAST_STALE_ALERT_WEEK = "last_stale_alert_week"
    }
}

class InMemoryLocalStore : LocalStore {
    private var selectedRisk: String? = null
    private var dashboardCache: String? = null
    private var historyCache: String? = null
    private var thesisMonitorCache: String? = null
    private var trackRecordCache: String? = null
    private var monthlyPickCache: String? = null
    private var lastReleaseNotificationWeek: String? = null
    private var lastThesisAlertKey: String? = null
    private var lastStaleAlertWeek: String? = null

    override fun readSelectedRisk(): String? = selectedRisk

    override fun writeSelectedRisk(risk: String) {
        selectedRisk = risk
    }

    override fun readDashboardCache(): String? = dashboardCache

    override fun writeDashboardCache(rawJson: String) {
        dashboardCache = rawJson
    }

    override fun readHistoryCache(): String? = historyCache

    override fun writeHistoryCache(rawJson: String) {
        historyCache = rawJson
    }

    override fun readThesisMonitorCache(): String? = thesisMonitorCache

    override fun writeThesisMonitorCache(rawJson: String) {
        thesisMonitorCache = rawJson
    }

    override fun readTrackRecordCache(): String? = trackRecordCache

    override fun writeTrackRecordCache(rawJson: String) {
        trackRecordCache = rawJson
    }

    override fun readMonthlyPickCache(): String? = monthlyPickCache

    override fun writeMonthlyPickCache(rawJson: String) {
        monthlyPickCache = rawJson
    }

    override fun readLastReleaseNotificationWeek(): String? = lastReleaseNotificationWeek

    override fun writeLastReleaseNotificationWeek(weekId: String) {
        lastReleaseNotificationWeek = weekId
    }

    override fun readLastThesisAlertKey(): String? = lastThesisAlertKey

    override fun writeLastThesisAlertKey(key: String) {
        lastThesisAlertKey = key
    }

    override fun readLastStaleAlertWeek(): String? = lastStaleAlertWeek

    override fun writeLastStaleAlertWeek(weekId: String) {
        lastStaleAlertWeek = weekId
    }
}
