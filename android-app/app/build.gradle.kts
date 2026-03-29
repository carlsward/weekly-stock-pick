plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

fun backendUrl(baseUrl: String, fileName: String): String {
    return "${baseUrl.trimEnd('/')}/$fileName"
}

val releaseBackendBaseUrl = providers
    .gradleProperty("nilu.backendBaseUrl")
    .orElse("https://raw.githubusercontent.com/carlsward/weekly-stock-pick/main/backend")
    .get()

val debugBackendBaseUrl = providers
    .gradleProperty("nilu.debugBackendBaseUrl")
    .orElse(releaseBackendBaseUrl)
    .get()

android {
    namespace = "com.nilu.weeklypicks"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.nilu.weeklypicks"
        minSdk = 26
        targetSdk = 36
        versionCode = 2
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            buildConfigField(
                "String",
                "DASHBOARD_URL",
                "\"${backendUrl(debugBackendBaseUrl, "risk_picks.json")}\""
            )
            buildConfigField(
                "String",
                "HISTORY_URL",
                "\"${backendUrl(debugBackendBaseUrl, "history.json")}\""
            )
            buildConfigField(
                "String",
                "THESIS_MONITOR_URL",
                "\"${backendUrl(debugBackendBaseUrl, "thesis_monitor.json")}\""
            )
            buildConfigField(
                "String",
                "TRACK_RECORD_URL",
                "\"${backendUrl(debugBackendBaseUrl, "track_record.json")}\""
            )
            buildConfigField(
                "String",
                "MONTHLY_PICK_URL",
                "\"${backendUrl(debugBackendBaseUrl, "monthly_pick.json")}\""
            )
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            buildConfigField(
                "String",
                "DASHBOARD_URL",
                "\"${backendUrl(releaseBackendBaseUrl, "risk_picks.json")}\""
            )
            buildConfigField(
                "String",
                "HISTORY_URL",
                "\"${backendUrl(releaseBackendBaseUrl, "history.json")}\""
            )
            buildConfigField(
                "String",
                "THESIS_MONITOR_URL",
                "\"${backendUrl(releaseBackendBaseUrl, "thesis_monitor.json")}\""
            )
            buildConfigField(
                "String",
                "TRACK_RECORD_URL",
                "\"${backendUrl(releaseBackendBaseUrl, "track_record.json")}\""
            )
            buildConfigField(
                "String",
                "MONTHLY_PICK_URL",
                "\"${backendUrl(releaseBackendBaseUrl, "monthly_pick.json")}\""
            )
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.okhttp)
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.okhttp.mockwebserver)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}
