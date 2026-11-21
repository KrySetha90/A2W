plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "com.enhancer.app"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.enhancer.app"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }
    androidResources { noCompress += "onnx" }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
dependencies { implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2") }
