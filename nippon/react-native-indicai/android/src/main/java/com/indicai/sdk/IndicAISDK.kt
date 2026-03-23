package com.indicai.sdk

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import java.io.File
import java.util.concurrent.TimeUnit

private const val TAG = "IndicAISDK"

data class DownloadProgress(
    val modelName: String,
    val downloadedBytes: Long,
    val totalBytes: Long,
) {
    val percent: Int get() = if (totalBytes > 0) ((downloadedBytes * 100) / totalBytes).toInt() else 0
}

sealed class IndicAIError(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class UnsupportedLanguage(lang: String) : IndicAIError("Language '$lang' not supported")
    class ModelDownloadFailed(model: String, cause: Throwable) : IndicAIError(
        "Download failed: $model${cause.message?.let { " ($it)" } ?: ""}",
        cause
    )
    class ModelCorrupted(model: String) : IndicAIError("SHA-256 mismatch: $model")
    class NotInitialized : IndicAIError("Call init() first")
    class NetworkUnavailable : IndicAIError("No network and no cached manifest")
}

class IndicAISDK private constructor(
    private val context: Context,
    private var config: SDKConfig,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val http = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    private val manifestManager get() = ManifestManager(context, config.manifestUrl, http)
    private val modelManager get() = ModelManager(context, config.s3BaseUrl, http)
    private val engine = InferenceEngine(context)
    private val intentEngine = IntentEngine()

    private var currentLang: LanguageConfig? = null
    private var manifest: SDKManifest? = null
    private var isReady = false

    fun init(
        languageCode: String,
        onProgress: ((DownloadProgress) -> Unit)? = null,
        onReady: () -> Unit,
        onError: (IndicAIError) -> Unit,
    ) {
        if (!LanguageRegistry.isSupported(languageCode)) {
            onError(IndicAIError.UnsupportedLanguage(languageCode)); return
        }
        scope.launch {
            try {
                currentLang = LanguageRegistry.get(languageCode)
                manifest = manifestManager.fetch()

                engine.initTokenizers()
                engine.loadTtsVocab(languageCode)

                val whisperFile = modelManager.ensureModel(manifest!!.whisper) { dl, total ->
                    onProgress?.invoke(DownloadProgress("Whisper", dl, total))
                }
                engine.loadWhisper(whisperFile)

                val indicFile = modelManager.ensureModel(manifest!!.indicEn) { dl, total ->
                    onProgress?.invoke(DownloadProgress("IndicTrans2 Indic→EN", dl, total))
                }
                engine.loadIndicEn(indicFile)

                val enIndicFile = modelManager.ensureModel(manifest!!.indicFromEn) { dl, total ->
                    onProgress?.invoke(DownloadProgress("IndicTrans2 EN→Indic", dl, total))
                }
                engine.loadEnIndic(enIndicFile)

                // Prefetch selected language TTS at initialization time.
                val lang = currentLang!!
                manifest!!.languageTts[lang.code]?.let { ttsEntry ->
                    modelManager.ensureModel(ttsEntry) { dl, total ->
                        onProgress?.invoke(DownloadProgress("TTS-${lang.code}", dl, total))
                    }
                }

                // Load intent model if available in manifest.
                // ensureModel handles zip download + extraction; returns path to .onnx inside zip.
                manifest!!.intent?.let { intentEntry ->
                    val intentOnnx = modelManager.ensureModel(intentEntry) { dl, total ->
                        onProgress?.invoke(DownloadProgress("Intent", dl, total))
                    }
                    val intentDir = intentOnnx.parentFile!!
                    intentEngine.load(
                        modelFile      = intentOnnx,
                        vocabFile      = File(intentDir, "tokenizer/vocab.txt"),
                        bankNpyFile    = File(intentDir, "intent_bank.npy"),
                        bankLabelsFile = File(intentDir, "intent_bank_labels.json"),
                    )
                }

                isReady = true
                Log.d(TAG, "Ready for $languageCode")
                withContext(Dispatchers.Main) { onReady() }
            } catch (e: IndicAIError) {
                withContext(Dispatchers.Main) { onError(e) }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    onError(IndicAIError.ModelDownloadFailed("init", e))
                }
            }
        }
    }

    fun transcribe(pcmBytes: ByteArray): String {
        checkReady()
        val floats = FloatArray(pcmBytes.size / 2) { i ->
            val sample = (pcmBytes[i * 2].toInt() and 0xFF) or (pcmBytes[i * 2 + 1].toInt() shl 8)
            sample.toShort() / 32768f
        }
        return engine.transcribe(floats, currentLang!!)
    }

    fun translateToEnglish(text: String): String {
        checkReady()
        return engine.translateToEn(text, currentLang!!)
    }

    fun translateToIndic(englishText: String): String {
        checkReady()
        return engine.translateToIndic(englishText, currentLang!!)
    }

    fun synthesize(
        text: String,
        onProgress: ((DownloadProgress) -> Unit)? = null,
        onResult: (FloatArray) -> Unit,
        onError: (IndicAIError) -> Unit,
    ) {
        checkReady()
        val lang = currentLang!!
        scope.launch {
            try {
                val ttsEntry = manifest!!.languageTts[lang.code]
                    ?: run { onError(IndicAIError.UnsupportedLanguage("${lang.code} TTS")); return@launch }
                val ttsFile = modelManager.ensureModel(ttsEntry) { dl, total ->
                    onProgress?.invoke(DownloadProgress("TTS-${lang.code}", dl, total))
                }
                engine.loadTts(ttsFile)
                engine.loadTtsVocab(lang.code)
                onResult(engine.synthesize(text, lang.code))
            } catch (e: SecurityException) {
                onError(IndicAIError.ModelCorrupted("TTS"))
            } catch (e: Exception) {
                onError(IndicAIError.ModelDownloadFailed("TTS", e))
            }
        }
    }

    fun respondWithSpeech(
        englishText: String,
        onProgress: ((DownloadProgress) -> Unit)? = null,
        onResult: (indicText: String, audio: FloatArray) -> Unit,
        onError: (IndicAIError) -> Unit,
    ) {
        checkReady()
        val lang = currentLang!!
        scope.launch {
            try {
                val indicText = engine.translateToIndic(englishText, lang)

                val ttsEntry = manifest!!.languageTts[lang.code]
                    ?: run { onError(IndicAIError.UnsupportedLanguage("${lang.code} TTS")); return@launch }
                val ttsFile = modelManager.ensureModel(ttsEntry) { dl, total ->
                    onProgress?.invoke(DownloadProgress("TTS-${lang.code}", dl, total))
                }
                engine.loadTts(ttsFile)
                engine.loadTtsVocab(lang.code)
                val audio = engine.synthesize(indicText, lang.code)
                onResult(indicText, audio)
            } catch (e: Exception) {
                onError(IndicAIError.ModelDownloadFailed("respondWithSpeech", e))
            }
        }
    }

    /**
     * Classify English text to an intent label.
     * Returns IntentResult(intent, confidence) or null if the intent model is not loaded.
     */
    fun classifyIntent(englishText: String): IntentResult? {
        if (!intentEngine.isLoaded()) return null
        return intentEngine.classify(englishText)
    }

    fun getSupportedLanguages(): List<String> = LanguageRegistry.SUPPORTED_LANGUAGES.keys.toList()
    fun isLanguageSupported(code: String) = LanguageRegistry.isSupported(code)
    fun isTtsCachedFor(code: String): Boolean {
        val entry = manifest?.languageTts?.get(code) ?: return false
        return modelManager.isModelCached(entry)
    }

    fun release() { engine.close(); intentEngine.close(); isReady = false }
    private fun checkReady() { if (!isReady) throw IndicAIError.NotInitialized() }

    data class SDKConfig(val manifestUrl: String, val s3BaseUrl: String)

    companion object {
        const val DEFAULT_MANIFEST_URL = "https://indicai-cdn.s3.ap-south-1.amazonaws.com/manifest.json"
        const val DEFAULT_S3_BASE_URL = "https://indicai-cdn.s3.ap-south-1.amazonaws.com/models"

        @Volatile private var instance: IndicAISDK? = null

        fun getInstance(context: Context) =
            instance ?: synchronized(this) {
                instance ?: IndicAISDK(
                    context.applicationContext,
                    SDKConfig(DEFAULT_MANIFEST_URL, DEFAULT_S3_BASE_URL)
                ).also { instance = it }
            }

        fun configure(context: Context, config: SDKConfig) {
            synchronized(this) {
                instance = IndicAISDK(context.applicationContext, config)
            }
        }
    }
}
