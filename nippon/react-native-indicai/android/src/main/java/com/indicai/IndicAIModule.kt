package com.indicai

import android.util.Base64
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.indicai.engine.ModelManager
import com.indicai.engine.WhisperEngine
import com.indicai.engine.TranslationEngine
import com.indicai.engine.TtsEngine
import kotlinx.coroutines.*
import java.io.File

class IndicAIModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "IndicAIModule"

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val modelsDir: File get() = File(reactContext.filesDir, "models")

    private var customManifestUrl: String? = null
    private var customS3BaseUrl: String? = null

    private var currentLanguageCode: String = "hi"

    private var modelManager: ModelManager? = null
    private var whisperEngine: WhisperEngine? = null
    private var translationEngineIndicEn: TranslationEngine? = null
    private var translationEngineEnIndic: TranslationEngine? = null
    private var ttsEngine: TtsEngine? = null

    private fun getModelManager(): ModelManager {
        if (modelManager == null) {
            modelManager = ModelManager(
                context = reactContext,
                modelsDir = modelsDir,
                manifestUrl = customManifestUrl ?: ModelManager.DEFAULT_MANIFEST_URL,
                s3BaseUrl = customS3BaseUrl ?: ModelManager.DEFAULT_S3_BASE_URL
            )
        }
        return modelManager!!
    }

    private fun sendProgressEvent(model: String, percent: Int, downloaded: Long, total: Long) {
        val params = Arguments.createMap().apply {
            putString("model", model)
            putInt("percent", percent)
            putDouble("downloaded", downloaded.toDouble())
            putDouble("total", total.toDouble())
        }
        reactContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit("IndicAI_Progress", params)
    }

    @ReactMethod
    fun configure(manifestUrl: String?, s3BaseUrl: String?) {
        customManifestUrl = manifestUrl
        customS3BaseUrl = s3BaseUrl
        modelManager = null // reset so it gets recreated with new URLs
    }

    @ReactMethod
    fun initialize(languageCode: String, promise: Promise) {
        currentLanguageCode = languageCode
        scope.launch {
            try {
                val mgr = getModelManager()

                // Download core models: whisper + both indictrans2 directions
                val coreModels = listOf("whisper", "indic_en", "indic_from_en")
                for (modelKey in coreModels) {
                    mgr.ensureModel(modelKey) { name, percent, downloaded, total ->
                        sendProgressEvent(name, percent, downloaded, total)
                    }
                }

                // Download TTS for the requested language
                mgr.ensureTtsModel(languageCode) { name, percent, downloaded, total ->
                    sendProgressEvent(name, percent, downloaded, total)
                }

                // Initialize engines
                whisperEngine = WhisperEngine(
                    modelsDir = modelsDir,
                    assets = reactContext.assets,
                    languageCode = languageCode
                )
                translationEngineIndicEn = TranslationEngine(
                    modelFile = File(modelsDir, "indic-trans2-indic-en/model_quantized.onnx"),
                    direction = TranslationEngine.Direction.INDIC_TO_EN,
                    assets = reactContext.assets
                )
                translationEngineEnIndic = TranslationEngine(
                    modelFile = File(modelsDir, "indic-trans2-en-indic/model_quantized.onnx"),
                    direction = TranslationEngine.Direction.EN_TO_INDIC,
                    assets = reactContext.assets
                )
                ttsEngine = TtsEngine(
                    modelsDir = modelsDir,
                    assets = reactContext.assets,
                    languageCode = languageCode
                )

                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("INIT_ERROR", e.message ?: "Initialization failed", e)
            }
        }
    }

    @ReactMethod
    fun transcribe(base64Audio: String, promise: Promise) {
        scope.launch {
            try {
                val engine = whisperEngine
                    ?: return@launch promise.reject("NOT_INITIALIZED", "Call initialize() first")
                val result = engine.transcribe(base64Audio)
                promise.resolve(result)
            } catch (e: Exception) {
                promise.reject("TRANSCRIBE_ERROR", e.message ?: "Transcription failed", e)
            }
        }
    }

    @ReactMethod
    fun translateToEnglish(text: String, promise: Promise) {
        scope.launch {
            try {
                val engine = translationEngineIndicEn
                    ?: return@launch promise.reject("NOT_INITIALIZED", "Call initialize() first")
                val result = engine.translate(text, currentLanguageCode, "en")
                promise.resolve(result)
            } catch (e: Exception) {
                promise.reject("TRANSLATE_ERROR", e.message ?: "Translation failed", e)
            }
        }
    }

    @ReactMethod
    fun translateToIndic(text: String, promise: Promise) {
        scope.launch {
            try {
                val engine = translationEngineEnIndic
                    ?: return@launch promise.reject("NOT_INITIALIZED", "Call initialize() first")
                val result = engine.translate(text, "en", currentLanguageCode)
                promise.resolve(result)
            } catch (e: Exception) {
                promise.reject("TRANSLATE_ERROR", e.message ?: "Translation failed", e)
            }
        }
    }

    @ReactMethod
    fun synthesize(text: String, promise: Promise) {
        scope.launch {
            try {
                val engine = ttsEngine
                    ?: return@launch promise.reject("NOT_INITIALIZED", "Call initialize() first")
                val outFile = File(reactContext.cacheDir, "indicai_tts.wav")
                val audioPath = engine.synthesize(text, outFile)
                promise.resolve(audioPath)
            } catch (e: Exception) {
                promise.reject("TTS_ERROR", e.message ?: "Synthesis failed", e)
            }
        }
    }

    @ReactMethod
    fun respondWithSpeech(englishText: String, promise: Promise) {
        scope.launch {
            try {
                val mtEngine = translationEngineEnIndic
                    ?: return@launch promise.reject("NOT_INITIALIZED", "Call initialize() first")
                val tts = ttsEngine
                    ?: return@launch promise.reject("NOT_INITIALIZED", "Call initialize() first")

                val indicText = mtEngine.translate(englishText, "en", currentLanguageCode)
                val outFile = File(reactContext.cacheDir, "indicai_tts.wav")
                val audioPath = tts.synthesize(indicText, outFile)

                val map = Arguments.createMap().apply {
                    putString("indicText", indicText)
                    putString("audioPath", audioPath)
                }
                promise.resolve(map)
            } catch (e: Exception) {
                promise.reject("RESPOND_ERROR", e.message ?: "respondWithSpeech failed", e)
            }
        }
    }

    @ReactMethod
    fun getSupportedLanguages(promise: Promise) {
        scope.launch {
            try {
                val mgr = getModelManager()
                val languages = mgr.getSupportedLanguages()
                val arr = Arguments.createArray()
                languages.forEach { arr.pushString(it) }
                promise.resolve(arr)
            } catch (e: Exception) {
                promise.reject("LANG_ERROR", e.message ?: "getSupportedLanguages failed", e)
            }
        }
    }

    @ReactMethod
    fun isTtsCached(languageCode: String, promise: Promise) {
        scope.launch {
            try {
                val ttsDir = File(modelsDir, "mms-tts-$languageCode")
                val modelFile = File(ttsDir, "model.onnx")
                promise.resolve(modelFile.exists())
            } catch (e: Exception) {
                promise.reject("CACHE_ERROR", e.message ?: "isTtsCached failed", e)
            }
        }
    }

    @ReactMethod
    fun release() {
        try {
            whisperEngine?.close()
            translationEngineIndicEn?.close()
            translationEngineEnIndic?.close()
            ttsEngine?.close()
            whisperEngine = null
            translationEngineIndicEn = null
            translationEngineEnIndic = null
            ttsEngine = null
        } catch (e: Exception) {
            // ignore cleanup errors
        }
    }

    override fun invalidate() {
        super.invalidate()
        scope.cancel()
        release()
    }

    override fun getConstants(): Map<String, Any> = mapOf("IndicAIModule" to "IndicAIModule")
}
