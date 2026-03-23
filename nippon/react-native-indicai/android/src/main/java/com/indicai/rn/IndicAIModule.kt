package com.indicai.rn

import android.util.Base64
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule.RCTDeviceEventEmitter
import com.indicai.sdk.IndicAISDK
import com.indicai.sdk.DownloadProgress
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class IndicAIModule(private val reactContext: ReactApplicationContext)
    : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "IndicAIModule"

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val sdk = IndicAISDK.getInstance(reactContext)
    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    private var recordingBuffer: ByteArrayOutputStream? = null
    @Volatile private var isRecording = false

    @ReactMethod
    fun configure(manifestUrl: String?, s3BaseUrl: String?) {
        val config = IndicAISDK.SDKConfig(
            manifestUrl = manifestUrl ?: IndicAISDK.DEFAULT_MANIFEST_URL,
            s3BaseUrl = s3BaseUrl ?: IndicAISDK.DEFAULT_S3_BASE_URL,
        )
        IndicAISDK.configure(reactContext, config)
    }

    @ReactMethod
    fun initialize(languageCode: String, promise: Promise) {
        sdk.init(
            languageCode = languageCode,
            onProgress = { p: DownloadProgress -> emitProgress(p) },
            onReady = { promise.resolve(null) },
            onError = { e -> promise.reject("INDICAI_INIT_ERROR", e.message, e) },
        )
    }

    @ReactMethod
    fun transcribe(base64Audio: String, promise: Promise) {
        scope.launch {
            try {
                val pcmBytes = Base64.decode(base64Audio, Base64.NO_WRAP)
                val result = sdk.transcribe(pcmBytes)
                promise.resolve(result)
            } catch (e: Exception) {
                promise.reject("INDICAI_TRANSCRIBE_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun translateToEnglish(text: String, promise: Promise) {
        scope.launch {
            try {
                promise.resolve(sdk.translateToEnglish(text))
            } catch (e: Exception) {
                promise.reject("INDICAI_TRANSLATE_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun translateToIndic(text: String, promise: Promise) {
        scope.launch {
            try {
                promise.resolve(sdk.translateToIndic(text))
            } catch (e: Exception) {
                promise.reject("INDICAI_TRANSLATE_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun synthesize(text: String, promise: Promise) {
        sdk.synthesize(
            text = text,
            onProgress = { p -> emitProgress(p) },
            onResult = { pcmFloat ->
                val outFile = File(reactContext.cacheDir, "indicai_tts.wav")
                writeWav(pcmFloat, outFile)
                promise.resolve(outFile.absolutePath)
            },
            onError = { e -> promise.reject("INDICAI_TTS_ERROR", e.message) },
        )
    }

    @ReactMethod
    fun respondWithSpeech(englishText: String, promise: Promise) {
        sdk.respondWithSpeech(
            englishText = englishText,
            onProgress = { p -> emitProgress(p) },
            onResult = { indicText, audio ->
                val outFile = File(reactContext.cacheDir, "indicai_tts.wav")
                writeWav(audio, outFile)
                val result = Arguments.createMap().apply {
                    putString("indicText", indicText)
                    putString("audioPath", outFile.absolutePath)
                }
                promise.resolve(result)
            },
            onError = { e -> promise.reject("INDICAI_RESPOND_ERROR", e.message) },
        )
    }

    /** Write Float32 PCM samples as a 16-bit PCM WAV file. */
    private fun writeWav(samples: FloatArray, file: File) {
        val dataBytes = samples.size * 2
        val sampleRate = 16000
        val buf = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray(Charsets.US_ASCII))
        buf.putInt(36 + dataBytes)
        buf.put("WAVE".toByteArray(Charsets.US_ASCII))
        buf.put("fmt ".toByteArray(Charsets.US_ASCII))
        buf.putInt(16)
        buf.putShort(1)
        buf.putShort(1)
        buf.putInt(sampleRate)
        buf.putInt(sampleRate * 2)
        buf.putShort(2)
        buf.putShort(16)
        buf.put("data".toByteArray(Charsets.US_ASCII))
        buf.putInt(dataBytes)
        for (s in samples) {
            buf.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        }
        file.writeBytes(buf.array())
    }

    @ReactMethod
    fun classifyIntent(text: String, promise: Promise) {
        scope.launch {
            try {
                val result = sdk.classifyIntent(text)
                if (result == null) {
                    promise.reject("INDICAI_INTENT_ERROR", "Intent model not loaded")
                } else {
                    val map = Arguments.createMap().apply {
                        putString("intent", result.intent)
                        putDouble("confidence", result.confidence.toDouble())
                    }
                    promise.resolve(map)
                }
            } catch (e: Exception) {
                promise.reject("INDICAI_INTENT_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun getSupportedLanguages(promise: Promise) {
        val arr = Arguments.createArray()
        sdk.getSupportedLanguages().forEach { arr.pushString(it) }
        promise.resolve(arr)
    }

    @ReactMethod
    fun isTtsCached(languageCode: String, promise: Promise) {
        promise.resolve(sdk.isTtsCachedFor(languageCode))
    }

    @ReactMethod
    fun release() {
        stopAndReleaseRecorder()
        sdk.release()
    }

    @ReactMethod
    fun startRecording(promise: Promise) {
        try {
            if (isRecording) {
                promise.resolve(null)
                return
            }
            val sampleRate = 16000
            val minBuffer = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            if (minBuffer <= 0) {
                promise.reject("INDICAI_RECORD_ERROR", "Invalid recorder buffer size")
                return
            }
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBuffer * 2
            )
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                promise.reject("INDICAI_RECORD_ERROR", "AudioRecord init failed")
                return
            }

            audioRecord = recorder
            recordingBuffer = ByteArrayOutputStream()
            isRecording = true
            recorder.startRecording()

            recordingJob = scope.launch {
                val chunk = ByteArray(minBuffer)
                while (isRecording) {
                    val n = recorder.read(chunk, 0, chunk.size)
                    if (n > 0) {
                        recordingBuffer?.write(chunk, 0, n)
                    }
                }
            }
            promise.resolve(null)
        } catch (e: Exception) {
            stopAndReleaseRecorder()
            promise.reject("INDICAI_RECORD_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun stopRecording(promise: Promise) {
        scope.launch {
            try {
                if (!isRecording) {
                    promise.resolve("")
                    return@launch
                }
                isRecording = false
                recordingJob?.join()
                val recorder = audioRecord
                try { recorder?.stop() } catch (_: Exception) {}
                recorder?.release()
                audioRecord = null
                recordingJob = null

                val pcm = recordingBuffer?.toByteArray() ?: ByteArray(0)
                recordingBuffer = null
                val b64 = Base64.encodeToString(pcm, Base64.NO_WRAP)
                promise.resolve(b64)
            } catch (e: Exception) {
                stopAndReleaseRecorder()
                promise.reject("INDICAI_RECORD_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod fun addListener(eventName: String) {}
    @ReactMethod fun removeListeners(count: Int) {}

    private fun emitProgress(p: DownloadProgress) {
        val map = Arguments.createMap().apply {
            putString("model", p.modelName)
            putInt("percent", p.percent)
            putDouble("downloaded", p.downloadedBytes.toDouble())
            putDouble("total", p.totalBytes.toDouble())
        }
        reactContext
            .getJSModule(RCTDeviceEventEmitter::class.java)
            .emit("IndicAI_Progress", map)
    }

    private fun stopAndReleaseRecorder() {
        isRecording = false
        try { audioRecord?.stop() } catch (_: Exception) {}
        try { audioRecord?.release() } catch (_: Exception) {}
        audioRecord = null
        recordingJob?.cancel()
        recordingJob = null
        recordingBuffer = null
    }
}
