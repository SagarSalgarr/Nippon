package com.indicai.engine

import ai.onnxruntime.*
import android.content.res.AssetManager
import android.util.Log
import com.indicai.tokenizer.VitsTokenizer
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * MMS-TTS (VITS) text-to-speech engine backed by ONNX Runtime.
 *
 * Input:  text string in the target Indic language
 * Output: PCM audio written to a WAV file; returns the absolute file path
 */
class TtsEngine(
    private val modelsDir: File,
    private val assets: AssetManager,
    private val languageCode: String
) : Closeable {

    companion object {
        private const val TAG = "TtsEngine"
        private const val NORMALIZE_SCALE = 0.9f
        private const val SAMPLE_RATE = 16000
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val tokenizer = VitsTokenizer(assets, languageCode)

    private val session: OrtSession by lazy {
        val modelPath = File(modelsDir, "mms-tts-$languageCode/model.onnx")
        if (!modelPath.exists()) throw IllegalStateException("TTS model not found: $modelPath")
        Log.d(TAG, "Loading TTS model from $modelPath")
        env.createSession(modelPath.absolutePath, OrtSession.SessionOptions())
    }

    /**
     * Synthesize speech from text and write to a WAV file.
     * @param text        Input text in the target Indic language
     * @param outputFile  Destination WAV file (will be overwritten)
     * @return            Absolute path to the written WAV file
     */
    fun synthesize(text: String, outputFile: File): String {
        val inputIds = tokenizer.encode(text)
        val seqLen = inputIds.size.toLong()
        Log.d(TAG, "TTS input tokens: $seqLen for text: $text")

        val inputIdsTensor = createLongTensor(inputIds, longArrayOf(1L, seqLen))
        val attentionMask = LongArray(inputIds.size) { 1L }
        val attentionMaskTensor = createLongTensor(attentionMask, longArrayOf(1L, seqLen))

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attentionMaskTensor
        )

        val outputs = session.run(inputs)
        val waveformTensor = outputs["waveform"].get() as OnnxTensor
        val waveform = extractWaveform(waveformTensor.value)

        inputIdsTensor.close()
        attentionMaskTensor.close()
        waveformTensor.close()
        outputs.close()

        val normalized = normalize(waveform)
        writeWav(normalized, outputFile)
        Log.d(TAG, "TTS wrote ${normalized.size} samples to ${outputFile.absolutePath}")
        return outputFile.absolutePath
    }

    private fun normalize(samples: FloatArray): FloatArray {
        if (samples.isEmpty()) return samples
        var maxAbs = 0.0f
        for (s in samples) {
            val a = abs(s)
            if (a > maxAbs) maxAbs = a
        }
        if (maxAbs < 1e-8f) return samples
        val scale = NORMALIZE_SCALE / maxAbs
        return FloatArray(samples.size) { i -> samples[i] * scale }
    }

    /**
     * Write Float32 samples as a standard 16-bit PCM WAV file.
     */
    private fun writeWav(samples: FloatArray, file: File) {
        val numSamples = samples.size
        val dataBytes = numSamples * 2  // 16-bit = 2 bytes per sample

        val buf = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header
        buf.put("RIFF".toByteArray(Charsets.US_ASCII))
        buf.putInt(36 + dataBytes)
        buf.put("WAVE".toByteArray(Charsets.US_ASCII))

        // fmt chunk
        buf.put("fmt ".toByteArray(Charsets.US_ASCII))
        buf.putInt(16)                        // subchunk size
        buf.putShort(1)                       // PCM = 1
        buf.putShort(1)                       // mono
        buf.putInt(SAMPLE_RATE)
        buf.putInt(SAMPLE_RATE * 2)           // byte rate
        buf.putShort(2)                       // block align
        buf.putShort(16)                      // bits per sample

        // data chunk
        buf.put("data".toByteArray(Charsets.US_ASCII))
        buf.putInt(dataBytes)
        for (s in samples) {
            val pcm = (s.coerceIn(-1f, 1f) * 32767f).toInt().toShort()
            buf.putShort(pcm)
        }

        file.parentFile?.mkdirs()
        file.writeBytes(buf.array())
    }

    @Suppress("UNCHECKED_CAST")
    private fun extractWaveform(value: Any?): FloatArray {
        return when (value) {
            is FloatArray -> value
            is Array<*> -> {
                val first = value[0]
                when (first) {
                    is FloatArray -> first
                    is Array<*> -> (first as Array<FloatArray>)[0]
                    else -> throw IllegalArgumentException("Unexpected waveform inner type: ${first?.javaClass}")
                }
            }
            else -> throw IllegalArgumentException("Unexpected waveform type: ${value?.javaClass}")
        }
    }

    private fun createLongTensor(data: LongArray, shape: LongArray): OnnxTensor {
        val buf = ByteBuffer.allocateDirect(data.size * 8).order(ByteOrder.nativeOrder())
        buf.asLongBuffer().put(data)
        buf.rewind()
        return OnnxTensor.createTensor(env, buf.asLongBuffer(), shape)
    }

    override fun close() {
        try { session.close() } catch (_: Exception) {}
    }
}
