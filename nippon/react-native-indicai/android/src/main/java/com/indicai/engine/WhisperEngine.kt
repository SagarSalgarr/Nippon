package com.indicai.engine

import ai.onnxruntime.*
import android.content.res.AssetManager
import android.util.Base64
import android.util.Log
import com.indicai.tokenizer.WhisperTokenizer
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Whisper STT engine backed by ONNX Runtime.
 *
 * Input:  base64-encoded Int16 LE PCM at 16kHz mono
 * Output: transcribed text in the target Indic language
 */
class WhisperEngine(
    private val modelsDir: File,
    private val assets: AssetManager,
    private val languageCode: String
) : Closeable {

    companion object {
        private const val TAG = "WhisperEngine"
        private const val MAX_DECODE_LEN = 224
        private const val ENCODER_INPUT = "input_features"
        private const val ENCODER_OUTPUT = "last_hidden_state"
        private const val DECODER_INPUT_IDS = "input_ids"
        private const val DECODER_ENCODER_HS = "encoder_hidden_states"
        private const val DECODER_OUTPUT_LOGITS = "logits"

        /** Map RN language code -> Whisper language code (usually same) */
        private val LANG_MAP = mapOf(
            "hi" to "hi", "mr" to "mr", "ta" to "ta", "te" to "te",
            "kn" to "kn", "ml" to "ml", "bn" to "bn", "gu" to "gu", "pa" to "pa",
            "ur" to "ur"
        )
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val melSpec = MelSpectrogram()
    private val tokenizer = WhisperTokenizer(assets)

    private val encoderSession: OrtSession by lazy {
        val modelPath = File(modelsDir, "whisper-small/encoder_model_quantized.onnx")
        if (!modelPath.exists()) throw IllegalStateException("Whisper encoder not found: $modelPath")
        Log.d(TAG, "Loading encoder from $modelPath")
        env.createSession(modelPath.absolutePath, OrtSession.SessionOptions())
    }

    private val decoderSession: OrtSession by lazy {
        val modelPath = File(modelsDir, "whisper-small/decoder_model_quantized.onnx")
        if (!modelPath.exists()) throw IllegalStateException("Whisper decoder not found: $modelPath")
        Log.d(TAG, "Loading decoder from $modelPath")
        env.createSession(modelPath.absolutePath, OrtSession.SessionOptions())
    }

    /**
     * Transcribe base64 Int16 LE PCM 16kHz mono audio to text.
     */
    fun transcribe(base64Audio: String): String {
        // 1. Decode base64 -> Int16 LE PCM bytes -> float samples
        val pcmBytes = Base64.decode(base64Audio, Base64.DEFAULT)
        val samples = int16BytesToFloats(pcmBytes)
        Log.d(TAG, "PCM samples: ${samples.size}")

        // 2. Compute mel spectrogram [80 * 3000]
        val melData = melSpec.compute(samples)

        // 3. Create encoder input tensor [1, 80, 3000]
        val encoderInput = createFloatTensor(melData, longArrayOf(1L, 80L, 3000L))
        val encoderInputs = mapOf(ENCODER_INPUT to encoderInput)
        val encoderOutputs = encoderSession.run(encoderInputs)
        val hiddenState = encoderOutputs[ENCODER_OUTPUT].get() as OnnxTensor

        // hidden state shape: [1, 1500, 768]
        val hsData = (hiddenState.value as Array<*>).let { arr ->
            flattenHiddenState(arr)
        }
        val hsShape = hiddenState.info.shape // [1, 1500, 768]
        encoderInput.close()
        encoderOutputs.close()

        // 4. Build decoder prompt
        val whisperLang = LANG_MAP[languageCode] ?: "hi"
        val promptIds = tokenizer.getDecoderPromptIds(whisperLang)

        // 5. Greedy decode
        val outputIds = mutableListOf<Long>()
        var decoderIds = promptIds.toMutableList()

        val hsTensor = createFloatTensor(hsData, hsShape)

        for (step in 0 until MAX_DECODE_LEN) {
            val inputIdsTensor = createLongTensor(
                decoderIds.toLongArray(),
                longArrayOf(1L, decoderIds.size.toLong())
            )
            val decoderInputs = mapOf(
                DECODER_INPUT_IDS to inputIdsTensor,
                DECODER_ENCODER_HS to hsTensor
            )
            val decoderOutputs = decoderSession.run(decoderInputs)
            val logitsTensor = decoderOutputs[DECODER_OUTPUT_LOGITS].get() as OnnxTensor
            // logits shape: [1, seq_len, vocab_size]
            // We only need the last timestep
            val logitsData = flattenLogits(logitsTensor.value)
            val seqLen = decoderIds.size
            val vocabSize = WhisperTokenizer.VOCAB_SIZE
            val lastLogitsStart = (seqLen - 1) * vocabSize
            var maxLogit = Float.NEGATIVE_INFINITY
            var nextTokenId = 0L
            for (v in 0 until vocabSize) {
                val logit = logitsData[lastLogitsStart + v]
                if (logit > maxLogit) {
                    maxLogit = logit
                    nextTokenId = v.toLong()
                }
            }

            inputIdsTensor.close()
            decoderOutputs.close()

            if (nextTokenId == WhisperTokenizer.EOS_ID) break

            // Skip special tokens in output but include in decoder input
            if (nextTokenId < WhisperTokenizer.EOS_ID) {
                outputIds.add(nextTokenId)
            }
            decoderIds.add(nextTokenId)
        }

        hsTensor.close()
        hiddenState.close()

        return tokenizer.decode(outputIds.toLongArray())
    }

    private fun int16BytesToFloats(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val count = bytes.size / 2
        val floats = FloatArray(count)
        for (i in 0 until count) {
            floats[i] = buf.short.toFloat() / 32768.0f
        }
        return floats
    }

    @Suppress("UNCHECKED_CAST")
    private fun flattenHiddenState(value: Any?): FloatArray {
        // value is Array<Array<FloatArray>> for [1, 1500, 768]
        return when (value) {
            is Array<*> -> {
                val outer = value as Array<Array<FloatArray>>
                val result = mutableListOf<Float>()
                for (seq in outer[0]) {
                    for (v in seq) result.add(v)
                }
                result.toFloatArray()
            }
            else -> throw IllegalArgumentException("Unexpected hidden state type: ${value?.javaClass}")
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun flattenLogits(value: Any?): FloatArray {
        // value is Array<Array<FloatArray>> for [1, seqLen, vocabSize]
        return when (value) {
            is Array<*> -> {
                val outer = value as Array<Array<FloatArray>>
                val result = mutableListOf<Float>()
                for (seq in outer[0]) {
                    for (v in seq) result.add(v)
                }
                result.toFloatArray()
            }
            else -> throw IllegalArgumentException("Unexpected logits type: ${value?.javaClass}")
        }
    }

    private fun createFloatTensor(data: FloatArray, shape: LongArray): OnnxTensor {
        val buf = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        buf.asFloatBuffer().put(data)
        buf.rewind()
        return OnnxTensor.createTensor(env, buf.asFloatBuffer(), shape)
    }

    private fun createLongTensor(data: LongArray, shape: LongArray): OnnxTensor {
        val buf = ByteBuffer.allocateDirect(data.size * 8).order(ByteOrder.nativeOrder())
        buf.asLongBuffer().put(data)
        buf.rewind()
        return OnnxTensor.createTensor(env, buf.asLongBuffer(), shape)
    }

    override fun close() {
        try { encoderSession.close() } catch (_: Exception) {}
        try { decoderSession.close() } catch (_: Exception) {}
    }
}
