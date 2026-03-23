package com.indicai.engine

import ai.onnxruntime.*
import android.content.res.AssetManager
import android.util.Log
import com.indicai.tokenizer.IndicTransTokenizer
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * IndicTrans2 neural machine translation engine.
 * Supports both Indic→EN and EN→Indic directions.
 */
class TranslationEngine(
    private val modelFile: File,
    val direction: Direction,
    private val assets: AssetManager
) : Closeable {

    enum class Direction { INDIC_TO_EN, EN_TO_INDIC }

    companion object {
        private const val TAG = "TranslationEngine"
        private const val MAX_DECODE_STEPS = 128
        private const val DECODER_START_ID = 2L  // EOS used as decoder_start_id in IndicTrans2
        private const val EOS_ID = 2L
        private const val PAD_ID = 1L

        /** Maps short ISO codes to IndicTrans2 Flores-200 codes */
        val LANG_CODE_MAP = mapOf(
            "hi" to "hin_Deva",
            "mr" to "mar_Deva",
            "ta" to "tam_Taml",
            "te" to "tel_Telu",
            "kn" to "kan_Knda",
            "ml" to "mal_Mlym",
            "bn" to "ben_Beng",
            "gu" to "guj_Gujr",
            "pa" to "pan_Guru",
            "ur" to "urd_Arab",
            "en" to "eng_Latn"
        )
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private val tokenizerDirection = when (direction) {
        Direction.INDIC_TO_EN -> IndicTransTokenizer.Direction.INDIC_TO_EN
        Direction.EN_TO_INDIC -> IndicTransTokenizer.Direction.EN_TO_INDIC
    }
    private val tokenizer = IndicTransTokenizer(assets, tokenizerDirection)

    private val session: OrtSession by lazy {
        if (!modelFile.exists()) throw IllegalStateException("MT model not found: $modelFile")
        Log.d(TAG, "Loading MT model ($direction) from $modelFile (size=${modelFile.length()} bytes)")
        val sess = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
        Log.d(TAG, "MT model inputs: ${sess.inputNames.toList()}")
        Log.d(TAG, "MT model outputs: ${sess.outputNames.toList()}")
        sess
    }

    /**
     * Translate text from source to target language.
     * @param text Input text
     * @param srcLang Source language code (e.g. "hin_Deva" or "eng_Latn")
     * @param tgtLang Target language code
     */
    fun translate(text: String, srcLang: String, tgtLang: String): String {
        // Map short codes to IndicTrans2 language codes
        val srcLangCode = mapToIndicTransLang(srcLang, direction, isSource = true)
        val tgtLangCode = mapToIndicTransLang(tgtLang, direction, isSource = false)

        Log.d(TAG, "Translating [$srcLangCode -> $tgtLangCode]: $text")

        // Tokenize input
        val inputIds = tokenizer.encode(text, srcLangCode, tgtLangCode)
        val seqLen = inputIds.size.toLong()

        // Attention mask: all ones
        val attentionMask = LongArray(inputIds.size) { 1L }

        val inputIdsTensor = createLongTensor(inputIds, longArrayOf(1L, seqLen))
        val attentionMaskTensor = createLongTensor(attentionMask, longArrayOf(1L, seqLen))

        // Greedy decode
        val outputIds = mutableListOf<Long>()
        var decoderIds = mutableListOf(DECODER_START_ID)

        for (step in 0 until MAX_DECODE_STEPS) {
            val decLen = decoderIds.size.toLong()
            val decoderInputIds = createLongTensor(
                decoderIds.toLongArray(),
                longArrayOf(1L, decLen)
            )

            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionMaskTensor,
                "decoder_input_ids" to decoderInputIds
            )

            val outputs = session.run(inputs)
            val logitsTensor = outputs["logits"].get() as OnnxTensor
            // logits shape: [1, decLen, vocabSize]
            val logitsData = flattenLogits(logitsTensor.value)
            val vocabSize = logitsData.size / decoderIds.size

            // Get last time step logits
            val lastOffset = (decoderIds.size - 1) * vocabSize
            var maxLogit = Float.NEGATIVE_INFINITY
            var nextId = EOS_ID
            for (v in 0 until vocabSize) {
                val logit = logitsData[lastOffset + v]
                if (logit > maxLogit) {
                    maxLogit = logit
                    nextId = v.toLong()
                }
            }

            decoderInputIds.close()
            outputs.close()

            if (nextId == EOS_ID) break
            outputIds.add(nextId)
            decoderIds.add(nextId)
        }

        inputIdsTensor.close()
        attentionMaskTensor.close()

        Log.d(TAG, "Decoder output ids (${outputIds.size} tokens): ${outputIds.take(30).joinToString(",", "[", if (outputIds.size > 30) "...]" else "]")}")
        return tokenizer.decode(outputIds.toLongArray())
    }

    /**
     * Maps short language codes (hi, en, ta...) to IndicTrans2 language codes (hin_Deva, eng_Latn...).
     */
    private fun mapToIndicTransLang(code: String, direction: Direction, isSource: Boolean): String {
        return LANG_CODE_MAP[code] ?: code
    }

    @Suppress("UNCHECKED_CAST")
    private fun flattenLogits(value: Any?): FloatArray {
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
