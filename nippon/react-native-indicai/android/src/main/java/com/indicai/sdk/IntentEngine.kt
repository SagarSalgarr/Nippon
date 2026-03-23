package com.indicai.sdk

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import org.json.JSONArray
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.text.Normalizer

private const val TAG      = "IndicAI.Intent"
private const val MAX_LEN  = 64
private const val CLS_ID   = 101L
private const val SEP_ID   = 102L
private const val UNK_ID   = 100L
private const val PAD_ID   = 0L
private const val EMB_DIM  = 384

data class IntentResult(
    val intent: String,
    val confidence: Float,
)

/**
 * On-device intent classification using MiniLM-L6 semantic similarity.
 *
 * How it works (mirrors sentence-transformers on-device):
 *   1. WordPiece-tokenize the English query
 *   2. Run ONNX encoder → normalised 384-dim sentence embedding
 *   3. Cosine similarity against pre-computed intent bank (intent_bank.npy)
 *   4. Nearest neighbour → intent label + similarity score as confidence
 *
 * Files loaded from the intent zip (same S3 / ModelManager flow as other models):
 *   intent_encoder.onnx       — MiniLM encoder (outputs L2-normalised embedding)
 *   tokenizer/vocab.txt       — BERT WordPiece vocabulary
 *   intent_bank.npy           — pre-computed bank embeddings (N × 384 float32, row-major)
 *   intent_bank_labels.json   — JSON array of N label strings matching bank rows
 */
internal class IntentEngine {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var vocab: Map<String, Int> = emptyMap()
    private var bankVecs: Array<FloatArray> = emptyArray()   // [N, 384]
    private var bankLabels: List<String> = emptyList()

    fun isLoaded() = session != null

    fun load(modelFile: File, vocabFile: File, bankNpyFile: File, bankLabelsFile: File) {
        session?.close()
        session = env.createSession(
            modelFile.absolutePath,
            OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
        )
        vocab       = buildVocab(vocabFile)
        bankVecs    = loadNpy(bankNpyFile)
        bankLabels  = parseLabels(bankLabelsFile)
        Log.d(TAG, "Loaded intent engine: ${bankLabels.size} bank phrases, ${bankLabels.toSet().size} intents")
    }

    fun classify(text: String): IntentResult {
        val sess = session ?: throw IllegalStateException("IntentEngine not loaded")

        // 1. Tokenise
        val tokenIds = wordpiece(text.lowercase().trim())
        val attn = LongArray(MAX_LEN) { if (tokenIds[it] != PAD_ID) 1L else 0L }

        val inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(tokenIds), longArrayOf(1, MAX_LEN.toLong()))
        val attnMask = OnnxTensor.createTensor(env, LongBuffer.wrap(attn),     longArrayOf(1, MAX_LEN.toLong()))

        // 2. Encoder → normalised embedding (1, 384)
        val outputs = sess.run(mapOf("input_ids" to inputIds, "attention_mask" to attnMask))
        val rawValue: Any = outputs[0].value
        Log.d(TAG, "classify rawValue type: ${rawValue.javaClass.name}, bankVecs=${bankVecs.size}")
        @Suppress("UNCHECKED_CAST")
        val queryVec: FloatArray = when (rawValue) {
            is Array<*> -> @Suppress("UNCHECKED_CAST") (rawValue as Array<FloatArray>)[0]
            is FloatArray -> rawValue
            else -> throw IllegalStateException("Unexpected output type: ${rawValue.javaClass.name}")
        }

        // 3. Cosine similarity — vecs are already L2-normalised → dot product = cosine sim
        var bestIdx = 0
        var bestSim = -1f
        for (i in bankVecs.indices) {
            var dot = 0f
            val row = bankVecs[i]
            for (d in 0 until EMB_DIM) dot += row[d] * queryVec[d]
            if (dot > bestSim) { bestSim = dot; bestIdx = i }
        }

        val intent = bankLabels.getOrElse(bestIdx) { "unknown" }
        Log.d(TAG, "classify('$text') → $intent (sim=${String.format("%.3f", bestSim)})")
        return IntentResult(intent, bestSim)
    }

    fun close() {
        session?.close()
        session = null
    }

    // ── WordPiece tokenizer (matches bert-base-uncased BasicTokenizer) ─────────

    private fun wordpiece(text: String): LongArray {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFD)
            .filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
        val words  = normalized.split(Regex("\\s+")).filter { it.isNotEmpty() }
        val pieces = mutableListOf<Long>()
        pieces.add(CLS_ID)

        outer@ for (word in words) {
            if (pieces.size >= MAX_LEN - 1) break
            val subTokens = mutableListOf<Long>()
            var start = 0
            var unk = false
            while (start < word.length) {
                var end = word.length
                var found: Long? = null
                while (start < end) {
                    val sub = if (start == 0) word.substring(start, end)
                              else "##${word.substring(start, end)}"
                    val id = vocab[sub]
                    if (id != null) { found = id.toLong(); break }
                    end--
                }
                if (found == null) { unk = true; break }
                subTokens.add(found); start = end
            }
            if (unk) {
                if (pieces.size < MAX_LEN - 1) pieces.add(UNK_ID)
            } else {
                for (tok in subTokens) {
                    if (pieces.size >= MAX_LEN - 1) break@outer
                    pieces.add(tok)
                }
            }
        }
        pieces.add(SEP_ID)
        return LongArray(MAX_LEN) { if (it < pieces.size) pieces[it] else PAD_ID }
    }

    private fun buildVocab(file: File): Map<String, Int> {
        val map = mutableMapOf<String, Int>()
        file.useLines { lines -> lines.forEachIndexed { idx, line -> map[line.trimEnd()] = idx } }
        return map
    }

    /**
     * Parse .npy file (NPY format v1.0, float32, C-contiguous 2D array).
     * Returns Array<FloatArray> of shape [rows, EMB_DIM].
     */
    private fun loadNpy(file: File): Array<FloatArray> {
        val bytes = file.readBytes()
        // NPY magic: \x93NUMPY (6 bytes) + major + minor + header_len (2 bytes LE) + header
        var offset = 10 + ((bytes[8].toInt() and 0xFF) or ((bytes[9].toInt() and 0xFF) shl 8))
        val totalFloats = (bytes.size - offset) / 4
        val rows = totalFloats / EMB_DIM
        val result = Array(rows) { FloatArray(EMB_DIM) }
        for (r in 0 until rows) {
            for (d in 0 until EMB_DIM) {
                val i = offset + (r * EMB_DIM + d) * 4
                val bits = ((bytes[i].toInt() and 0xFF))        or
                           ((bytes[i+1].toInt() and 0xFF) shl 8)  or
                           ((bytes[i+2].toInt() and 0xFF) shl 16) or
                           ((bytes[i+3].toInt() and 0xFF) shl 24)
                result[r][d] = java.lang.Float.intBitsToFloat(bits)
            }
        }
        return result
    }

    private fun parseLabels(file: File): List<String> {
        val arr = JSONArray(file.readText())
        return List(arr.length()) { arr.getString(it) }
    }
}
