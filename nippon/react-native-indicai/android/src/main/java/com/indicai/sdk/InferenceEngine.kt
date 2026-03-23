package com.indicai.sdk

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.*

private const val TAG = "IndicAI.Inference"
private const val SAMPLE_RATE = 16000
private const val N_MELS = 80
private const val N_FRAMES = 3000
private const val HOP_LENGTH = 160
private const val N_FFT = 400
private const val CHUNK_LENGTH_S = 30

internal class InferenceEngine(private val context: Context) {
    private val env = OrtEnvironment.getEnvironment()

    private var whisperEncSession: OrtSession? = null   // encoder: input_features → last_hidden_state
    private var whisperDecSession: OrtSession? = null   // decoder: input_ids + encoder_hidden_states → logits
    private var indicEnSession: OrtSession? = null
    private var enIndicSession: OrtSession? = null
    private var ttsSession: OrtSession? = null

    // Tokenizer data loaded from assets
    private var indicEnSrcVocab: Map<String, Int> = emptyMap()
    private var indicEnTgtVocab: Map<Int, String> = emptyMap()
    private var enIndicSrcVocab: Map<String, Int> = emptyMap()
    private var enIndicTgtVocab: Map<Int, String> = emptyMap()
    private var whisperVocab: Map<Int, String> = emptyMap()
    private var whisperMerges: List<Pair<String, String>> = emptyList()
    private var ttsVocabs: MutableMap<String, Map<String, Int>> = mutableMapOf()
    private var melFilters: FloatArray = FloatArray(0)

    // Read from bundled config.json — not hardcoded, changes if models change
    private var indicEnDecoderStartId = 2
    private var indicEnEosTokenId = 2
    private var enIndicDecoderStartId = 2
    private var enIndicEosTokenId = 2

    private fun opts(threads: Int) = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }

    fun initTokenizers() {
        indicEnSrcVocab = loadVocabAsMap("tokenizers/indic-en/dict.SRC.json")
        indicEnTgtVocab = loadVocabAsReverseMap("tokenizers/indic-en/dict.TGT.json")
        enIndicSrcVocab = loadVocabAsMap("tokenizers/en-indic/dict.SRC.json")
        enIndicTgtVocab = loadVocabAsReverseMap("tokenizers/en-indic/dict.TGT.json")
        whisperVocab = loadVocabAsReverseMap("tokenizers/whisper/vocab.json")
        whisperMerges = loadMerges("tokenizers/whisper/merges.txt")
        melFilters = computeMelFilterbank()

        val ieCfg = loadJsonAsset("tokenizers/indic-en/config.json")
        indicEnDecoderStartId = ieCfg.optInt("decoder_start_token_id", 2)
        indicEnEosTokenId = ieCfg.optInt("eos_token_id", 2)

        val eiCfg = loadJsonAsset("tokenizers/en-indic/config.json")
        enIndicDecoderStartId = eiCfg.optInt("decoder_start_token_id", 2)
        enIndicEosTokenId = eiCfg.optInt("eos_token_id", 2)

        Log.d(TAG, "Tokenizers initialized: indicEn=${indicEnSrcVocab.size} whisper=${whisperVocab.size}")
        Log.d(TAG, "IndicEn decoder_start=$indicEnDecoderStartId eos=$indicEnEosTokenId | EnIndic decoder_start=$enIndicDecoderStartId eos=$enIndicEosTokenId")
    }

    fun loadTtsVocab(langCode: String) {
        if (ttsVocabs.containsKey(langCode)) return
        ttsVocabs[langCode] = loadVocabAsMap("tokenizers/tts/vocab_$langCode.json")
        Log.d(TAG, "TTS vocab loaded for $langCode: ${ttsVocabs[langCode]!!.size} entries")
    }

    fun loadWhisper(f: File) {
        if (whisperEncSession != null) return  // already loaded
        // f is the first .onnx found by ModelManager — sibling files are in same dir
        val dir = f.parentFile ?: throw IllegalStateException("Whisper model dir not found")
        val encFile = File(dir, "encoder_model_quantized.onnx")
        val decFile = File(dir, "decoder_model_quantized.onnx")
        if (!encFile.exists()) throw IllegalStateException("Whisper encoder not found: $encFile")
        if (!decFile.exists()) throw IllegalStateException("Whisper decoder not found: $decFile")
        whisperEncSession = env.createSession(encFile.absolutePath, opts(4))
        whisperDecSession = env.createSession(decFile.absolutePath, opts(4))
        Log.d(TAG, "Whisper loaded: enc=${encFile.name}, dec=${decFile.name}")
    }

    fun loadIndicEn(f: File) {
        indicEnSession?.close()
        indicEnSession = env.createSession(f.absolutePath, opts(4))
        Log.d(TAG, "IndicTrans2 Indic→EN loaded")
    }

    fun loadEnIndic(f: File) {
        enIndicSession?.close()
        enIndicSession = env.createSession(f.absolutePath, opts(4))
        Log.d(TAG, "IndicTrans2 EN→Indic loaded")
    }

    fun loadTts(f: File) {
        if (ttsSession != null) return  // already loaded, skip reload
        ttsSession = env.createSession(f.absolutePath, opts(2))
        Log.d(TAG, "TTS loaded")
    }

    // ── Whisper STT ─────────────────────────────────────────────────────────────

    fun transcribe(pcm: FloatArray, lang: LanguageConfig): String {
        val encSession = whisperEncSession ?: throw IllegalStateException("Whisper not loaded")
        val decSession = whisperDecSession ?: throw IllegalStateException("Whisper not loaded")
        val mel = computeLogMel(pcm)

        // ── Stage 1: Encoder ───────────────────────────────────────────────────
        val featTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(mel),
            longArrayOf(1, N_MELS.toLong(), N_FRAMES.toLong())
        )
        val encOut = encSession.run(mapOf("input_features" to featTensor))
        val encHidden = encOut[0] as OnnxTensor  // (1, 1500, 768)

        // ── Stage 2: Decoder loop ──────────────────────────────────────────────
        val langId = WHISPER_LANG_TOKENS[lang.whisperLang] ?: 50301
        // Forced prompt: <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>
        val promptIds = longArrayOf(50258L, langId.toLong(), 50359L, 50363L)
        var decoderIds = promptIds.copyOf()
        val eosId = 50257L
        val maxSteps = 224

        for (step in 0 until maxSteps) {
            val decTensor = OnnxTensor.createTensor(
                env, LongBuffer.wrap(decoderIds),
                longArrayOf(1, decoderIds.size.toLong())
            )
            val out = decSession.run(
                mapOf("input_ids" to decTensor, "encoder_hidden_states" to encHidden)
            )
            val logits = out[0].value as Array<Array<FloatArray>>
            val lastLogits = logits[0][logits[0].size - 1]
            val nextToken = lastLogits.indices.maxByOrNull { lastLogits[it] } ?: break
            decTensor.close()
            out.close()
            if (nextToken.toLong() == eosId) break
            decoderIds = decoderIds + nextToken.toLong()
        }

        featTensor.close()
        encHidden.close()
        encOut.close()

        // Skip forced prompt tokens; decode the generated portion
        val tokenIds = decoderIds.drop(promptIds.size).map { it.toInt() }.filter { it < 50257 }
        Log.d(TAG, "Whisper prompt ids: ${promptIds.toList()}")
        Log.d(TAG, "Whisper raw token ids (${tokenIds.size}): $tokenIds")
        val pieces = tokenIds.mapNotNull { whisperVocab[it] }
        Log.d(TAG, "Whisper token pieces: $pieces")
        return whisperBpeDecode(tokenIds)
    }

    // ── IndicTrans2 MT ──────────────────────────────────────────────────────────

    fun translateToEn(text: String, lang: LanguageConfig): String {
        val session = indicEnSession ?: throw IllegalStateException("IndicTrans2 Indic→EN not loaded")
        val normalized = unicodeNormalize(text.trim())
        val inputIds = indicTransTokenize(normalized, lang.flores, "eng_Latn", indicEnSrcVocab)
        Log.d(TAG, "translateToEn input_ids(${inputIds.size}): ${inputIds.take(15).joinToString(",", "[", "]")}")
        return indicTransInfer(session, inputIds, indicEnDecoderStartId, indicEnEosTokenId, indicEnTgtVocab, "eng_Latn")
    }

    fun translateToIndic(text: String, lang: LanguageConfig): String {
        val session = enIndicSession ?: throw IllegalStateException("IndicTrans2 EN→Indic not loaded")
        val normalized = unicodeNormalize(text.trim())
        val inputIds = indicTransTokenize(normalized, "eng_Latn", lang.flores, enIndicSrcVocab)
        Log.d(TAG, "translateToIndic input_ids(${inputIds.size}): ${inputIds.take(15).joinToString(",", "[", "]")}")
        return indicTransInfer(session, inputIds, enIndicDecoderStartId, enIndicEosTokenId, enIndicTgtVocab, lang.flores)
    }

    private fun indicTransInfer(
        session: OrtSession,
        inputIds: LongArray,
        decoderStartId: Int,
        eosId: Int,
        tgtVocab: Map<Int, String>,
        tgtLang: String
    ): String {
        val attn = LongArray(inputIds.size) { 1L }
        val shape = longArrayOf(1, inputIds.size.toLong())
        val inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape)
        val attnTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attn), shape)

        val outIds = greedyDecode(session, inputTensor, attnTensor, decoderStartId, eosId, 128)
        inputTensor.close()
        attnTensor.close()

        val rawText = indicTransDetokenize(outIds, tgtVocab)
        Log.d(TAG, "decode outIds(${outIds.size}): ${outIds.take(15).joinToString(",", "[", "]")}")
        Log.d(TAG, "decode rawText: '$rawText'")
        val result = indicPostprocess(rawText, tgtLang)
        Log.d(TAG, "translate result: '$result'")
        return result
    }

    private fun greedyDecode(
        session: OrtSession,
        inputTensor: OnnxTensor,
        attnTensor: OnnxTensor,
        startId: Int,
        eosId: Int,
        maxLen: Int
    ): List<Int> {
        val ids = mutableListOf(startId)

        for (step in 0 until maxLen) {
            val decArray = ids.map { it.toLong() }.toLongArray()
            val decTensor = OnnxTensor.createTensor(
                env, LongBuffer.wrap(decArray),
                longArrayOf(1, decArray.size.toLong())
            )
            val out = session.run(
                mapOf(
                    "input_ids" to inputTensor,
                    "attention_mask" to attnTensor,
                    "decoder_input_ids" to decTensor
                ),
                setOf("logits")
            )
            val logits = out[0].value as Array<Array<FloatArray>>
            val lastLogits = logits[0][logits[0].size - 1]
            val nextToken = lastLogits.indices.maxByOrNull { lastLogits[it] } ?: break

            decTensor.close()
            out.close()

            if (nextToken == eosId) break
            ids.add(nextToken)
        }
        return ids
    }

    // ── MMS-TTS ─────────────────────────────────────────────────────────────────

    fun synthesize(text: String, langCode: String): FloatArray {
        val session = ttsSession ?: throw IllegalStateException("TTS not loaded")
        val vocab = ttsVocabs[langCode]
            ?: throw IllegalStateException("TTS vocab not loaded for $langCode")

        val normalizedText = unicodeNormalize(text)
        val ids = ttsTokenize(normalizedText, vocab)
        if (ids.isEmpty()) throw IllegalArgumentException("TTS tokenization produced empty input for '$text'")

        val tensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(ids),
            longArrayOf(1, ids.size.toLong())
        )
        val inputMap = mutableMapOf<String, OnnxTensor>()
        inputMap["input_ids"] = tensor

        // Some exported MMS-TTS graphs require attention_mask in addition to input_ids.
        if (session.inputInfo.containsKey("attention_mask")) {
            val attn = LongArray(ids.size) { 1L }
            val attnTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(attn),
                longArrayOf(1, attn.size.toLong())
            )
            inputMap["attention_mask"] = attnTensor
        }

        val out = session.run(inputMap)
        val waveform = (out[0].value as Array<FloatArray>)[0]
        inputMap.values.forEach { it.close() }
        out.close()

        val maxAbs = waveform.maxOfOrNull { abs(it) } ?: 0f
        if (maxAbs > 0f) {
            val scale = 0.9f / maxAbs
            for (i in waveform.indices) waveform[i] *= scale
        }
        return waveform
    }

    fun close() {
        try { whisperEncSession?.close() } catch (_: Exception) {}
        try { whisperDecSession?.close() } catch (_: Exception) {}
        try { indicEnSession?.close() } catch (_: Exception) {}
        try { enIndicSession?.close() } catch (_: Exception) {}
        try { ttsSession?.close() } catch (_: Exception) {}
        whisperEncSession = null
        whisperDecSession = null
        indicEnSession = null
        enIndicSession = null
        ttsSession = null
    }

    // ── IndicTrans2 Tokenizer ───────────────────────────────────────────────────

    /**
     * Encode text for IndicTrans2 in HuggingFace format:
     *   [srcLangId, tgtLangId, ▁word1Tokens..., ▁word2Tokens..., EOS]
     *
     * Language codes (hin_Deva, eng_Latn, etc.) are special tokens and looked
     * up directly — they are NOT passed through BPE segmentation.
     * Every text word gets ▁ prefix (handled internally by bpeSegment).
     */
    private fun indicTransTokenize(
        text: String,
        srcLang: String,
        tgtLang: String,
        vocab: Map<String, Int>
    ): LongArray {
        val unkId = vocab["<unk>"] ?: 3
        val eosId = vocab["</s>"] ?: 2

        val ids = mutableListOf<Long>()

        // Language code tokens first — direct lookup, no BPE
        ids.add((vocab[srcLang] ?: unkId).toLong())
        ids.add((vocab[tgtLang] ?: unkId).toLong())

        // Text words — each gets ▁ prefix via bpeSegment
        val words = text.trim().split("\\s+".toRegex()).filter { it.isNotEmpty() }
        for (word in words) {
            val subwords = bpeSegment(word, vocab)
            for (sw in subwords) {
                ids.add((vocab[sw] ?: unkId).toLong())
            }
        }

        ids.add(eosId.toLong())
        return ids.toLongArray()
    }

    private fun bpeSegment(word: String, vocab: Map<String, Int>): List<String> {
        if (word.isEmpty()) return emptyList()

        val spWord = "▁$word"
        if (vocab.containsKey(spWord)) return listOf(spWord)

        val chars = spWord.toList()
        val pieces = mutableListOf<String>()
        var i = 0

        while (i < chars.size) {
            var bestLen = 0
            var bestPiece = ""

            for (end in minOf(chars.size, i + 20) downTo i + 1) {
                val candidate = if (pieces.isEmpty() && i == 0) {
                    chars.subList(i, end).joinToString("")
                } else if (i == 0) {
                    chars.subList(i, end).joinToString("")
                } else {
                    chars.subList(i, end).joinToString("")
                }
                if (vocab.containsKey(candidate) && (end - i) > bestLen) {
                    bestLen = end - i
                    bestPiece = candidate
                }
            }

            if (bestLen > 0) {
                pieces.add(bestPiece)
                i += bestLen
            } else {
                pieces.add(chars[i].toString())
                i++
            }
        }

        return pieces
    }

    private fun indicTransDetokenize(ids: List<Int>, vocab: Map<Int, String>): String {
        val specialIds = setOf(0, 1, 2, 3) // <s>, <pad>, </s>, <unk>
        val tokens = ids.filter { it !in specialIds }.mapNotNull { vocab[it] }
        return tokens.joinToString("").replace("▁", " ").trim()
    }

    // ── IndicProcessor Pre/Post-processing ──────────────────────────────────────

    private fun indicPostprocess(text: String, lang: String): String {
        return text.replace("  ", " ").trim()
    }

    private fun unicodeNormalize(text: String): String {
        return java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
    }

    // ── TTS Character Tokenizer ─────────────────────────────────────────────────

    private fun ttsTokenize(text: String, vocab: Map<String, Int>): LongArray {
        // VitsTokenizer with add_blank=true: insert blank (ID 0) before each char and at end
        // Pattern: [0, char1, 0, char2, 0, char3, 0, ...]
        val BLANK_ID = 0L
        val ids = mutableListOf<Long>()
        for (ch in text) {
            val id = vocab[ch.toString()] ?: continue
            ids.add(BLANK_ID)
            ids.add(id.toLong())
        }
        if (ids.isNotEmpty()) ids.add(BLANK_ID)
        return ids.toLongArray()
    }

    // ── Whisper BPE Decoder ─────────────────────────────────────────────────────

    /**
     * Decode Whisper token IDs to text.
     * Whisper uses GPT-2 byte-level BPE where each byte value maps to a unicode char
     * via bytes_to_unicode(). We invert that mapping to recover the original UTF-8 bytes.
     */
    private fun whisperBpeDecode(tokenIds: List<Int>): String {
        val pieces = tokenIds.mapNotNull { whisperVocab[it] }
        val joined = pieces.joinToString("")
        // Invert GPT-2 bytes_to_unicode: each char maps back to a byte value
        val bytes = ByteArray(joined.length * 4)
        var len = 0
        for (ch in joined) {
            val b = WHISPER_BYTES_DECODER[ch]
            if (b != null) {
                bytes[len++] = b.toByte()
            } else {
                // Not in bytes map — copy raw UTF-8 (shouldn't happen for valid tokens)
                val utf8 = ch.toString().toByteArray(Charsets.UTF_8)
                utf8.copyInto(bytes, len); len += utf8.size
            }
        }
        return String(bytes, 0, len, Charsets.UTF_8).trim()
    }

    // ── Log-Mel Spectrogram ─────────────────────────────────────────────────────

    private fun computeLogMel(audio: FloatArray): FloatArray {
        val targetLen = SAMPLE_RATE * CHUNK_LENGTH_S
        val padded = FloatArray(targetLen)
        audio.copyInto(padded, 0, 0, minOf(audio.size, targetLen))

        val numFrames = N_FRAMES
        val stft = Array(N_FFT / 2 + 1) { FloatArray(numFrames) }
        val window = hanningWindow(N_FFT)

        for (frame in 0 until numFrames) {
            val start = frame * HOP_LENGTH
            val real = FloatArray(N_FFT)
            val imag = FloatArray(N_FFT)

            for (i in 0 until N_FFT) {
                val idx = start + i
                real[i] = if (idx < padded.size) padded[idx] * window[i] else 0f
            }

            fft(real, imag, N_FFT)

            for (k in 0..N_FFT / 2) {
                stft[k][frame] = real[k] * real[k] + imag[k] * imag[k]
            }
        }

        val mel = FloatArray(N_MELS * numFrames)
        for (m in 0 until N_MELS) {
            for (f in 0 until numFrames) {
                var sum = 0f
                for (k in 0..N_FFT / 2) {
                    sum += melFilters[m * (N_FFT / 2 + 1) + k] * stft[k][f]
                }
                mel[m * numFrames + f] = ln(maxOf(sum, 1e-10f))
            }
        }

        val maxVal = mel.maxOrNull() ?: 0f
        for (i in mel.indices) {
            mel[i] = maxOf(mel[i], maxVal - 8f)
            mel[i] = (mel[i] + 4f) / 4f
        }

        return mel
    }

    private fun hanningWindow(n: Int): FloatArray {
        return FloatArray(n) { i ->
            (0.5f * (1f - cos(2.0 * PI * i / n))).toFloat()
        }
    }

    private fun fft(real: FloatArray, imag: FloatArray, n: Int) {
        // Cooley-Tukey below assumes power-of-two length. Whisper uses n_fft=400,
        // so we pad to next power of two to avoid index errors in on-device STT.
        if ((n and (n - 1)) != 0) {
            var m = 1
            while (m < n) m = m shl 1
            val pr = FloatArray(m)
            val pi = FloatArray(m)
            for (i in 0 until n) {
                pr[i] = real[i]
                pi[i] = imag[i]
            }
            fft(pr, pi, m)
            for (i in 0 until n) {
                real[i] = pr[i]
                imag[i] = pi[i]
            }
            return
        }

        if (n <= 1) return

        val bits = (ln(n.toDouble()) / ln(2.0)).toInt()
        for (i in 0 until n) {
            val j = reverseBits(i, bits)
            if (i < j) {
                val tr = real[i]; real[i] = real[j]; real[j] = tr
                val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
            }
        }

        var len = 2
        while (len <= n) {
            val halfLen = len / 2
            val angle = -2.0 * PI / len
            for (i in 0 until n step len) {
                for (j in 0 until halfLen) {
                    val wReal = cos(angle * j).toFloat()
                    val wImag = sin(angle * j).toFloat()
                    val ur = real[i + j]
                    val ui = imag[i + j]
                    val vr = real[i + j + halfLen] * wReal - imag[i + j + halfLen] * wImag
                    val vi = real[i + j + halfLen] * wImag + imag[i + j + halfLen] * wReal
                    real[i + j] = ur + vr
                    imag[i + j] = ui + vi
                    real[i + j + halfLen] = ur - vr
                    imag[i + j + halfLen] = ui - vi
                }
            }
            len *= 2
        }
    }

    private fun reverseBits(x: Int, bits: Int): Int {
        var result = 0
        var v = x
        for (i in 0 until bits) {
            result = (result shl 1) or (v and 1)
            v = v shr 1
        }
        return result
    }

    private fun computeMelFilterbank(): FloatArray {
        val nFreqs = N_FFT / 2 + 1
        val filters = FloatArray(N_MELS * nFreqs)

        val fMin = 0.0
        val fMax = SAMPLE_RATE / 2.0

        fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
        fun melToHz(mel: Double) = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = DoubleArray(N_MELS + 2) { i ->
            melToHz(melMin + (melMax - melMin) * i / (N_MELS + 1))
        }
        val freqBins = DoubleArray(N_MELS + 2) { i ->
            (melPoints[i] * (N_FFT + 1) / SAMPLE_RATE)
        }

        for (m in 0 until N_MELS) {
            val fLeft = freqBins[m]
            val fCenter = freqBins[m + 1]
            val fRight = freqBins[m + 2]

            for (k in 0 until nFreqs) {
                val kd = k.toDouble()
                val weight = when {
                    kd < fLeft -> 0.0
                    kd <= fCenter -> (kd - fLeft) / (fCenter - fLeft)
                    kd <= fRight -> (fRight - kd) / (fRight - fCenter)
                    else -> 0.0
                }
                filters[m * nFreqs + k] = weight.toFloat()
            }

            val norm = 2.0 / (melPoints[m + 2] - melPoints[m])
            for (k in 0 until nFreqs) {
                filters[m * nFreqs + k] *= norm.toFloat()
            }
        }

        return filters
    }

    // ── Asset Loading ───────────────────────────────────────────────────────────

    private fun loadJsonAsset(assetPath: String): JSONObject {
        val json = context.assets.open(assetPath).bufferedReader().use { it.readText() }
        return JSONObject(json)
    }

    private fun loadVocabAsMap(assetPath: String): Map<String, Int> {
        val json = context.assets.open(assetPath).bufferedReader().use { it.readText() }
        val obj = JSONObject(json)
        val map = HashMap<String, Int>(obj.length())
        val keys = obj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            map[key] = obj.getInt(key)
        }
        return map
    }

    private fun loadVocabAsReverseMap(assetPath: String): Map<Int, String> {
        val json = context.assets.open(assetPath).bufferedReader().use { it.readText() }
        val obj = JSONObject(json)
        val map = HashMap<Int, String>(obj.length())
        val keys = obj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            map[obj.getInt(key)] = key
        }
        return map
    }

    private fun loadMerges(assetPath: String): List<Pair<String, String>> {
        val lines = context.assets.open(assetPath).bufferedReader().readLines()
        return lines
            .filter { !it.startsWith("#") && it.isNotBlank() }
            .mapNotNull { line ->
                val parts = line.split(" ")
                if (parts.size == 2) Pair(parts[0], parts[1]) else null
            }
    }

    companion object {
        val WHISPER_LANG_TOKENS = mapOf(
            "hi" to 50276, "mr" to 50320, "ta" to 50287,
            "te" to 50299, "kn" to 50306, "ml" to 50296,
            "bn" to 50302, "gu" to 50333, "pa" to 50321, "ur" to 50290,
        )

        /**
         * Inverse of GPT-2 bytes_to_unicode().
         * Maps each unicode char back to its original byte value (0..255).
         */
        val WHISPER_BYTES_DECODER: Map<Char, Int> by lazy {
            val bs = mutableListOf<Int>()
            bs.addAll(33..126)
            bs.addAll(161..172)
            bs.addAll(174..255)
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
            }
            cs.mapIndexed { i, c -> c.toChar() to bs[i] }.toMap()
        }
    }
}
