package com.indicai.tokenizer

import android.content.res.AssetManager
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.InputStreamReader

/**
 * Greedy subword tokenizer for IndicTrans2 models.
 * Simulates SentencePiece-style tokenization using pre-built vocabulary dictionaries.
 *
 * Direction determines which vocab files to load:
 *   INDIC_TO_EN: indic_en_src_vocab.json (source), indic_en_tgt_vocab.json (target)
 *   EN_TO_INDIC: en_indic_src_vocab.json (source), en_indic_tgt_vocab.json (target)
 */
class IndicTransTokenizer(
    private val assets: AssetManager,
    private val direction: Direction
) {
    enum class Direction { INDIC_TO_EN, EN_TO_INDIC }

    companion object {
        private const val TAG = "IndicTransTokenizer"
        private const val BOS_ID = 0
        private const val PAD_ID = 1
        private const val EOS_ID = 2
        private const val UNK_ID = 3
        private const val SPACE_CHAR = "\u2581"  // ▁ (U+2581)
    }

    // srcVocab: token -> id  (for encoding)
    val srcVocab: Map<String, Int>
    // tgtVocab: id -> token  (for decoding)
    val tgtVocab: Map<Int, String>

    // Sorted vocab keys by length descending (for greedy matching)
    private val sortedSrcKeys: List<String>

    init {
        val gson = Gson()
        val strIntType = object : TypeToken<Map<String, Int>>() {}.type

        val (srcAsset, tgtAsset) = when (direction) {
            Direction.INDIC_TO_EN -> Pair("tokenizers/indic-en/dict.SRC.json", "tokenizers/indic-en/dict.TGT.json")
            Direction.EN_TO_INDIC -> Pair("tokenizers/en-indic/dict.SRC.json", "tokenizers/en-indic/dict.TGT.json")
        }

        srcVocab = assets.open(srcAsset).use { stream ->
            gson.fromJson(InputStreamReader(stream, Charsets.UTF_8), strIntType)
        }
        val tgtVocabStrInt: Map<String, Int> = assets.open(tgtAsset).use { stream ->
            gson.fromJson(InputStreamReader(stream, Charsets.UTF_8), strIntType)
        }
        tgtVocab = tgtVocabStrInt.entries.associate { (k, v) -> v to k }

        // Sort source vocab keys by length descending for greedy matching
        sortedSrcKeys = srcVocab.keys.sortedByDescending { it.length }

        Log.d(TAG, "Loaded $direction tokenizer: srcVocab=${srcVocab.size}, tgtVocab=${tgtVocab.size}")

        // Log a few known token IDs for verification
        Log.d(TAG, "Spot-check srcVocab: hin_Deva=${srcVocab["hin_Deva"]}, eng_Latn=${srcVocab["eng_Latn"]}, ▁मेरा=${srcVocab["\u2581मेरा"]}, ▁नाम=${srcVocab["\u2581नाम"]}")
        Log.d(TAG, "Spot-check tgtVocab[345]=${tgtVocab[345]}, tgtVocab[264]=${tgtVocab[264]}, tgtVocab[12]=${tgtVocab[12]}")
    }

    /**
     * Encode text for IndicTrans2.
     *
     * The HuggingFace IndicTrans2 tokenizer format (verified by running the actual tokenizer):
     *   "{srcLang} {tgtLang} {text}"
     *   e.g. "hin_Deva eng_Latn मेरा नाम सगर है"
     *   → token IDs: [8, 4, 2262, 352, 83571, 11, 2]
     *
     * Language codes are user-defined special tokens (single IDs, no ▁ prefix).
     * Every text word gets ▁ prefix since they all follow a space.
     * There are NO > or < bracket tokens in this format.
     */
    fun encode(text: String, srcLang: String, tgtLang: String): LongArray {
        val tokens = mutableListOf<Long>()

        // Language code special tokens (stored without ▁ prefix in vocab)
        val srcLangId = srcVocab[srcLang]?.toLong() ?: UNK_ID.toLong()
        val tgtLangId = srcVocab[tgtLang]?.toLong() ?: UNK_ID.toLong()
        tokens.add(srcLangId)
        tokens.add(tgtLangId)

        // All text words get ▁ prefix — they follow a space (after tgtLang token)
        val words = text.trim().split(" ").filter { it.isNotEmpty() }
        for (word in words) {
            tokenizeWord("$SPACE_CHAR$word", tokens)
        }

        tokens.add(EOS_ID.toLong())
        val result = tokens.toLongArray()
        Log.d(TAG, "encode('$text', $srcLang, $tgtLang) → ${result.take(20).joinToString(",", "[", if (result.size > 20) "...]" else "]")}")
        return result
    }

    private fun tokenizeWord(word: String, tokens: MutableList<Long>) {
        var remaining = word
        while (remaining.isNotEmpty()) {
            var matched = false
            // Try longest match first
            for (key in sortedSrcKeys) {
                if (remaining.startsWith(key)) {
                    tokens.add((srcVocab[key] ?: UNK_ID).toLong())
                    remaining = remaining.substring(key.length)
                    matched = true
                    break
                }
            }
            if (!matched) {
                // Fallback: emit character by character with ▁ prefix for spaces
                val ch = remaining[0].toString()
                // Try with space prefix variant
                val withSpace = "$SPACE_CHAR$ch"
                val tokenId = when {
                    srcVocab.containsKey(ch) -> srcVocab[ch]!!.toLong()
                    srcVocab.containsKey(withSpace) -> srcVocab[withSpace]!!.toLong()
                    else -> UNK_ID.toLong()
                }
                tokens.add(tokenId)
                remaining = remaining.substring(1)
            }
        }
    }

    /**
     * Decode output token IDs to text.
     * Skips BOS(0), PAD(1), EOS(2).
     * Replaces ▁ with space, strips leading space.
     */
    fun decode(ids: LongArray): String {
        val parts = ids
            .filter { it.toInt() !in listOf(BOS_ID, PAD_ID, EOS_ID) }
            .mapNotNull { id -> tgtVocab[id.toInt()] }

        val text = parts.joinToString("")
            .replace(SPACE_CHAR, " ")
            .trimStart()
            .trim()

        Log.d(TAG, "decode input ids: ${ids.take(30).joinToString(",", "[", if (ids.size > 30) "...]" else "]")}")
        Log.d(TAG, "decode raw tokens: ${parts.take(20)}")
        Log.d(TAG, "decode joined text (before strip): '$text'")

        // IndicTrans2 decoder always prefixes output with "> lang_code <" — strip it
        val stripped = text.replace(Regex("^>\\s*\\S+\\s*<\\s*"), "").trim()
        Log.d(TAG, "decode final output: '$stripped'")
        return stripped
    }
}
