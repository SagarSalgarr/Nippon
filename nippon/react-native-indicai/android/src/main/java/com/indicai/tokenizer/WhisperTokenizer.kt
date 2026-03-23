package com.indicai.tokenizer

import android.content.res.AssetManager
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.InputStreamReader

/**
 * GPT-2 BPE tokenizer for Whisper (decode-only).
 * Handles special tokens and bytes_to_unicode inverse mapping.
 */
class WhisperTokenizer(assets: AssetManager) {
    companion object {
        private const val TAG = "WhisperTokenizer"
        private const val ASSET_VOCAB = "tokenizers/whisper/vocab.json"

        // Special token IDs
        const val SOT_ID = 50258L         // <|startoftranscript|>
        const val TRANSCRIBE_ID = 50359L  // <|transcribe|>
        const val NO_TIMESTAMPS_ID = 50363L
        const val EOS_ID = 50257L         // <|endoftext|>
        const val NOTIMESTAMPS_ID = 50363L
        const val VOCAB_SIZE = 51865
    }

    // vocab: token string -> id
    private val vocab: Map<String, Int>
    // reverseVocab: id -> token string
    private val reverseVocab: Map<Int, String>
    // bytes decoder: unicode char -> original byte value
    private val bytesDecoder: Map<Char, Int>

    init {
        val gson = Gson()
        val type = object : TypeToken<Map<String, Int>>() {}.type
        vocab = assets.open(ASSET_VOCAB).use { stream ->
            gson.fromJson(InputStreamReader(stream, Charsets.UTF_8), type)
        }
        reverseVocab = vocab.entries.associate { (k, v) -> v to k }
        bytesDecoder = buildBytesDecoder()
    }

    /**
     * Inverse of GPT-2 bytes_to_unicode.
     * Returns a map from the unicode representation char -> original byte value.
     */
    private fun buildBytesDecoder(): Map<Char, Int> {
        // bs = printable ASCII (33..126) + extended (161..172, 174..255)
        val bs = mutableListOf<Int>()
        bs.addAll(33..126)
        bs.addAll(161..172)
        bs.addAll(174..255)
        val cs = bs.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        // cs[i].toChar() -> bs[i] (byte value)
        return cs.mapIndexed { i, c -> c.toChar() to bs[i] }.toMap()
    }

    /**
     * Returns the decoder prompt token IDs for a given language code.
     * [SOT_ID, LANG_TOKEN_ID, TRANSCRIBE_ID, NO_TIMESTAMPS_ID]
     */
    fun getDecoderPromptIds(langCode: String): LongArray {
        val langToken = "<|$langCode|>"
        val langTokenId = vocab[langToken]?.toLong()
            ?: run {
                Log.w(TAG, "Language token not found: $langToken, defaulting to Hindi")
                vocab["<|hi|>"]?.toLong() ?: 50301L
            }
        return longArrayOf(SOT_ID, langTokenId, TRANSCRIBE_ID, NO_TIMESTAMPS_ID)
    }

    /**
     * Decode a sequence of token IDs to a string.
     * Skips special tokens (>= 50257), looks up in reverseVocab,
     * joins token strings, applies bytes decoder, decodes as UTF-8.
     */
    fun decode(ids: LongArray): String {
        // Filter out special tokens
        val tokens = ids.filter { it < EOS_ID }
            .mapNotNull { id -> reverseVocab[id.toInt()] }

        // Join token strings (they use GPT-2 unicode encoding)
        val joined = tokens.joinToString("")

        // Convert GPT-2 unicode chars back to UTF-8 bytes
        val bytes = ByteArray(joined.length)
        var validLen = 0
        for (ch in joined) {
            val byteVal = bytesDecoder[ch]
            if (byteVal != null) {
                bytes[validLen++] = byteVal.toByte()
            } else {
                // Character not in decoder map - use as-is UTF-8
                val utf8 = ch.toString().toByteArray(Charsets.UTF_8)
                utf8.copyInto(bytes, validLen)
                validLen += utf8.size
            }
        }

        return String(bytes, 0, validLen, Charsets.UTF_8).trim()
    }
}
