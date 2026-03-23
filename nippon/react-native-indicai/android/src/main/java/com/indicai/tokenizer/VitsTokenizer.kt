package com.indicai.tokenizer

import android.content.res.AssetManager
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.InputStreamReader

/**
 * Character-level tokenizer with blank insertion for MMS-TTS (VITS) models.
 *
 * Each character is separated by a blank token (id=0).
 * Format: [blank, char1, blank, char2, blank, ...]
 */
class VitsTokenizer(
    private val assets: AssetManager,
    private val languageCode: String
) {
    companion object {
        private const val TAG = "VitsTokenizer"
        private const val BLANK_ID = 0L
        private const val DEFAULT_UNK = "<unk>"
    }

    // vocab: character -> id
    private val vocab: Map<String, Int>
    private val unkId: Long

    init {
        val assetName = "tokenizers/tts/vocab_${languageCode}.json"
        val gson = Gson()
        val type = object : TypeToken<Map<String, Int>>() {}.type
        vocab = try {
            assets.open(assetName).use { stream ->
                gson.fromJson(InputStreamReader(stream, Charsets.UTF_8), type)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load TTS vocab for $languageCode: ${e.message}")
            emptyMap()
        }
        unkId = vocab[DEFAULT_UNK]?.toLong() ?: 1L
        Log.d(TAG, "Loaded TTS vocab for $languageCode: ${vocab.size} tokens")
    }

    /**
     * Encode text to token IDs with blank tokens inserted between characters.
     * Output format: [blank, char1, blank, char2, blank, ..., charN, blank]
     */
    fun encode(text: String): LongArray {
        val normalizedText = text.trim().lowercase()
        val ids = mutableListOf<Long>()

        for (ch in normalizedText) {
            // Add blank token before each character
            ids.add(BLANK_ID)
            // Add character token
            val chStr = ch.toString()
            val tokenId = vocab[chStr]?.toLong() ?: unkId
            ids.add(tokenId)
        }

        // Add final blank token
        ids.add(BLANK_ID)

        return ids.toLongArray()
    }
}
