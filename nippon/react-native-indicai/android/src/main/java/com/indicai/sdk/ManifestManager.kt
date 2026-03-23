// ManifestManager.kt — fetches + caches manifest.json from CDN

package com.indicai.sdk

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File

private const val TAG          = "IndicAI.Manifest"
private const val CACHE_TTL_MS = 6 * 60 * 60 * 1000L

data class ModelEntry(
    val file:    String,
    val sha256:  String,
    val sizeMb:  Double,
    val version: Int,
)

data class SDKManifest(
    val version:      Int,
    val updatedAt:    String,
    val whisper:      ModelEntry,
    val indicEn:      ModelEntry,
    val indicFromEn:  ModelEntry,
    val languageTts:  Map<String, ModelEntry>,
    val intent:       ModelEntry? = null,
)

internal class ManifestManager(
    private val context:     Context,
    private val manifestUrl: String,
    private val http:        OkHttpClient,
) {
    private val cacheFile  = File(context.filesDir, "indicai_manifest.json")
    private val tsFile     = File(context.filesDir, "indicai_manifest_ts")

    suspend fun fetch(): SDKManifest = withContext(Dispatchers.IO) {
        val age = System.currentTimeMillis() -
            (tsFile.takeIf { it.exists() }?.readText()?.toLongOrNull() ?: 0L)

        if (age < CACHE_TTL_MS && cacheFile.exists()) {
            Log.d(TAG, "Using cached manifest")
            return@withContext parse(cacheFile.readText())
        }
        try {
            val body = http.newCall(Request.Builder().url(manifestUrl).build())
                .execute().use { it.body!!.string() }
            cacheFile.writeText(body)
            tsFile.writeText(System.currentTimeMillis().toString())
            parse(body)
        } catch (e: Exception) {
            if (cacheFile.exists()) parse(cacheFile.readText())
            else throw IllegalStateException("No manifest available. Check network.", e)
        }
    }

    private fun parse(json: String): SDKManifest {
        val root   = JSONObject(json)
        val models = root.getJSONObject("models")
        val langs  = root.getJSONObject("languages")

        fun JSONObject.toEntry() = ModelEntry(
            file    = getString("file"),
            sha256  = getString("sha256"),
            sizeMb  = getDouble("size_mb"),
            version = getInt("version"),
        )

        val tts = mutableMapOf<String, ModelEntry>()
        langs.keys().forEach { code ->
            val obj = langs.getJSONObject(code)
            if (obj.has("tts")) tts[code] = obj.getJSONObject("tts").toEntry()
        }

        return SDKManifest(
            version     = root.getInt("version"),
            updatedAt   = root.getString("updated_at"),
            whisper     = models.getJSONObject("whisper").toEntry(),
            indicEn     = models.getJSONObject("indic_en").toEntry(),
            indicFromEn = models.getJSONObject("indic_from_en").toEntry(),
            languageTts = tts,
            intent      = if (models.has("intent")) models.getJSONObject("intent").toEntry() else null,
        )
    }
}
