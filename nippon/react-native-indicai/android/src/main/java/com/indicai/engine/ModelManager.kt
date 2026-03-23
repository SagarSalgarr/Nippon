package com.indicai.engine

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream

typealias ProgressCallback = (modelName: String, percent: Int, downloadedBytes: Long, totalBytes: Long) -> Unit

data class ModelEntry(
    val file: String,
    val sha256: String?,
    @SerializedName("size_mb") val sizeMb: Int
)

data class TtsEntry(
    val file: String,
    val sha256: String?,
    @SerializedName("size_mb") val sizeMb: Int
)

data class LanguageEntry(
    val name: String,
    val tts: TtsEntry?
)

data class Manifest(
    val version: Int,
    val models: Map<String, ModelEntry>,
    val languages: Map<String, LanguageEntry>
)

class ModelManager(
    private val context: Context,
    val modelsDir: File,
    private val manifestUrl: String,
    private val s3BaseUrl: String
) {
    companion object {
        const val DEFAULT_MANIFEST_URL =
            "https://indicai-cdn.s3.ap-south-1.amazonaws.com/manifest.json"
        const val DEFAULT_S3_BASE_URL =
            "https://indicai-cdn.s3.ap-south-1.amazonaws.com/models"
        private const val TAG = "ModelManager"

        private val MODEL_DIR_MAP = mapOf(
            "whisper" to "whisper-small",
            "indic_en" to "indic-trans2-indic-en",
            "indic_from_en" to "indic-trans2-en-indic"
        )
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()
    private var cachedManifest: Manifest? = null

    private fun fetchManifest(): Manifest {
        cachedManifest?.let { return it }
        val request = Request.Builder().url(manifestUrl).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) throw Exception("Failed to fetch manifest: ${response.code}")
        val body = response.body?.string() ?: throw Exception("Empty manifest response")
        val manifest = gson.fromJson(body, Manifest::class.java)
        cachedManifest = manifest
        return manifest
    }

    fun getSupportedLanguages(): List<String> {
        return try {
            fetchManifest().languages.keys.toList()
        } catch (e: Exception) {
            Log.w(TAG, "Could not fetch manifest for languages: ${e.message}")
            listOf("hi", "mr", "ta", "te", "kn", "ml", "bn", "gu", "pa")
        }
    }

    fun ensureModel(modelKey: String, progress: ProgressCallback) {
        val manifest = fetchManifest()
        val entry = manifest.models[modelKey]
            ?: throw Exception("Unknown model key: $modelKey")
        val dirName = MODEL_DIR_MAP[modelKey]
            ?: throw Exception("No dir mapping for model: $modelKey")
        val extractDir = File(modelsDir, dirName)

        // Determine which ONNX file(s) we need
        val expectedFiles = when (modelKey) {
            "whisper" -> listOf("encoder_model_quantized.onnx", "decoder_model_quantized.onnx")
            else -> listOf("model_quantized.onnx")
        }

        // Check if already extracted
        if (expectedFiles.all { File(extractDir, it).exists() }) {
            Log.d(TAG, "Model $modelKey already cached at $extractDir")
            progress(dirName, 100, 0L, 0L)
            return
        }

        // Download zip
        val zipFile = File(modelsDir, "${dirName}.zip")
        modelsDir.mkdirs()
        downloadFile(
            url = "$s3BaseUrl/${entry.file}",
            dest = zipFile,
            modelName = dirName,
            progress = progress
        )

        // Verify SHA256 if available
        entry.sha256?.let { expectedHash ->
            if (expectedHash.isNotBlank()) {
                val actualHash = sha256Hex(zipFile)
                if (!actualHash.equals(expectedHash, ignoreCase = true)) {
                    zipFile.delete()
                    throw Exception("SHA256 mismatch for $dirName: expected $expectedHash, got $actualHash")
                }
            }
        }

        // Extract
        extractDir.mkdirs()
        extractZip(zipFile, extractDir)
        zipFile.delete()
        Log.d(TAG, "Model $modelKey extracted to $extractDir")
    }

    fun ensureTtsModel(languageCode: String, progress: ProgressCallback) {
        val manifest = fetchManifest()
        val langEntry = manifest.languages[languageCode]
            ?: throw Exception("Language not supported: $languageCode")
        val ttsEntry = langEntry.tts
            ?: throw Exception("No TTS model for language: $languageCode")

        val dirName = "mms-tts-$languageCode"
        val extractDir = File(modelsDir, dirName)
        val modelFile = File(extractDir, "model.onnx")

        if (modelFile.exists()) {
            Log.d(TAG, "TTS model for $languageCode already cached")
            progress(dirName, 100, 0L, 0L)
            return
        }

        val zipFile = File(modelsDir, "${dirName}.zip")
        modelsDir.mkdirs()
        downloadFile(
            url = "$s3BaseUrl/${ttsEntry.file}",
            dest = zipFile,
            modelName = dirName,
            progress = progress
        )

        ttsEntry.sha256?.let { expectedHash ->
            if (expectedHash.isNotBlank()) {
                val actualHash = sha256Hex(zipFile)
                if (!actualHash.equals(expectedHash, ignoreCase = true)) {
                    zipFile.delete()
                    throw Exception("SHA256 mismatch for TTS $languageCode")
                }
            }
        }

        extractDir.mkdirs()
        extractZip(zipFile, extractDir)
        zipFile.delete()
        Log.d(TAG, "TTS model for $languageCode extracted to $extractDir")
    }

    private fun downloadFile(
        url: String,
        dest: File,
        modelName: String,
        progress: ProgressCallback
    ) {
        Log.d(TAG, "Downloading $modelName from $url")
        val request = Request.Builder().url(url).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw Exception("Download failed for $modelName: HTTP ${response.code}")
        }

        val body = response.body ?: throw Exception("Empty response body for $modelName")
        val totalBytes = body.contentLength()

        body.byteStream().use { input ->
            FileOutputStream(dest).use { output ->
                val buffer = ByteArray(8192)
                var downloadedBytes = 0L
                var bytesRead: Int

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    downloadedBytes += bytesRead

                    if (totalBytes > 0) {
                        val percent = ((downloadedBytes * 100) / totalBytes).toInt()
                        progress(modelName, percent, downloadedBytes, totalBytes)
                    } else {
                        progress(modelName, -1, downloadedBytes, -1L)
                    }
                }
            }
        }
        progress(modelName, 100, dest.length(), dest.length())
    }

    private fun extractZip(zipFile: File, destDir: File) {
        ZipInputStream(zipFile.inputStream().buffered()).use { zis ->
            var entry = zis.nextEntry
            while (entry != null) {
                if (!entry.isDirectory) {
                    val outFile = File(destDir, entry.name)
                    outFile.parentFile?.mkdirs()
                    FileOutputStream(outFile).use { out ->
                        zis.copyTo(out)
                    }
                } else {
                    File(destDir, entry.name).mkdirs()
                }
                zis.closeEntry()
                entry = zis.nextEntry
            }
        }
    }

    private fun sha256Hex(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (input.read(buffer).also { bytesRead = it } != -1) {
                digest.update(buffer, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
