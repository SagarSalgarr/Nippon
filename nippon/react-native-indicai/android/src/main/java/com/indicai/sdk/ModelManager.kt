// ModelManager.kt — downloads, sha256-verifies, and caches .ort model files

package com.indicai.sdk

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.zip.ZipInputStream

private const val TAG = "IndicAI.Models"
private const val MAX_DOWNLOAD_RETRIES = 3

typealias ProgressCallback = (downloaded: Long, total: Long) -> Unit

internal class ModelManager(
    context:               Context,
    private val s3BaseUrl: String,
    private val http:      OkHttpClient,
) {
    private val modelDir = File(context.filesDir, "indicai_models").also { it.mkdirs() }
    private val extractedDir = File(modelDir, "_extracted").also { it.mkdirs() }

    suspend fun ensureModel(entry: ModelEntry, onProgress: ProgressCallback? = null): File =
        withContext(Dispatchers.IO) {
            val local = File(modelDir, entry.file)
            if (local.exists() && verify(local, entry.sha256)) {
                Log.d(TAG, "${entry.file} — cache hit")
                return@withContext resolveModelFile(local, entry)
            }
            if (local.exists()) {
                Log.w(TAG, "Hash mismatch — re-downloading ${entry.file}")
                local.delete()
            }

            var lastError: Exception? = null
            repeat(MAX_DOWNLOAD_RETRIES) { idx ->
                val attempt = idx + 1
                try {
                    download("$s3BaseUrl/${entry.file}", local, onProgress)
                    if (!verify(local, entry.sha256)) {
                        local.delete()
                        throw SecurityException("SHA-256 mismatch for ${entry.file}")
                    }
                    Log.d(TAG, "${entry.file} downloaded OK (attempt $attempt)")
                    return@withContext resolveModelFile(local, entry)
                } catch (e: Exception) {
                    lastError = e
                    local.delete()
                    if (attempt < MAX_DOWNLOAD_RETRIES) {
                        Log.w(TAG, "Download failed for ${entry.file}, retry $attempt/$MAX_DOWNLOAD_RETRIES", e)
                        Thread.sleep(1000L * attempt)
                    }
                }
            }
            throw RuntimeException(
                "Failed to download ${entry.file} after $MAX_DOWNLOAD_RETRIES attempts",
                lastError
            )
        }

    fun isModelCached(entry: ModelEntry) =
        File(modelDir, entry.file).let { it.exists() && verify(it, entry.sha256) }

    private fun resolveModelFile(archiveOrModel: File, entry: ModelEntry): File {
        if (!archiveOrModel.name.endsWith(".zip")) return archiveOrModel

        val extractRoot = File(extractedDir, archiveOrModel.name.removeSuffix(".zip"))
        val marker = File(extractRoot, ".ready")
        if (marker.exists()) {
            val extractedForSha = marker.readText().trim()
            val existingModel = findModelFile(extractRoot)
            if (existingModel != null && extractedForSha == entry.sha256) return existingModel
            marker.delete()
        }

        if (extractRoot.exists()) extractRoot.deleteRecursively()
        extractRoot.mkdirs()
        unzip(archiveOrModel, extractRoot)

        val modelFile = findModelFile(extractRoot)
            ?: throw RuntimeException("No .onnx/.ort model found inside ${entry.file}")
        marker.writeText(entry.sha256)
        Log.d(TAG, "Extracted ${entry.file} -> ${modelFile.absolutePath}")
        return modelFile
    }

    private fun findModelFile(root: File): File? {
        return root.walkTopDown()
            .firstOrNull { it.isFile && (it.name.endsWith(".onnx") || it.name.endsWith(".ort")) }
    }

    private fun unzip(zipFile: File, outputDir: File) {
        ZipInputStream(zipFile.inputStream().buffered()).use { zis ->
            var entry = zis.nextEntry
            while (entry != null) {
                val outFile = File(outputDir, entry.name).canonicalFile
                val outRoot = outputDir.canonicalFile
                if (!outFile.path.startsWith(outRoot.path + File.separator)) {
                    throw SecurityException("Invalid zip path: ${entry.name}")
                }
                if (entry.isDirectory) {
                    outFile.mkdirs()
                } else {
                    outFile.parentFile?.mkdirs()
                    FileOutputStream(outFile).use { fos ->
                        zis.copyTo(fos)
                    }
                }
                zis.closeEntry()
                entry = zis.nextEntry
            }
        }
    }

    private fun download(url: String, dest: File, onProgress: ProgressCallback?) {
        val resp = http.newCall(Request.Builder().url(url).build()).execute()
        resp.use {
            if (!it.isSuccessful) throw RuntimeException("HTTP ${it.code} for $url")
            val body = it.body ?: throw RuntimeException("Empty body for $url")
            val total = body.contentLength()
            var dl = 0L
            val tmp = File(dest.parent, "${dest.name}.tmp")
            try {
                tmp.outputStream().use { out ->
                    body.byteStream().use { inp ->
                        val buf = ByteArray(8192); var n: Int
                        while (inp.read(buf).also { n = it } != -1) {
                            out.write(buf, 0, n); dl += n; onProgress?.invoke(dl, total)
                        }
                    }
                }
                tmp.renameTo(dest)
            } catch (e: Exception) {
                tmp.delete()
                throw e
            }
        }
    }

    private fun verify(file: File, expected: String): Boolean {
        val d = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { s ->
            val b = ByteArray(65536); var n: Int
            while (s.read(b).also { n = it } != -1) d.update(b, 0, n)
        }
        return d.digest().joinToString("") { "%02x".format(it) } == expected
    }
}
