package com.indicai.engine

import org.apache.commons.math3.transform.DftNormalization
import org.apache.commons.math3.transform.FastFourierTransformer
import org.apache.commons.math3.transform.TransformType
import kotlin.math.*

/**
 * Computes Whisper-compatible log-mel spectrogram.
 *
 * Whisper parameters:
 *   n_fft         = 400 (window size; zero-pad to 512 for FFT)
 *   fft_size      = 512 (actual FFT size used)
 *   n_freq_bins   = 201 (n_fft/2 + 1 = 400/2 + 1)
 *   hop_length    = 160
 *   n_mels        = 80
 *   sample_rate   = 16000
 *   audio_len     = 480000 (30 s)
 *   n_frames      = 3000
 */
class MelSpectrogram {
    companion object {
        private const val N_FFT = 400
        private const val FFT_SIZE = 512
        private const val N_FREQ_BINS = 201   // N_FFT/2 + 1
        private const val HOP_LENGTH = 160
        private const val N_MELS = 80
        private const val SAMPLE_RATE = 16000
        private const val N_FRAMES = 3000
        private const val AUDIO_LEN = 480000  // 30 * 16000
        private const val F_MIN = 0.0
        private const val F_MAX = 8000.0      // SAMPLE_RATE / 2
    }

    private val fft = FastFourierTransformer(DftNormalization.STANDARD)
    private val hannWindow: FloatArray = buildHannWindow()
    private val melFilterbank: Array<FloatArray> = buildMelFilterbank()

    private fun buildHannWindow(): FloatArray {
        return FloatArray(N_FFT) { i ->
            (0.5 * (1.0 - cos(2.0 * PI * i / N_FFT))).toFloat()
        }
    }

    /** Convert Hz to HTK mel scale */
    private fun hzToMel(hz: Double): Double = 2595.0 * log10(1.0 + hz / 700.0)

    /** Convert HTK mel to Hz */
    private fun melToHz(mel: Double): Double = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

    private fun buildMelFilterbank(): Array<FloatArray> {
        val melMin = hzToMel(F_MIN)
        val melMax = hzToMel(F_MAX)

        // n_mels + 2 evenly-spaced points in mel scale
        val melPoints = DoubleArray(N_MELS + 2) { i ->
            melToHz(melMin + i * (melMax - melMin) / (N_MELS + 1))
        }

        // Frequency of each FFT bin (only first 201 bins used)
        val freqBins = DoubleArray(N_FREQ_BINS) { i ->
            i.toDouble() * SAMPLE_RATE / FFT_SIZE
        }

        // Build filterbank matrix [n_mels x N_FREQ_BINS]
        return Array(N_MELS) { m ->
            FloatArray(N_FREQ_BINS) { f ->
                val freq = freqBins[f]
                val lower = melPoints[m]
                val center = melPoints[m + 1]
                val upper = melPoints[m + 2]
                when {
                    freq < lower || freq > upper -> 0.0f
                    freq <= center -> ((freq - lower) / (center - lower)).toFloat()
                    else -> ((upper - freq) / (upper - center)).toFloat()
                }
            }
        }
    }

    /**
     * Compute log-mel spectrogram from raw PCM float samples.
     * @param samples Float array of PCM at 16kHz (padded/truncated to 480000)
     * @return FloatArray of shape [80 * 3000] (n_mels * n_frames)
     */
    fun compute(samples: FloatArray): FloatArray {
        // Pad or truncate to exactly AUDIO_LEN
        val audio = if (samples.size >= AUDIO_LEN) {
            samples.copyOf(AUDIO_LEN)
        } else {
            FloatArray(AUDIO_LEN).also {
                samples.copyInto(it)
            }
        }

        // Reflected padding: pad by N_FFT/2 on each side
        val padSize = N_FFT / 2
        val padded = FloatArray(audio.size + 2 * padSize)
        // Left reflection pad
        for (i in 0 until padSize) {
            padded[padSize - 1 - i] = audio[i + 1]
        }
        // Copy original
        audio.copyInto(padded, destinationOffset = padSize)
        // Right reflection pad
        for (i in 0 until padSize) {
            val srcIdx = audio.size - 2 - i
            if (srcIdx >= 0) {
                padded[padSize + audio.size + i] = audio[srcIdx]
            }
        }

        // Output: [N_MELS, N_FRAMES]
        val melSpec = Array(N_MELS) { FloatArray(N_FRAMES) }

        // Zero-padded frame buffer for 512-pt FFT
        val fftInput = DoubleArray(FFT_SIZE)
        val fftInputImag = DoubleArray(FFT_SIZE)

        for (frameIdx in 0 until N_FRAMES) {
            val start = frameIdx * HOP_LENGTH

            // Apply Hann window and fill first N_FFT elements, zero-pad rest
            for (i in 0 until FFT_SIZE) {
                if (i < N_FFT) {
                    fftInput[i] = padded[start + i].toDouble() * hannWindow[i]
                } else {
                    fftInput[i] = 0.0
                }
                fftInputImag[i] = 0.0
            }

            // Compute 512-pt FFT using Apache Commons Math
            val complexResult = fft.transform(fftInput, TransformType.FORWARD)

            // Compute power spectrum: only first N_FREQ_BINS (201) bins
            val powerSpec = FloatArray(N_FREQ_BINS) { binIdx ->
                val re = complexResult[binIdx].real
                val im = complexResult[binIdx].imaginary
                (re * re + im * im).toFloat()
            }

            // Apply mel filterbank
            for (melIdx in 0 until N_MELS) {
                var melVal = 0.0f
                for (binIdx in 0 until N_FREQ_BINS) {
                    melVal += melFilterbank[melIdx][binIdx] * powerSpec[binIdx]
                }
                // Clamp to avoid log(0)
                melSpec[melIdx][frameIdx] = max(melVal, 1e-10f)
            }
        }

        // Log10 and normalization
        val logMel = FloatArray(N_MELS * N_FRAMES)
        var logMax = Float.NEGATIVE_INFINITY

        for (melIdx in 0 until N_MELS) {
            for (frameIdx in 0 until N_FRAMES) {
                val v = log10(melSpec[melIdx][frameIdx])
                logMel[melIdx * N_FRAMES + frameIdx] = v
                if (v > logMax) logMax = v
            }
        }

        // Clamp: max(log_mel, log_max - 8)
        val clampMin = logMax - 8.0f
        for (i in logMel.indices) {
            if (logMel[i] < clampMin) logMel[i] = clampMin
        }

        // Normalize: (log_mel + 4) / 8  →  Whisper uses (log_mel - max) / 4 + 1
        for (i in logMel.indices) {
            logMel[i] = (logMel[i] - logMax) / 4.0f + 1.0f
        }

        return logMel
    }
}
