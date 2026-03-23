import Foundation
import React
import AVFoundation

@objc(IndicAIModule)
class IndicAIModule: RCTEventEmitter {

    private let sdk = IndicAISDK.shared
    private var audioEngine: AVAudioEngine?
    private var recordedPCMData = Data()
    @objc private var isRecording = false

    override static func requiresMainQueueSetup() -> Bool { false }
    override func supportedEvents() -> [String]! { ["IndicAI_Progress"] }

    // Required for RCTEventEmitter
    @objc override func addListener(_ eventName: String) {}
    @objc override func removeListeners(_ count: Double) {}

    @objc func configure(_ manifestUrl: String?, s3BaseUrl: String?) {
        let cfg = SDKConfig(
            manifestURL: URL(string: manifestUrl ?? IndicAISDK.defaultManifestUrl)!,
            s3BaseURL:   URL(string: s3BaseUrl ?? IndicAISDK.defaultS3BaseUrl)!
        )
        Task { await sdk.configure(cfg) }
    }

    @objc func initialize(
        _ languageCode: String,
        resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        Task {
            do {
                try await sdk.initialize(
                    language: languageCode,
                    onProgress: { [weak self] p in
                        self?.sendEvent(withName: "IndicAI_Progress", body: [
                            "model":      p.modelName,
                            "percent":    p.percent,
                            "downloaded": p.downloaded,
                            "total":      p.total,
                        ])
                    }
                )
                resolver(nil)
            } catch {
                rejecter("INDICAI_INIT_ERROR", error.localizedDescription, error)
            }
        }
    }

    @objc func transcribe(
        _ base64Audio: String,
        resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        guard let pcmData = Data(base64Encoded: base64Audio) else {
            rejecter("INDICAI_TRANSCRIBE_ERROR", "Invalid base64 audio data", nil)
            return
        }
        do {
            resolver(try sdk.transcribe(pcmData: pcmData))
        } catch {
            rejecter("INDICAI_TRANSCRIBE_ERROR", error.localizedDescription, error)
        }
    }

    @objc func translateToEnglish(
        _ text: String,
        resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        do {
            resolver(try sdk.translateToEnglish(text))
        } catch {
            rejecter("INDICAI_TRANSLATE_ERROR", error.localizedDescription, error)
        }
    }

    @objc func translateToIndic(
        _ text: String,
        resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        do {
            resolver(try sdk.translateToIndic(text))
        } catch {
            rejecter("INDICAI_TRANSLATE_ERROR", error.localizedDescription, error)
        }
    }

    /// Returns the absolute path to a WAV file written in the tmp directory.
    @objc func synthesize(
        _ text: String,
        resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        Task {
            do {
                let waveform = try await sdk.synthesize(text)
                let path = NSTemporaryDirectory() + "indicai_tts.wav"
                try writeWav(samples: waveform, to: path)
                resolver(path)
            } catch {
                rejecter("INDICAI_TTS_ERROR", error.localizedDescription, error)
            }
        }
    }

    /// Returns { indicText, audioPath } — mirrors Android's respondWithSpeech.
    @objc func respondWithSpeech(
        _ englishText: String,
        resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        Task {
            do {
                let result = try await sdk.respondWithSpeech(englishText: englishText)
                let path = NSTemporaryDirectory() + "indicai_tts.wav"
                try writeWav(samples: result.audio, to: path)
                resolver(["indicText": result.indicText, "audioPath": path])
            } catch {
                rejecter("INDICAI_RESPOND_ERROR", error.localizedDescription, error)
            }
        }
    }

    /// Classify English text → intent label + confidence. Mirrors Android classifyIntent.
    @objc func classifyIntent(
        _ text: String,
        resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        do {
            guard let result = try sdk.classifyIntent(text) else {
                rejecter("INDICAI_INTENT_ERROR", "Intent model not loaded", nil)
                return
            }
            resolver(["intent": result.intent, "confidence": result.confidence] as [String: Any])
        } catch {
            rejecter("INDICAI_INTENT_ERROR", error.localizedDescription, error)
        }
    }

    @objc func getSupportedLanguages(
        _ resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        resolver(Array(LanguageRegistry.supported.keys))
    }

    @objc func isTtsCached(
        _ languageCode: String,
        resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        resolver(sdk.isTtsCached(for: languageCode))
    }

    @objc func release() {
        stopAndReleaseRecorder()
    }

    // ── Native microphone recording (raw PCM Int16 16 kHz mono → base64) ────────

    @objc func startRecording(
        _ resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        guard !isRecording else { resolver(nil); return }

        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)

            let engine = AVAudioEngine()
            audioEngine = engine
            let inputNode = engine.inputNode
            // Request 16 kHz mono Int16 — matches what Whisper expects
            let fmt = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: 16000,
                channels: 1,
                interleaved: true
            )!

            recordedPCMData = Data()
            isRecording = true

            inputNode.installTap(onBus: 0, bufferSize: 1600, format: fmt) { [weak self] buf, _ in
                guard let self = self, self.isRecording else { return }
                let ptr = buf.int16ChannelData![0]
                let bytes = Data(bytes: ptr, count: Int(buf.frameLength) * 2)
                self.recordedPCMData.append(bytes)
            }

            try engine.start()
            resolver(nil)
        } catch {
            rejecter("INDICAI_RECORD_ERROR", error.localizedDescription, error)
        }
    }

    @objc func stopRecording(
        _ resolver: @escaping RCTPromiseResolveBlock,
        rejecter: @escaping RCTPromiseRejectBlock
    ) {
        guard isRecording else { resolver(""); return }
        stopAndReleaseRecorder()
        resolver(recordedPCMData.base64EncodedString())
        recordedPCMData = Data()
    }

    // ── Helpers ──────────────────────────────────────────────────────────────────

    private func stopAndReleaseRecorder() {
        isRecording = false
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
    }

    /// Write Float32 PCM samples as a 16-bit PCM WAV file.
    private func writeWav(samples: [Float], to path: String) throws {
        let sampleRate: Int32 = 16000
        let dataLen = Int32(samples.count * 2)
        var buf = Data()

        func appendLE<T: FixedWidthInteger>(_ v: T) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { buf.append(contentsOf: $0) }
        }

        buf.append(contentsOf: "RIFF".utf8); appendLE(36 + dataLen)
        buf.append(contentsOf: "WAVE".utf8)
        buf.append(contentsOf: "fmt ".utf8); appendLE(Int32(16))
        appendLE(Int16(1));  appendLE(Int16(1))   // PCM, mono
        appendLE(sampleRate); appendLE(sampleRate * 2)
        appendLE(Int16(2));  appendLE(Int16(16))   // block align, bits
        buf.append(contentsOf: "data".utf8); appendLE(dataLen)

        for s in samples {
            appendLE(Int16(max(-32768, min(32767, Int(s * 32767)))))
        }

        try buf.write(to: URL(fileURLWithPath: path))
    }
}
