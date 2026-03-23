import Foundation
import AVFoundation

public enum IndicAIError: Error, LocalizedError {
    case unsupportedLanguage(String)
    case downloadFailed(String)
    case modelCorrupted(String)
    case notInitialized
    case networkUnavailable
    case inferenceError(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedLanguage(let l): return "Language '\(l)' not supported"
        case .downloadFailed(let m):      return "Download failed: \(m)"
        case .modelCorrupted(let m):      return "SHA-256 mismatch: \(m)"
        case .notInitialized:             return "Call initialize() first"
        case .networkUnavailable:         return "No network, no cached manifest"
        case .inferenceError(let s):      return "Inference error: \(s)"
        }
    }
}

public struct DownloadProgress {
    public let modelName: String
    public let downloaded: Int64
    public let total: Int64
    public var percent: Int { total > 0 ? Int(downloaded * 100 / total) : 0 }
}

public struct SDKConfig {
    public let manifestURL: URL
    public let s3BaseURL: URL
}

@MainActor
public final class IndicAISDK {

    public static let shared = IndicAISDK()
    public static let defaultManifestUrl = "https://indicai-cdn.s3.ap-south-1.amazonaws.com/manifest.json"
    public static let defaultS3BaseUrl   = "https://indicai-cdn.s3.ap-south-1.amazonaws.com/models"

    private var config = SDKConfig(
        manifestURL: URL(string: defaultManifestUrl)!,
        s3BaseURL:   URL(string: defaultS3BaseUrl)!
    )
    private var manifestManager: ManifestManager?
    private var modelManager: ModelManager?
    private var engine: InferenceEngine?
    private var intentEngine = IntentEngine()

    private var currentLang: LanguageConfig?
    private var manifest: SDKManifest?
    private var ready = false

    private init() {}

    public func configure(_ cfg: SDKConfig) {
        config = cfg
        ready = false
    }

    public func initialize(
        language: String,
        onProgress: ((DownloadProgress) -> Void)? = nil
    ) async throws {
        guard LanguageRegistry.isSupported(language) else {
            throw IndicAIError.unsupportedLanguage(language)
        }
        currentLang = try LanguageRegistry.config(for: language)
        manifestManager = ManifestManager(manifestURL: config.manifestURL)
        modelManager = ModelManager(s3BaseURL: config.s3BaseURL)
        engine = InferenceEngine()

        engine!.initTokenizers()
        engine!.loadTtsVocab(langCode: language)

        manifest = try await manifestManager!.fetch()
        let m = manifest!

        // ── Whisper ────────────────────────────────────────────────────────────
        let whisperURL = try await modelManager!.ensureModel(m.models.whisper) { dl, total in
            onProgress?(DownloadProgress(modelName: "Whisper", downloaded: dl, total: total))
        }
        try engine!.loadWhisper(from: whisperURL)

        // ── IndicTrans2 Indic→EN ───────────────────────────────────────────────
        let indicURL = try await modelManager!.ensureModel(m.models.indicEn) { dl, total in
            onProgress?(DownloadProgress(modelName: "IndicTrans2 Indic→EN", downloaded: dl, total: total))
        }
        try engine!.loadIndicEn(from: indicURL)

        // ── IndicTrans2 EN→Indic ───────────────────────────────────────────────
        let enIndicURL = try await modelManager!.ensureModel(m.models.indicFromEn) { dl, total in
            onProgress?(DownloadProgress(modelName: "IndicTrans2 EN→Indic", downloaded: dl, total: total))
        }
        try engine!.loadEnIndic(from: enIndicURL)

        // ── TTS (prefetch selected language) ──────────────────────────────────
        if let lang = currentLang, let ttsEntry = m.languages[lang.code]?.tts {
            let _ = try await modelManager!.ensureModel(ttsEntry) { dl, total in
                onProgress?(DownloadProgress(modelName: "TTS-\(lang.code)", downloaded: dl, total: total))
            }
        }

        // ── Intent MiniLM (optional — present when manifest has intent model) ──
        if let intentEntry = m.models.intent {
            let intentOnnx = try await modelManager!.ensureModel(intentEntry) { dl, total in
                onProgress?(DownloadProgress(modelName: "Intent", downloaded: dl, total: total))
            }
            let intentDir = intentOnnx.deletingLastPathComponent()
            try intentEngine.load(
                modelFile:      intentOnnx,
                vocabFile:      intentDir.appendingPathComponent("tokenizer/vocab.txt"),
                bankNpyFile:    intentDir.appendingPathComponent("intent_bank.npy"),
                bankLabelsFile: intentDir.appendingPathComponent("intent_bank_labels.json")
            )
        }

        ready = true
    }

    public func transcribe(pcmData: Data) throws -> String {
        try checkReady()
        let floats = pcmInt16ToFloat(pcmData)
        return try engine!.transcribe(audio: floats, lang: currentLang!)
    }

    public func translateToEnglish(_ text: String) throws -> String {
        try checkReady()
        return try engine!.translateToEn(text: text, lang: currentLang!)
    }

    public func translateToIndic(_ text: String) throws -> String {
        try checkReady()
        return try engine!.translateToIndic(text: text, lang: currentLang!)
    }

    public func synthesize(
        _ text: String,
        onProgress: ((DownloadProgress) -> Void)? = nil
    ) async throws -> [Float] {
        try checkReady()
        let lang = currentLang!
        guard let ttsEntry = manifest?.languages[lang.code]?.tts else {
            throw IndicAIError.unsupportedLanguage("\(lang.code) has no TTS")
        }
        let ttsURL = try await modelManager!.ensureModel(ttsEntry) { dl, total in
            onProgress?(DownloadProgress(modelName: "TTS-\(lang.code)", downloaded: dl, total: total))
        }
        try engine!.loadTts(from: ttsURL)
        engine!.loadTtsVocab(langCode: lang.code)
        return try engine!.synthesize(text: text, langCode: lang.code)
    }

    public func respondWithSpeech(
        englishText: String,
        onProgress: ((DownloadProgress) -> Void)? = nil
    ) async throws -> (indicText: String, audio: [Float]) {
        try checkReady()
        let lang = currentLang!

        let indicText = try engine!.translateToIndic(text: englishText, lang: lang)

        guard let ttsEntry = manifest?.languages[lang.code]?.tts else {
            throw IndicAIError.unsupportedLanguage("\(lang.code) has no TTS")
        }
        let ttsURL = try await modelManager!.ensureModel(ttsEntry) { dl, total in
            onProgress?(DownloadProgress(modelName: "TTS-\(lang.code)", downloaded: dl, total: total))
        }
        try engine!.loadTts(from: ttsURL)
        engine!.loadTtsVocab(langCode: lang.code)
        let audio = try engine!.synthesize(text: indicText, langCode: lang.code)

        return (indicText, audio)
    }

    /// Classify English text to an intent label using on-device MiniLM semantic similarity.
    /// Returns nil if intent model is not loaded (not present in manifest).
    public func classifyIntent(_ text: String) throws -> IntentResult? {
        guard intentEngine.isLoaded() else { return nil }
        return try intentEngine.classify(text)
    }

    public var supportedLanguages: [String] { Array(LanguageRegistry.supported.keys) }
    public func isSupported(_ code: String) -> Bool { LanguageRegistry.isSupported(code) }
    public func isTtsCached(for code: String) -> Bool {
        guard let entry = manifest?.languages[code]?.tts else { return false }
        return (try? modelManager?.isModelCached(entry)) == true
    }

    private func checkReady() throws {
        if !ready { throw IndicAIError.notInitialized }
    }

    private func pcmInt16ToFloat(_ data: Data) -> [Float] {
        var out = [Float](repeating: 0, count: data.count / 2)
        for i in 0..<out.count {
            let lo = UInt16(data[i * 2])
            let hi = UInt16(data[i * 2 + 1])
            out[i] = Float(Int16(bitPattern: lo | (hi << 8))) / 32768.0
        }
        return out
    }
}
