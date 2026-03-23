import Foundation
import CryptoKit
import onnxruntime_objc
import Accelerate

// ── Language registry ─────────────────────────────────────────────────────────

public struct LanguageConfig {
    public let code: String; public let name: String
    public let whisperLang: String; public let flores: String
    public let sttPrompt: String; public let hasTts: Bool
}

public enum LanguageRegistry {
    public static let supported: [String: LanguageConfig] = [
        "hi": .init(code:"hi", name:"Hindi",     whisperLang:"hi", flores:"hin_Deva", sttPrompt:"हिंदी में लिखें।",         hasTts:true),
        "mr": .init(code:"mr", name:"Marathi",   whisperLang:"mr", flores:"mar_Deva", sttPrompt:"मराठी मध्ये लिहा।",         hasTts:true),
        "ta": .init(code:"ta", name:"Tamil",     whisperLang:"ta", flores:"tam_Taml", sttPrompt:"தமிழில் எழுதுக।",           hasTts:true),
        "te": .init(code:"te", name:"Telugu",    whisperLang:"te", flores:"tel_Telu", sttPrompt:"తెలుగులో రాయండి।",          hasTts:true),
        "kn": .init(code:"kn", name:"Kannada",   whisperLang:"kn", flores:"kan_Knda", sttPrompt:"ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ।",       hasTts:true),
        "ml": .init(code:"ml", name:"Malayalam", whisperLang:"ml", flores:"mal_Mlym", sttPrompt:"മലയാളത്തിൽ എഴുതുക।",         hasTts:true),
        "bn": .init(code:"bn", name:"Bengali",   whisperLang:"bn", flores:"ben_Beng", sttPrompt:"বাংলায় লিখুন।",             hasTts:true),
        "gu": .init(code:"gu", name:"Gujarati",  whisperLang:"gu", flores:"guj_Gujr", sttPrompt:"ગુજરાતીમાં લખો।",           hasTts:true),
        "pa": .init(code:"pa", name:"Punjabi",   whisperLang:"pa", flores:"pan_Guru", sttPrompt:"ਪੰਜਾਬੀ ਵਿੱਚ ਲਿਖੋ।",         hasTts:true),
        "ur": .init(code:"ur", name:"Urdu",      whisperLang:"ur", flores:"urd_Arab", sttPrompt:"اردو میں لکھیں۔",            hasTts:true),
    ]
    public static func config(for code: String) throws -> LanguageConfig {
        guard let c = supported[code] else { throw IndicAIError.unsupportedLanguage(code) }
        return c
    }
    public static func isSupported(_ code: String) -> Bool { supported[code] != nil }
}

// ── Manifest types ────────────────────────────────────────────────────────────

struct ModelEntry: Codable {
    let file: String; let sha256: String; let sizeMb: Double; let version: Int
    enum CodingKeys: String, CodingKey { case file, sha256, version; case sizeMb = "size_mb" }
}
struct LanguageManifestEntry: Codable {
    let name: String; let whisperLang: String; let flores: String; let tts: ModelEntry?
    enum CodingKeys: String, CodingKey { case name, flores, tts; case whisperLang = "whisper_lang" }
}
struct SDKModels: Codable {
    let whisper: ModelEntry; let indicEn: ModelEntry; let indicFromEn: ModelEntry
    let intent: ModelEntry?
    enum CodingKeys: String, CodingKey {
        case whisper
        case indicEn = "indic_en"
        case indicFromEn = "indic_from_en"
        case intent
    }
}
struct SDKManifest: Codable {
    let version: Int; let updatedAt: String; let models: SDKModels
    let languages: [String: LanguageManifestEntry]
    enum CodingKeys: String, CodingKey { case version, models, languages; case updatedAt="updated_at" }
}

// ── ManifestManager ───────────────────────────────────────────────────────────

actor ManifestManager {
    private let manifestURL: URL
    private let cacheFile: URL
    private let tsFile: URL
    private let ttl: TimeInterval = 6 * 3600

    init(manifestURL: URL) {
        self.manifestURL = manifestURL
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("IndicAI", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        cacheFile = dir.appendingPathComponent("manifest.json")
        tsFile = dir.appendingPathComponent("manifest.ts")
    }

    func fetch() async throws -> SDKManifest {
        let ts = Double(String(data: (try? Data(contentsOf: tsFile)) ?? Data(), encoding: .utf8) ?? "") ?? 0
        let age = Date().timeIntervalSince1970 - ts
        if age < ttl, let data = try? Data(contentsOf: cacheFile) {
            return try JSONDecoder().decode(SDKManifest.self, from: data)
        }
        do {
            let (data, _) = try await URLSession.shared.data(from: manifestURL)
            try data.write(to: cacheFile)
            try "\(Date().timeIntervalSince1970)".data(using: .utf8)!.write(to: tsFile)
            return try JSONDecoder().decode(SDKManifest.self, from: data)
        } catch {
            if let data = try? Data(contentsOf: cacheFile) {
                return try JSONDecoder().decode(SDKManifest.self, from: data)
            }
            throw IndicAIError.networkUnavailable
        }
    }
}

// ── ModelManager ─────────────────────────────────────────────────────────────

actor ModelManager {
    private let s3BaseURL: URL
    private let modelDir: URL
    private let extractedDir: URL

    init(s3BaseURL: URL) {
        self.s3BaseURL = s3BaseURL
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = appSupport.appendingPathComponent("IndicAI/models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.modelDir = dir
        let ext = dir.appendingPathComponent("_extracted", isDirectory: true)
        try? FileManager.default.createDirectory(at: ext, withIntermediateDirectories: true)
        self.extractedDir = ext
    }

    func ensureModel(_ entry: ModelEntry, onProgress: ((Int64, Int64) -> Void)? = nil) async throws -> URL {
        let local = modelDir.appendingPathComponent(entry.file)

        // Cache hit: zip present + hash valid
        if FileManager.default.fileExists(atPath: local.path),
           (try? verifyHash(at: local, expected: entry.sha256)) == true {
            return try resolveModelFile(local, entry: entry)
        }

        // Remove stale file and re-download
        try? FileManager.default.removeItem(at: local)
        let remote = s3BaseURL.appendingPathComponent(entry.file)
        try await streamDownload(from: remote, to: local, onProgress: onProgress)

        guard (try? verifyHash(at: local, expected: entry.sha256)) == true else {
            try? FileManager.default.removeItem(at: local)
            throw IndicAIError.modelCorrupted(entry.file)
        }
        return try resolveModelFile(local, entry: entry)
    }

    func isModelCached(_ entry: ModelEntry) throws -> Bool {
        let local = modelDir.appendingPathComponent(entry.file)
        return FileManager.default.fileExists(atPath: local.path) &&
               ((try? verifyHash(at: local, expected: entry.sha256)) == true)
    }

    // ── Zip resolution ────────────────────────────────────────────────────────

    private func resolveModelFile(_ archive: URL, entry: ModelEntry) throws -> URL {
        guard archive.pathExtension == "zip" else { return archive }

        let extractRoot = extractedDir.appendingPathComponent(
            archive.deletingPathExtension().lastPathComponent, isDirectory: true)
        let marker = extractRoot.appendingPathComponent(".ready")

        // Already extracted for this sha?
        if let stored = try? String(contentsOf: marker, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           stored == entry.sha256,
           let modelFile = findModelFile(in: extractRoot) {
            return modelFile
        }

        // Remove stale extract and re-extract
        if FileManager.default.fileExists(atPath: extractRoot.path) {
            try? FileManager.default.removeItem(at: extractRoot)
        }
        try FileManager.default.createDirectory(at: extractRoot, withIntermediateDirectories: true)
        try unzip(archive, to: extractRoot)

        guard let modelFile = findModelFile(in: extractRoot) else {
            throw IndicAIError.downloadFailed("No .onnx/.ort model found in \(entry.file)")
        }
        try entry.sha256.write(to: marker, atomically: true, encoding: .utf8)
        NSLog("[IndicAI.ModelManager] Extracted %@ → %@", entry.file, modelFile.path)
        return modelFile
    }

    private func findModelFile(in dir: URL) -> URL? {
        guard let enumerator = FileManager.default.enumerator(
            at: dir,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }
        for case let url as URL in enumerator {
            let ext = url.pathExtension
            if ext == "onnx" || ext == "ort" { return url }
        }
        return nil
    }

    // ── Streaming download (writes to temp file — no RAM spike) ───────────────

    private func streamDownload(from remote: URL, to local: URL, onProgress: ((Int64, Int64) -> Void)?) async throws {
        NSLog("[IndicAI.ModelManager] Downloading %@", remote.lastPathComponent)
        let (tmpURL, response) = try await URLSession.shared.download(from: remote)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            try? FileManager.default.removeItem(at: tmpURL)
            throw IndicAIError.downloadFailed(remote.lastPathComponent)
        }
        try? FileManager.default.removeItem(at: local)
        try FileManager.default.moveItem(at: tmpURL, to: local)
        let size = (try? local.resourceValues(forKeys: [.fileSizeKey]))?.fileSize.map { Int64($0) } ?? 0
        onProgress?(size, size)
        NSLog("[IndicAI.ModelManager] Downloaded %@ (%lld bytes)", remote.lastPathComponent, size)
    }

    // ── SHA-256 verify ────────────────────────────────────────────────────────

    private func verifyHash(at url: URL, expected: String) throws -> Bool {
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined() == expected
    }

    // ── ZIP extractor (libz, no external dependency) ──────────────────────────

    private func unzip(_ zipURL: URL, to destURL: URL) throws {
        guard let fh = FileHandle(forReadingAtPath: zipURL.path) else {
            throw IndicAIError.downloadFailed("Cannot open \(zipURL.lastPathComponent)")
        }
        defer { fh.closeFile() }

        while true {
            // Read 30-byte local file header
            let hdrData = fh.readData(ofLength: 30)
            guard hdrData.count == 30 else { break }
            let hdr = [UInt8](hdrData)

            // PK♥♦ signature
            guard hdr[0] == 0x50, hdr[1] == 0x4B, hdr[2] == 0x03, hdr[3] == 0x04 else { break }

            let compressionMethod = UInt16(hdr[8]) | UInt16(hdr[9]) << 8
            let compressedSize    = Int(UInt32(hdr[18]) | UInt32(hdr[19]) << 8 | UInt32(hdr[20]) << 16 | UInt32(hdr[21]) << 24)
            let uncompressedSize  = Int(UInt32(hdr[22]) | UInt32(hdr[23]) << 8 | UInt32(hdr[24]) << 16 | UInt32(hdr[25]) << 24)
            let filenameLen       = Int(UInt16(hdr[26]) | UInt16(hdr[27]) << 8)
            let extraLen          = Int(UInt16(hdr[28]) | UInt16(hdr[29]) << 8)

            let fnData = fh.readData(ofLength: filenameLen)
            guard fnData.count == filenameLen else { break }
            let filename = String(bytes: [UInt8](fnData), encoding: .utf8) ?? ""

            // Skip extra field
            if extraLen > 0 { fh.seek(toFileOffset: fh.offsetInFile + UInt64(extraLen)) }

            // Skip directory entries
            if filename.isEmpty || filename.hasSuffix("/") {
                if compressedSize > 0 { fh.seek(toFileOffset: fh.offsetInFile + UInt64(compressedSize)) }
                continue
            }

            // Sanitize path (prevent traversal)
            let parts = filename.components(separatedBy: "/").filter { !$0.isEmpty && $0 != ".." }
            guard !parts.isEmpty else {
                if compressedSize > 0 { fh.seek(toFileOffset: fh.offsetInFile + UInt64(compressedSize)) }
                continue
            }
            let safePath = parts.joined(separator: "/")
            let outURL = destURL.appendingPathComponent(safePath)

            try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)

            let compData = fh.readData(ofLength: compressedSize)
            guard compData.count == compressedSize else { break }

            switch compressionMethod {
            case 0: // STORE — raw bytes
                try compData.write(to: outURL)
            case 8: // DEFLATE
                let decompressed = try inflateDeflate([UInt8](compData), uncompressedSize: uncompressedSize)
                try decompressed.write(to: outURL)
            default:
                NSLog("[IndicAI.ModelManager] Skipping unsupported compression method %d for %@",
                      compressionMethod, filename)
            }
        }
    }

    /// Raw DEFLATE decompression using libz (wbits = -15).
    private func inflateDeflate(_ compressed: [UInt8], uncompressedSize: Int) throws -> Data {
        var outData = Data(count: max(uncompressedSize, 1))
        var inBuf   = compressed

        let ret: Int32 = inBuf.withUnsafeMutableBytes { inPtr in
            outData.withUnsafeMutableBytes { outPtr in
                var strm = z_stream()
                strm.next_in  = inPtr.baseAddress?.assumingMemoryBound(to: Bytef.self)
                strm.avail_in = uInt(inBuf.count)
                strm.next_out = outPtr.baseAddress?.assumingMemoryBound(to: Bytef.self)
                strm.avail_out = uInt(outData.count)

                guard inflateInit2_(&strm, -15, ZLIB_VERSION,
                                    Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
                    return Z_STREAM_ERROR
                }
                let r = inflate(&strm, Z_FINISH)
                inflateEnd(&strm)
                return r
            }
        }

        guard ret == Z_STREAM_END || ret == Z_OK else {
            throw IndicAIError.downloadFailed("ZIP inflate failed: \(ret)")
        }
        return outData
    }
}

// ── InferenceEngine ───────────────────────────────────────────────────────────

class InferenceEngine {
    // Whisper uses SEPARATE encoder + decoder sessions (mirrors Android)
    private var whisperEncSession: ORTSession?
    private var whisperDecSession: ORTSession?
    private var indicEnSession: ORTSession?
    private var enIndicSession: ORTSession?
    private var ttsSession: ORTSession?
    private let env = try! ORTEnv(loggingLevel: .warning)

    // Tokenizer data
    private var indicEnSrcVocab: [String: Int] = [:]
    private var indicEnTgtVocab: [Int: String] = [:]
    private var enIndicSrcVocab: [String: Int] = [:]
    private var enIndicTgtVocab: [Int: String] = [:]
    private var whisperVocab: [Int: String] = [:]
    private var whisperMerges: [(String, String)] = []
    private var ttsVocabs: [String: [String: Int]] = [:]
    private var melFilters: [Float] = []

    // Read from bundled config.json
    private var indicEnDecoderStartId: Int = 2
    private var indicEnEosTokenId: Int = 2
    private var enIndicDecoderStartId: Int = 2
    private var enIndicEosTokenId: Int = 2

    private func opts(_ threads: Int32) throws -> ORTSessionOptions {
        let o = try ORTSessionOptions()
        try o.setIntraOpNumThreads(threads)
        try o.setGraphOptimizationLevel(.all)
        return o
    }

    func initTokenizers() {
        let bundle = Bundle(for: type(of: self))

        indicEnSrcVocab = loadVocabJson(bundle: bundle, dir: "indic-en", file: "dict.SRC")
        indicEnTgtVocab = loadReverseVocabJson(bundle: bundle, dir: "indic-en", file: "dict.TGT")
        enIndicSrcVocab = loadVocabJson(bundle: bundle, dir: "en-indic", file: "dict.SRC")
        enIndicTgtVocab = loadReverseVocabJson(bundle: bundle, dir: "en-indic", file: "dict.TGT")
        whisperVocab = loadReverseVocabJson(bundle: bundle, dir: "whisper", file: "vocab")
        whisperMerges = loadMerges(bundle: bundle)
        melFilters = computeMelFilterbank()

        let ieCfg = loadConfigJson(bundle: bundle, dir: "indic-en")
        indicEnDecoderStartId = ieCfg["decoder_start_token_id"] ?? 2
        indicEnEosTokenId = ieCfg["eos_token_id"] ?? 2

        let eiCfg = loadConfigJson(bundle: bundle, dir: "en-indic")
        enIndicDecoderStartId = eiCfg["decoder_start_token_id"] ?? 2
        enIndicEosTokenId = eiCfg["eos_token_id"] ?? 2
    }

    func loadTtsVocab(langCode: String) {
        if ttsVocabs[langCode] != nil { return }
        let bundle = Bundle(for: type(of: self))
        ttsVocabs[langCode] = loadVocabJson(bundle: bundle, dir: "tts", file: "vocab_\(langCode)")
    }

    /// Load Whisper encoder + decoder from the directory containing the first .onnx found.
    func loadWhisper(from url: URL) throws {
        if whisperEncSession != nil { return }  // already loaded
        let dir = url.deletingLastPathComponent()
        let encURL = dir.appendingPathComponent("encoder_model_quantized.onnx")
        let decURL = dir.appendingPathComponent("decoder_model_quantized.onnx")
        guard FileManager.default.fileExists(atPath: encURL.path) else {
            throw IndicAIError.inferenceError("Whisper encoder not found: \(encURL.path)")
        }
        guard FileManager.default.fileExists(atPath: decURL.path) else {
            throw IndicAIError.inferenceError("Whisper decoder not found: \(decURL.path)")
        }
        whisperEncSession = try ORTSession(env: env, modelPath: encURL.path, sessionOptions: try opts(4))
        whisperDecSession = try ORTSession(env: env, modelPath: decURL.path, sessionOptions: try opts(4))
        NSLog("[IndicAI.Inference] Whisper loaded: enc + dec")
    }

    func loadIndicEn(from url: URL) throws {
        indicEnSession = try ORTSession(env: env, modelPath: url.path, sessionOptions: try opts(4))
    }
    func loadEnIndic(from url: URL) throws {
        enIndicSession = try ORTSession(env: env, modelPath: url.path, sessionOptions: try opts(4))
    }
    func loadTts(from url: URL) throws {
        ttsSession = try ORTSession(env: env, modelPath: url.path, sessionOptions: try opts(2))
    }

    // ── Whisper STT (encoder + decoder, mirrors Android) ────────────────────────

    func transcribe(audio: [Float], lang: LanguageConfig) throws -> String {
        guard let encSession = whisperEncSession,
              let decSession = whisperDecSession else { throw IndicAIError.notInitialized }

        let mel = computeLogMel(audio)
        let melData = mel.withUnsafeBufferPointer { Data(buffer: $0) }
        let melTensor = try ORTValue(
            tensorData: NSMutableData(data: melData),
            elementType: .float,
            shape: [1, 80, 3000]
        )

        // Stage 1: Encoder → last_hidden_state (1, 1500, 768)
        let encOut = try encSession.run(
            withInputs: ["input_features": melTensor],
            outputNames: ["last_hidden_state"],
            runOptions: nil
        )
        let encHidden = encOut["last_hidden_state"]!

        // Stage 2: Decoder greedy loop
        let langId = InferenceEngine.langTokens[lang.whisperLang] ?? 50301
        var decoderIds: [Int64] = [50258, langId, 50359, 50363]  // forced prompt
        let eosId: Int64 = 50257
        let maxSteps = 224

        for _ in 0..<maxSteps {
            let decData = decoderIds.withUnsafeBufferPointer { Data(buffer: $0) }
            let decTensor = try ORTValue(
                tensorData: NSMutableData(data: decData),
                elementType: .int64,
                shape: [1, NSNumber(value: decoderIds.count)]
            )

            let out = try decSession.run(
                withInputs: ["input_ids": decTensor, "encoder_hidden_states": encHidden],
                outputNames: ["logits"],
                runOptions: nil
            )

            let logitsData = try out["logits"]!.tensorData() as Data
            let vocabSize = logitsData.count / (MemoryLayout<Float>.size * decoderIds.count)
            let lastOffset = (decoderIds.count - 1) * vocabSize

            let nextToken = logitsData.withUnsafeBytes { buf -> Int64 in
                let floats = buf.bindMemory(to: Float.self)
                var maxVal: Float = -Float.infinity
                var maxIdx = 0
                for i in 0..<vocabSize {
                    let v = floats[lastOffset + i]
                    if v > maxVal { maxVal = v; maxIdx = i }
                }
                return Int64(maxIdx)
            }

            if nextToken == eosId { break }
            decoderIds.append(nextToken)
        }

        // Drop forced prompt, decode tokens
        let tokenIds = Array(decoderIds.dropFirst(4)).map { Int($0) }.filter { $0 < 50257 }
        return whisperBpeDecode(tokenIds)
    }

    // ── IndicTrans2 MT ──────────────────────────────────────────────────────────

    func translateToEn(text: String, lang: LanguageConfig) throws -> String {
        guard let session = indicEnSession else { throw IndicAIError.notInitialized }
        let preprocessed = indicPreprocess(text: text, srcLang: lang.flores, tgtLang: "eng_Latn")
        let inputIds = indicTransTokenize(preprocessed, vocab: indicEnSrcVocab)
        return try indicTransInfer(session: session, inputIds: inputIds,
                                    decoderStartId: indicEnDecoderStartId, eosId: indicEnEosTokenId,
                                    tgtVocab: indicEnTgtVocab, tgtLang: "eng_Latn")
    }

    func translateToIndic(text: String, lang: LanguageConfig) throws -> String {
        guard let session = enIndicSession else { throw IndicAIError.notInitialized }
        let preprocessed = indicPreprocess(text: text, srcLang: "eng_Latn", tgtLang: lang.flores)
        let inputIds = indicTransTokenize(preprocessed, vocab: enIndicSrcVocab)
        return try indicTransInfer(session: session, inputIds: inputIds,
                                    decoderStartId: enIndicDecoderStartId, eosId: enIndicEosTokenId,
                                    tgtVocab: enIndicTgtVocab, tgtLang: lang.flores)
    }

    private func indicTransInfer(
        session: ORTSession,
        inputIds: [Int64],
        decoderStartId: Int,
        eosId: Int,
        tgtVocab: [Int: String],
        tgtLang: String
    ) throws -> String {
        var ids = inputIds
        var attn = [Int64](repeating: 1, count: ids.count)
        let shape: [NSNumber] = [1, NSNumber(value: ids.count)]

        let inTensor = try ORTValue(
            tensorData: NSMutableData(data: ids.withUnsafeBufferPointer { Data(buffer: $0) }),
            elementType: .int64, shape: shape
        )
        let atTensor = try ORTValue(
            tensorData: NSMutableData(data: attn.withUnsafeBufferPointer { Data(buffer: $0) }),
            elementType: .int64, shape: shape
        )

        let outIds = try greedyDecode(session: session, inputTensor: inTensor, attnTensor: atTensor,
                                       startId: decoderStartId, eosId: eosId, maxLen: 128)
        let rawText = indicTransDetokenize(outIds, vocab: tgtVocab)
        return indicPostprocess(rawText, lang: tgtLang)
    }

    private func greedyDecode(
        session: ORTSession,
        inputTensor: ORTValue,
        attnTensor: ORTValue,
        startId: Int,
        eosId: Int,
        maxLen: Int
    ) throws -> [Int] {
        var ids: [Int64] = [Int64(startId)]

        for _ in 0..<maxLen {
            let decData = ids.withUnsafeBufferPointer { Data(buffer: $0) }
            let decTensor = try ORTValue(
                tensorData: NSMutableData(data: decData),
                elementType: .int64,
                shape: [1, NSNumber(value: ids.count)]
            )

            let out = try session.run(
                withInputs: [
                    "input_ids": inputTensor,
                    "attention_mask": attnTensor,
                    "decoder_input_ids": decTensor
                ],
                outputNames: ["logits"],
                runOptions: nil
            )

            let logitsData = try out["logits"]!.tensorData() as Data
            let seqLen = ids.count
            let vocabSize = logitsData.count / (MemoryLayout<Float>.size * seqLen)
            let lastOffset = (seqLen - 1) * vocabSize

            let nextToken = logitsData.withUnsafeBytes { buf -> Int in
                let floats = buf.bindMemory(to: Float.self)
                var maxVal: Float = -Float.infinity
                var maxIdx = 0
                for i in 0..<vocabSize {
                    let v = floats[lastOffset + i]
                    if v > maxVal { maxVal = v; maxIdx = i }
                }
                return maxIdx
            }

            if nextToken == eosId { break }
            ids.append(Int64(nextToken))
        }

        return ids.map { Int($0) }
    }

    // ── MMS-TTS ─────────────────────────────────────────────────────────────────

    func synthesize(text: String, langCode: String) throws -> [Float] {
        guard let session = ttsSession else { throw IndicAIError.notInitialized }
        guard let vocab = ttsVocabs[langCode] else {
            throw IndicAIError.inferenceError("TTS vocab not loaded for \(langCode)")
        }

        let ids = ttsTokenize(text, vocab: vocab)
        guard !ids.isEmpty else {
            throw IndicAIError.inferenceError("TTS tokenization produced empty input")
        }

        var mutableIds = ids
        let tensor = try ORTValue(
            tensorData: NSMutableData(data: mutableIds.withUnsafeBufferPointer { Data(buffer: $0) }),
            elementType: .int64,
            shape: [1, NSNumber(value: ids.count)]
        )
        let out = try session.run(
            withInputs: ["input_ids": tensor],
            outputNames: ["waveform"],
            runOptions: nil
        )
        let data = try out["waveform"]!.tensorData() as Data
        let count = data.count / MemoryLayout<Float>.size
        var waveform = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }

        let maxAbs = waveform.map { abs($0) }.max() ?? 0
        if maxAbs > 0 {
            let scale = 0.9 / maxAbs
            for i in waveform.indices { waveform[i] *= scale }
        }
        return waveform
    }

    // ── IndicTrans2 Tokenizer ───────────────────────────────────────────────────

    private func indicTransTokenize(_ text: String, vocab: [String: Int]) -> [Int64] {
        let unkId = vocab["<unk>"] ?? 3
        let bosId = vocab["<s>"] ?? 0
        let eosId = vocab["</s>"] ?? 2

        let words = text.trimmingCharacters(in: .whitespaces).components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        var ids: [Int64] = [Int64(bosId)]

        for word in words {
            let subwords = bpeSegment(word, vocab: vocab)
            for sw in subwords {
                ids.append(Int64(vocab[sw] ?? unkId))
            }
        }

        ids.append(Int64(eosId))
        return ids
    }

    private func bpeSegment(_ word: String, vocab: [String: Int]) -> [String] {
        if word.isEmpty { return [] }

        let spWord = "▁\(word)"
        if vocab[spWord] != nil { return [spWord] }

        let chars = Array(spWord)
        var pieces: [String] = []
        var i = 0

        while i < chars.count {
            var bestLen = 0
            var bestPiece = ""

            for end in stride(from: min(chars.count, i + 20), through: i + 1, by: -1) {
                let candidate = String(chars[i..<end])
                if vocab[candidate] != nil && (end - i) > bestLen {
                    bestLen = end - i
                    bestPiece = candidate
                }
            }

            if bestLen > 0 {
                pieces.append(bestPiece)
                i += bestLen
            } else {
                pieces.append(String(chars[i]))
                i += 1
            }
        }
        return pieces
    }

    private func indicTransDetokenize(_ ids: [Int], vocab: [Int: String]) -> String {
        let specialIds: Set<Int> = [0, 1, 2, 3]
        let tokens = ids.filter { !specialIds.contains($0) }.compactMap { vocab[$0] }
        return tokens.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }

    // ── IndicProcessor Pre/Post-processing ──────────────────────────────────────

    private func indicPreprocess(text: String, srcLang: String, tgtLang: String) -> String {
        var processed = text.trimmingCharacters(in: .whitespaces)
        processed = processed.precomposedStringWithCanonicalMapping  // NFC normalize
        return ">> \(tgtLang) << \(processed)"
    }

    private func indicPostprocess(_ text: String, lang: String) -> String {
        var result = text.trimmingCharacters(in: .whitespaces)
        result = result.replacingOccurrences(of: "  ", with: " ")
        if let range = result.range(of: #">> \w+ <<"#, options: .regularExpression) {
            result = result.replacingCharacters(in: range, with: "").trimmingCharacters(in: .whitespaces)
        }
        return result
    }

    // ── TTS Character Tokenizer ─────────────────────────────────────────────────

    private func ttsTokenize(_ text: String, vocab: [String: Int]) -> [Int64] {
        return text.compactMap { ch -> Int64? in
            guard let id = vocab[String(ch)] else { return nil }
            return Int64(id)
        }
    }

    // ── Whisper BPE Decoder ─────────────────────────────────────────────────────

    private func whisperBpeDecode(_ tokenIds: [Int]) -> String {
        let pieces = tokenIds.compactMap { whisperVocab[$0] }
        return pieces.joined().replacingOccurrences(of: "Ġ", with: " ").trimmingCharacters(in: .whitespaces)
    }

    // ── Log-Mel Spectrogram (Accelerate) ────────────────────────────────────────

    private func computeLogMel(_ audio: [Float]) -> [Float] {
        let targetLen = 16000 * 30
        var padded = [Float](repeating: 0, count: targetLen)
        let copyLen = min(audio.count, targetLen)
        for i in 0..<copyLen { padded[i] = audio[i] }

        let numFrames = 3000
        let nFft = 400
        let hopLength = 160
        let nMels = 80
        let nFreqs = nFft / 2 + 1

        let window = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: nFft, isHalfWindow: false)

        var stft = [[Float]](repeating: [Float](repeating: 0, count: numFrames), count: nFreqs)

        for frame in 0..<numFrames {
            let start = frame * hopLength
            var windowed = [Float](repeating: 0, count: nFft)
            for i in 0..<nFft {
                let idx = start + i
                windowed[i] = idx < padded.count ? padded[idx] * window[i] : 0
            }

            var real = [Float](repeating: 0, count: nFft)
            var imag = [Float](repeating: 0, count: nFft)
            for i in 0..<nFft { real[i] = windowed[i] }

            cooleyTukeyFFT(&real, &imag, nFft)

            for k in 0..<nFreqs {
                stft[k][frame] = real[k] * real[k] + imag[k] * imag[k]
            }
        }

        var mel = [Float](repeating: 0, count: nMels * numFrames)
        for m in 0..<nMels {
            for f in 0..<numFrames {
                var sum: Float = 0
                for k in 0..<nFreqs {
                    sum += melFilters[m * nFreqs + k] * stft[k][f]
                }
                mel[m * numFrames + f] = logf(max(sum, 1e-10))
            }
        }

        let maxVal = mel.max() ?? 0
        for i in mel.indices {
            mel[i] = max(mel[i], maxVal - 8)
            mel[i] = (mel[i] + 4) / 4
        }

        return mel
    }

    private func cooleyTukeyFFT(_ real: inout [Float], _ imag: inout [Float], _ n: Int) {
        if n <= 1 { return }

        let bits = Int(log2(Double(n)))
        for i in 0..<n {
            let j = reverseBits(i, bits)
            if i < j {
                real.swapAt(i, j)
                imag.swapAt(i, j)
            }
        }

        var len = 2
        while len <= n {
            let halfLen = len / 2
            let angle = -2.0 * Double.pi / Double(len)
            for i in stride(from: 0, to: n, by: len) {
                for j in 0..<halfLen {
                    let wReal = Float(cos(angle * Double(j)))
                    let wImag = Float(sin(angle * Double(j)))
                    let ur = real[i + j]
                    let ui = imag[i + j]
                    let vr = real[i + j + halfLen] * wReal - imag[i + j + halfLen] * wImag
                    let vi = real[i + j + halfLen] * wImag + imag[i + j + halfLen] * wReal
                    real[i + j] = ur + vr
                    imag[i + j] = ui + vi
                    real[i + j + halfLen] = ur - vr
                    imag[i + j + halfLen] = ui - vi
                }
            }
            len *= 2
        }
    }

    private func reverseBits(_ x: Int, _ bits: Int) -> Int {
        var result = 0
        var v = x
        for _ in 0..<bits {
            result = (result << 1) | (v & 1)
            v >>= 1
        }
        return result
    }

    private func computeMelFilterbank() -> [Float] {
        let sampleRate = 16000
        let nFft = 400
        let nMels = 80
        let nFreqs = nFft / 2 + 1

        func hzToMel(_ hz: Double) -> Double { 2595.0 * log10(1.0 + hz / 700.0) }
        func melToHz(_ mel: Double) -> Double { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

        let fMax = Double(sampleRate) / 2.0
        let melMin = hzToMel(0)
        let melMax = hzToMel(fMax)
        let melPoints = (0...(nMels + 1)).map { i -> Double in
            melToHz(melMin + (melMax - melMin) * Double(i) / Double(nMels + 1))
        }
        let freqBins = melPoints.map { $0 * Double(nFft + 1) / Double(sampleRate) }

        var filters = [Float](repeating: 0, count: nMels * nFreqs)
        for m in 0..<nMels {
            let fLeft = freqBins[m]
            let fCenter = freqBins[m + 1]
            let fRight = freqBins[m + 2]

            for k in 0..<nFreqs {
                let kd = Double(k)
                let weight: Double
                if kd < fLeft { weight = 0 }
                else if kd <= fCenter { weight = (kd - fLeft) / (fCenter - fLeft) }
                else if kd <= fRight { weight = (fRight - kd) / (fRight - fCenter) }
                else { weight = 0 }
                filters[m * nFreqs + k] = Float(weight)
            }

            let norm = 2.0 / (melPoints[m + 2] - melPoints[m])
            for k in 0..<nFreqs { filters[m * nFreqs + k] *= Float(norm) }
        }
        return filters
    }

    // ── Asset Loading ───────────────────────────────────────────────────────────

    private func loadConfigJson(bundle: Bundle, dir: String) -> [String: Int] {
        guard let url = bundle.url(forResource: "config", withExtension: "json", subdirectory: "assets/tokenizers/\(dir)"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        var result: [String: Int] = [:]
        for (k, v) in dict {
            if let intVal = v as? Int { result[k] = intVal }
        }
        return result
    }

    private func loadVocabJson(bundle: Bundle, dir: String, file: String) -> [String: Int] {
        guard let url = bundle.url(forResource: file, withExtension: "json", subdirectory: "assets/tokenizers/\(dir)"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Int]
        else { return [:] }
        return dict
    }

    private func loadReverseVocabJson(bundle: Bundle, dir: String, file: String) -> [Int: String] {
        guard let url = bundle.url(forResource: file, withExtension: "json", subdirectory: "assets/tokenizers/\(dir)"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Int]
        else { return [:] }
        var result: [Int: String] = [:]
        for (k, v) in dict { result[v] = k }
        return result
    }

    private func loadMerges(bundle: Bundle) -> [(String, String)] {
        guard let url = bundle.url(forResource: "merges", withExtension: "txt", subdirectory: "assets/tokenizers/whisper"),
              let text = try? String(contentsOf: url)
        else { return [] }
        return text.components(separatedBy: "\n")
            .filter { !$0.hasPrefix("#") && !$0.isEmpty }
            .compactMap { line -> (String, String)? in
                let parts = line.components(separatedBy: " ")
                guard parts.count == 2 else { return nil }
                return (parts[0], parts[1])
            }
    }

    static let langTokens: [String: Int64] = [
        "hi":50301,"mr":50305,"ta":50315,"te":50316,"kn":50302,
        "ml":50306,"bn":50292,"gu":50300,"pa":50308,"ur":50319,
    ]
}
