import Foundation
import onnxruntime_objc

private let intentTAG    = "IndicAI.Intent"
private let INTENT_MAX_LEN  = 64
private let INTENT_CLS_ID: Int64 = 101
private let INTENT_SEP_ID: Int64 = 102
private let INTENT_UNK_ID: Int64 = 100
private let INTENT_PAD_ID: Int64 = 0
private let INTENT_EMB_DIM  = 384

/**
 * On-device intent classification using MiniLM-L6 semantic similarity.
 *
 * Mirrors Android IntentEngine.kt exactly:
 *   1. WordPiece-tokenize the English query
 *   2. Run ONNX encoder → L2-normalised 384-dim sentence embedding
 *   3. Cosine similarity against pre-computed intent bank (intent_bank.npy)
 *   4. Nearest neighbour → intent label + similarity score as confidence
 *
 * Files loaded from the extracted intent zip:
 *   intent_encoder.onnx       — MiniLM encoder (output: "sentence_embedding")
 *   tokenizer/vocab.txt       — BERT WordPiece vocabulary
 *   intent_bank.npy           — (N × 384) float32 row-major L2-normalised embeddings
 *   intent_bank_labels.json   — JSON array of N label strings
 */
class IntentEngine {
    private var session: ORTSession?
    private var intentEnv: ORTEnv?
    private var vocab: [String: Int] = [:]
    private var bankVecs: [[Float]] = []     // [N, 384]
    private var bankLabels: [String] = []

    func isLoaded() -> Bool { session != nil }

    func load(modelFile: URL, vocabFile: URL, bankNpyFile: URL, bankLabelsFile: URL) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        self.intentEnv = env
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(2)
        try opts.setGraphOptimizationLevel(.all)
        session = try ORTSession(env: env, modelPath: modelFile.path, sessionOptions: opts)
        vocab = try buildVocab(from: vocabFile)
        bankVecs = try loadNpy(from: bankNpyFile)
        bankLabels = try parseLabels(from: bankLabelsFile)
        NSLog("[%@] Loaded: %d bank phrases, %d unique intents",
              intentTAG, bankLabels.count, Set(bankLabels).count)
    }

    func classify(_ text: String) throws -> IntentResult {
        guard let session = session, let env = intentEnv else {
            throw IndicAIError.notInitialized
        }

        // 1. WordPiece tokenize
        let tokenIds = wordpiece(text.lowercased().trimmingCharacters(in: .whitespaces))
        let attn: [Int64] = tokenIds.map { $0 != INTENT_PAD_ID ? 1 : 0 }

        let inputIdsData = tokenIds.withUnsafeBufferPointer { Data(buffer: $0) }
        let attnData     = attn.withUnsafeBufferPointer { Data(buffer: $0) }

        let inputIds = try ORTValue(
            tensorData: NSMutableData(data: inputIdsData),
            elementType: .int64,
            shape: [1, NSNumber(value: INTENT_MAX_LEN)]
        )
        let attnMask = try ORTValue(
            tensorData: NSMutableData(data: attnData),
            elementType: .int64,
            shape: [1, NSNumber(value: INTENT_MAX_LEN)]
        )

        // 2. Encoder → L2-normalised sentence embedding (1, 384)
        let outputs = try session.run(
            withInputs: ["input_ids": inputIds, "attention_mask": attnMask],
            outputNames: ["sentence_embedding"],
            runOptions: nil
        )

        let embData = try outputs["sentence_embedding"]!.tensorData() as Data
        let count = embData.count / MemoryLayout<Float>.size
        let queryVec: [Float] = embData.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self).prefix(count))
        }

        // 3. Cosine similarity — vecs are L2-normalised → dot product = cosine sim
        var bestIdx = 0
        var bestSim: Float = -1
        for (i, row) in bankVecs.enumerated() {
            var dot: Float = 0
            for d in 0..<INTENT_EMB_DIM { dot += row[d] * queryVec[d] }
            if dot > bestSim { bestSim = dot; bestIdx = i }
        }

        let intent = bestIdx < bankLabels.count ? bankLabels[bestIdx] : "unknown"
        NSLog("[%@] classify('%@') → %@ (sim=%.3f)", intentTAG, text, intent, bestSim)
        return IntentResult(intent: intent, confidence: bestSim)
    }

    func close() { session = nil; intentEnv = nil }

    // MARK: - WordPiece tokenizer (matches BERT bert-base-uncased BasicTokenizer)

    private func wordpiece(_ text: String) -> [Int64] {
        // NFD normalize, strip non-spacing marks (diacritics)
        let nfd = text.decomposedStringWithCompatibilityMapping
        var normalized = ""
        for scalar in nfd.unicodeScalars {
            if scalar.properties.generalCategory != .nonspacingMark {
                normalized.unicodeScalars.append(scalar)
            }
        }

        let words = normalized.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        var pieces: [Int64] = [INTENT_CLS_ID]

        outer: for word in words {
            if pieces.count >= INTENT_MAX_LEN - 1 { break }
            var subTokens: [Int64] = []
            var start = word.startIndex
            var unk = false

            while start < word.endIndex {
                var end = word.endIndex
                var found: Int64? = nil

                while start < end {
                    let sub = start == word.startIndex
                        ? String(word[start..<end])
                        : "##\(String(word[start..<end]))"
                    if let id = vocab[sub] {
                        found = Int64(id)
                        break
                    }
                    end = word.index(before: end)
                }

                if found == nil { unk = true; break }
                subTokens.append(found!)
                start = end
            }

            if unk {
                if pieces.count < INTENT_MAX_LEN - 1 { pieces.append(INTENT_UNK_ID) }
            } else {
                for tok in subTokens {
                    if pieces.count >= INTENT_MAX_LEN - 1 { break outer }
                    pieces.append(tok)
                }
            }
        }
        pieces.append(INTENT_SEP_ID)

        return (0..<INTENT_MAX_LEN).map { $0 < pieces.count ? pieces[$0] : INTENT_PAD_ID }
    }

    // MARK: - Vocab, NPY, Labels loaders

    private func buildVocab(from file: URL) throws -> [String: Int] {
        let text = try String(contentsOf: file, encoding: .utf8)
        var map: [String: Int] = [:]
        let lines = text.components(separatedBy: "\n")
        for (idx, line) in lines.enumerated() {
            let token = line.trimmingCharacters(in: .newlines)
            if !token.isEmpty { map[token] = idx }
        }
        return map
    }

    /// Parse NPY v1.0 file: magic(6) + major(1) + minor(1) + header_len(2 LE) + header + float32 data.
    private func loadNpy(from file: URL) throws -> [[Float]] {
        let bytes = try Data(contentsOf: file)
        guard bytes.count >= 10 else {
            throw IndicAIError.inferenceError("NPY file too small")
        }
        let headerLen = Int(bytes[8]) | (Int(bytes[9]) << 8)
        let offset = 10 + headerLen
        let totalFloats = (bytes.count - offset) / 4
        let rows = totalFloats / INTENT_EMB_DIM
        guard rows > 0 else {
            throw IndicAIError.inferenceError("Empty intent bank NPY")
        }

        var result = [[Float]](repeating: [Float](repeating: 0, count: INTENT_EMB_DIM), count: rows)
        bytes.withUnsafeBytes { raw in
            // All Apple Silicon and x86 iOS devices are little-endian — float32 maps directly
            raw.baseAddress!.advanced(by: offset)
                .withMemoryRebound(to: Float.self, capacity: totalFloats) { floats in
                    for r in 0..<rows {
                        for d in 0..<INTENT_EMB_DIM {
                            result[r][d] = floats[r * INTENT_EMB_DIM + d]
                        }
                    }
                }
        }
        return result
    }

    private func parseLabels(from file: URL) throws -> [String] {
        let data = try Data(contentsOf: file)
        guard let arr = try JSONSerialization.jsonObject(with: data) as? [String] else {
            throw IndicAIError.inferenceError("Invalid intent_bank_labels.json")
        }
        return arr
    }
}

// Shared result type (also used by IndicAIModule)
struct IntentResult {
    let intent: String
    let confidence: Float
}
