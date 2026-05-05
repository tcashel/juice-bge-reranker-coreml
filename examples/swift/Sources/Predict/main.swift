// Minimal Swift example: download a tagged release of bge-reranker-base-coreml
// from the Hugging Face Hub, load the .mlpackage, score one (query, doc) pair,
// and print the sigmoid-mapped relevance.
//
// The construction below mirrors the integration contract documented in the
// repo's MODEL_CARD.md — paired-input template, fixed batch=20, BC1S input
// shape, sigmoid on the (20, 1) Float32 logit output.

import CoreML
import Foundation
import Hub
import Tokenizers

// XLM-RoBERTa special tokens (verbatim from MODEL_CARD).
let bos: Int32 = 0  // <s>
let pad: Int32 = 1  // <pad>
let eos: Int32 = 2  // </s>

// Match `convert.py`: batch is fixed at 20, S ∈ {128, 256, 512} via EnumeratedShapes.
let fixedBatch = 20
let seqLen = 128

struct Args {
    var repoId = "tcashel/bge-reranker-base-coreml"
    var tag = "v0.1-ane"
    var query = "what is the capital of france?"
    var doc = "Paris is the capital of France."
    var useNeuralEngine = true
}

func parseArgs() -> Args {
    var args = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let flag = it.next() {
        switch flag {
        case "--tag": args.tag = it.next() ?? args.tag
        case "--repo": args.repoId = it.next() ?? args.repoId
        case "--query": args.query = it.next() ?? args.query
        case "--doc": args.doc = it.next() ?? args.doc
        case "--cpugpu": args.useNeuralEngine = false
        default: FileHandle.standardError.write(Data("ignoring unknown arg: \(flag)\n".utf8))
        }
    }
    return args
}

// Build the XLM-R paired-input id sequence: [<s>, query..., </s>, </s>, doc..., </s>].
// We assemble token IDs directly because swift-transformers does not expose
// `encode(text:textPair:)` for the Unigram path used by this tokenizer.
func encodePair(tokenizer: any Tokenizer, query: String, doc: String, maxLength: Int) -> (ids: [Int32], mask: [Int32]) {
    let q = tokenizer.encode(text: query, addSpecialTokens: false).map(Int32.init)
    let d = tokenizer.encode(text: doc, addSpecialTokens: false).map(Int32.init)
    let reservedSpecials = 4  // <s>, </s>, </s>, </s>
    let docBudget = max(0, maxLength - q.count - reservedSpecials)
    let dTrunc = Array(d.prefix(docBudget))
    var ids: [Int32] = [bos] + q + [eos, eos] + dTrunc + [eos]
    if ids.count > maxLength {
        // Query alone overflowed; truncate the tail. Per MODEL_CARD this should be flagged
        // upstream by the orchestrator, but we degrade gracefully here for the example.
        ids = Array(ids.prefix(maxLength - 1)) + [eos]
    }
    var mask = [Int32](repeating: 1, count: ids.count)
    if ids.count < maxLength {
        ids += [Int32](repeating: pad, count: maxLength - ids.count)
        mask += [Int32](repeating: 0, count: maxLength - mask.count)
    }
    return (ids, mask)
}

func makeBatchedInputs(ids: [Int32], mask: [Int32], batch: Int, seq: Int) throws -> (MLMultiArray, MLMultiArray) {
    let shape = [NSNumber(value: batch), 1, 1, NSNumber(value: seq)]
    let idsArr = try MLMultiArray(shape: shape, dataType: .int32)
    let maskArr = try MLMultiArray(shape: shape, dataType: .int32)
    let idsPtr = idsArr.dataPointer.bindMemory(to: Int32.self, capacity: batch * seq)
    let maskPtr = maskArr.dataPointer.bindMemory(to: Int32.self, capacity: batch * seq)
    for i in 0..<seq {
        idsPtr[i] = ids[i]
        maskPtr[i] = mask[i]
    }
    // Remaining rows are all-pad / all-zero-mask. The model still emits 20 logits;
    // consumers read row 0 and discard the rest (per MODEL_CARD).
    for b in 1..<batch {
        for i in 0..<seq {
            idsPtr[b * seq + i] = pad
            maskPtr[b * seq + i] = 0
        }
    }
    return (idsArr, maskArr)
}

@main
struct Predict {
    static func main() async throws {
        let args = parseArgs()

        // Snapshot the tagged commit. The HF repo is laid out so that `tokenizer.json`,
        // `model.mlpackage`, and `provenance.json` all sit at the repo root.
        let repo = Hub.Repo(id: args.repoId, type: .models)
        let folder = try await HubApi.shared.snapshot(from: repo, revision: args.tag)
        print("snapshot: \(folder.path)")

        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)

        let (ids, mask) = encodePair(tokenizer: tokenizer, query: args.query, doc: args.doc, maxLength: seqLen)
        let (idsArr, maskArr) = try makeBatchedInputs(ids: ids, mask: mask, batch: fixedBatch, seq: seqLen)

        let config = MLModelConfiguration()
        config.computeUnits = args.useNeuralEngine ? .cpuAndNeuralEngine : .cpuAndGPU
        let modelURL = folder.appendingPathComponent("model.mlpackage")
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let prediction = try await model.prediction(from: provider)
        guard let logits = prediction.featureValue(for: "logit")?.multiArrayValue else {
            throw NSError(domain: "Predict", code: 1, userInfo: [NSLocalizedDescriptionKey: "no `logit` output"])
        }
        let logit = Double(truncating: logits[[0, 0] as [NSNumber]])
        let score = 1.0 / (1.0 + exp(-logit))

        print("\nquery: \(args.query)")
        print("doc:   \(args.doc)")
        print(String(format: "logit: %+.4f", logit))
        print(String(format: "score: %.4f  (sigmoid)", score))
    }
}
