// ultratex-bench — how long one DiT pass takes on this Mac at a given kept-token count.
//
//   swift run -c release ultratex-bench --tokens 19174 [--text 512] [--size 9b|4b] [--stream bf16|fp32]
//                                        [--attn dense|chunked] [--chunk-gb 4] [--passes 3]
//                                        [--transformer <diffusers transformer dir> [--lora <dit_lora.safetensors>]]
//
// Without --transformer the weights are RANDOM (klein-9B-shaped, bf16): time and memory, not output. With it the
// real DiT is loaded (and the LoRA merged), which also proves the loader; the inputs are still random. One pass is one
// sampling step with a single CFG branch, which is what the port runs (AB-R-0878: the second branch is a no-op).
// Prints one JSON line.

import Foundation
import MLX
import MLXNN
import MLXRandom
import UltraTex

func arg(_ name: String, _ fallback: String) -> String {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: "--" + name), i + 1 < a.count else { return fallback }
    return a[i + 1]
}

let tokens = Int(arg("tokens", "19174"))!
let textTokens = Int(arg("text", "512"))!
let size = arg("size", "9b")
let stream = arg("stream", "bf16")
let attn = arg("attn", "chunked")
let chunkGB = Double(arg("chunk-gb", "4"))!
let passes = Int(arg("passes", "3"))!

let config: DiTConfig = size == "4b" ? .klein4B : .klein9B
AttentionUtils.backend = attn == "dense" ? .dense : .denseChunked(maxScoreBytes: Int(chunkGB * 1_073_741_824))

func mb(_ bytes: Int) -> Double { (Double(bytes) / 1_048_576).rounded() }

// Build, cast to bf16, and materialize the parameters a few at a time (one giant eval can trip the Metal watchdog).
let t0 = Date()
let transformerDir = arg("transformer", "")
let loraPath = arg("lora", "")
var paramCount = 0
var weightsLabel = "random"
let model: Flux2Transformer
if transformerDir.isEmpty {
    model = Flux2Transformer(config: config)
    let flat = model.parameters().flattened()
    var cast: [(String, MLXArray)] = []
    for (i, (key, value)) in flat.enumerated() {
        let half = value.asType(.bfloat16)
        cast.append((key, half))
        paramCount += value.size
        if i % 8 == 7 { eval(cast.suffix(8).map { $0.1 }) }
    }
    eval(cast.map { $0.1 })
    model.update(parameters: ModuleParameters.unflattened(cast))
} else {
    let loaded = try UltraTexWeights.loadTransformer(
        directory: URL(fileURLWithPath: transformerDir),
        lora: loraPath.isEmpty ? nil : URL(fileURLWithPath: loraPath), config: config)
    model = loaded.model
    paramCount = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
    weightsLabel = "klein-9B base, \(loaded.report.baseTensors) tensors"
        + (loraPath.isEmpty ? "" : " + LoRA (\(loaded.report.loraModules) modules, \(loaded.report.mergedWeights) weights merged)")
}
if stream == "bf16" { model.streamDType = .bfloat16 }
MLX.Memory.clearCache()
let loadSeconds = Date().timeIntervalSince(t0)
let resident = MLX.Memory.activeMemory

MLXRandom.seed(0)
let hidden = MLXRandom.normal([1, tokens, 128]).asType(.bfloat16)
let encoder = MLXRandom.normal([1, textTokens, config.jointAttentionDim]).asType(.bfloat16)
// Position ids the way UltraTex lays them out: axis 0 is the image plane (0 target, 10 normals, 20 reference),
// axes 1-2 are rows and columns of the 2x3 canvas, axis 3 is zero for image tokens.
let plane = MLXRandom.randInt(low: 0, high: 3, [tokens, 1]) * 10
let row = MLXRandom.randInt(low: 0, high: 256, [tokens, 1])
let col = MLXRandom.randInt(low: 0, high: 384, [tokens, 1])
let imgIds = concatenated([plane, row, col, MLXArray.zeros([tokens, 1], dtype: .int32)], axis: 1)
let txtIds = concatenated([MLXArray.zeros([textTokens, 3], dtype: .int32),
                           MLXArray(Array(0 ..< Int32(textTokens))).reshaped(textTokens, 1)], axis: 1)
eval(hidden, encoder, imgIds, txtIds)

var seconds: [Double] = []
var finite = true
var outDType = ""
MLX.Memory.peakMemory = 0
for pass in 0 ... passes {            // pass 0 warms up (kernel compilation) and is not counted
    let t = Date()
    let out = model(hidden, encoder: encoder, timestep: MLXArray([Float(0.5)]), guidance: nil,
                    imgIds: imgIds, txtIds: txtIds)
    eval(out)
    let dt = Date().timeIntervalSince(t)
    if pass > 0 { seconds.append(dt) }
    if pass == passes {
        finite = !MLX.isNaN(out.asType(.float32)).any().item(Bool.self)
        outDType = "\(out.dtype)"
    }
    MLX.Memory.clearCache()
    FileHandle.standardError.write("pass \(pass): \(String(format: "%.2f", dt)) s\n".data(using: .utf8)!)
}
let mean = seconds.reduce(0, +) / Double(max(seconds.count, 1))
let report: [String: Any] = [
    "size": size, "params": paramCount, "tokens": tokens, "text_tokens": textTokens, "stream": stream,
    "attention": attn, "chunk_gb": chunkGB, "attention_chunks": AttentionUtils.lastChunkCount,
    "pass_seconds": seconds.map { ($0 * 100).rounded() / 100 }, "pass_seconds_mean": (mean * 100).rounded() / 100,
    "steps25_minutes": (mean * 25 / 60 * 10).rounded() / 10,
    "resident_mb": mb(resident), "peak_mb": mb(MLX.Memory.peakMemory), "load_seconds": (loadSeconds * 10).rounded() / 10,
    "output_dtype": outDType, "output_finite": finite, "weights": weightsLabel,
]
let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
