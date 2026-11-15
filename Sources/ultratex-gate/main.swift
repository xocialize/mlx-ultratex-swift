// ultratex-gate — the Swift DiT against the CUDA goldens (DEV/UltraTexOracle, --plan golden).
//
//   swift run -c release ultratex-gate --transformer <diffusers transformer dir> [--lora <dit_lora.safetensors>]
//                                      --golden <golden run dir>/st [--step 0] [--stream bf16|fp32] [--attn chunked|dense]
//
// Feeds the golden step's inputs (tokens, position ids, timestep, text embedding) through the DiT with dense
// attention and compares the velocity, and at step 0 the dumped block outputs. Cosine and relative L2 are over
// all elements; the goldens' block tensors are fp16 on disk, so the few elements that overflowed to infinity
// there are left out of both sides. Use it against the DENSE golden: the sparse golden's blocks differ by design.

import Foundation
import MLX
import MLXNN
import UltraTex

func arg(_ name: String, _ fallback: String) -> String {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: "--" + name), i + 1 < a.count else { return fallback }
    return a[i + 1]
}

let golden = URL(fileURLWithPath: arg("golden", ""))
let step = Int(arg("step", "0"))!
let stream = arg("stream", "bf16")
AttentionUtils.backend = arg("attn", "chunked") == "dense" ? .dense : .denseChunked(maxScoreBytes: 4 << 30)

func load(_ name: String) throws -> [String: MLXArray] {
    try Device.withDefaultDevice(.cpu) {
        try MLX.loadArrays(url: golden.appendingPathComponent(name + ".safetensors"))
    }
}

/// (cosine, relative L2 error, elements left out because the reference is not finite).
func compare(_ ours: MLXArray, _ reference: MLXArray) -> (Float, Float, Int) {
    let r = reference.asType(.float32)
    let o = ours.asType(.float32)
    let bad = MLX.logicalNot(MLX.isFinite(r))
    let zero = MLXArray(Float(0))
    let rr = which(bad, zero, r)
    let oo = which(bad, zero, o)
    let dot = (rr * oo).sum().item(Float.self)
    let nr = (rr * rr).sum().sqrt().item(Float.self)
    let no = (oo * oo).sum().sqrt().item(Float.self)
    let diff = ((oo - rr) * (oo - rr)).sum().sqrt().item(Float.self)
    return (dot / (nr * no), diff / nr, bad.sum().item(Int.self))
}

nonisolated(unsafe) var rows: [(String, Float, Float, Int)] = []
func record(_ name: String, _ ours: MLXArray, _ reference: MLXArray) {
    precondition(ours.shape == reference.shape, "\(name): ours \(ours.shape), golden \(reference.shape)")
    let (c, e, n) = compare(ours, reference)
    rows.append((name, c, e, n))
    FileHandle.standardError.write("\(name): cos \(c) rel \(e)\n".data(using: .utf8)!)
}

let t0 = Date()
let loraPath = arg("lora", "")                           // leave out for a control run on the bare base model
let loaded = try UltraTexWeights.loadTransformer(
    directory: URL(fileURLWithPath: arg("transformer", "")),
    lora: loraPath.isEmpty ? nil : URL(fileURLWithPath: loraPath))
let model = loaded.model
if stream == "bf16" { model.streamDType = .bfloat16 }
let loadSeconds = Date().timeIntervalSince(t0)

let s = try load(String(format: "step_%02d", step))
let s0 = step == 0 ? s : try load("step_00")            // the text embedding is dumped with step 0 only
let hidden = s["x"]!.expandedDimensions(axis: 0).asType(.bfloat16)
let encoder = s0["ctx"]!.expandedDimensions(axis: 0).asType(.bfloat16)
// Upstream multiplies the bf16 timestep by 1000 in bf16 before the sinusoid, so the model sees e.g. 644, not
// 644.53. A value above 1 also switches off the DiT's own x1000.
let timestep = (s["t"]!.asType(.bfloat16) * MLXArray(Float(1000)).asType(.bfloat16)).asType(.float32).reshaped(1)

if step == 0 {
    let doubles = [0, 3, 7], singles = [0, 11, 23]
    model.blockTap = { name, a, b in
        let parts = name.split(separator: "_")
        guard let index = Int(parts[1]) else { return }
        do {
            if parts[0] == "double", doubles.contains(index), let b {
                let g = try load(String(format: "block_double_%02d", index))
                record("double \(index) image", a[0], g["img"]!)
                record("double \(index) text", b[0], g["txt"]!)
            } else if parts[0] == "single", singles.contains(index) {
                let g = try load(String(format: "block_single_%02d", index))
                record("single \(index) text+image", a[0], g["seq"]!)
            }
        } catch {
            fatalError("\(name): \(error)")
        }
    }
}

let t1 = Date()
let out = model(hidden, encoder: encoder, timestep: timestep, guidance: nil,
                imgIds: s["x_ids"]!, txtIds: s0["ctx_ids"]!)
eval(out)
let passSeconds = Date().timeIntervalSince(t1)
record("velocity", out[0], s["velocity"]!)

print("golden \(golden.deletingLastPathComponent().lastPathComponent), step \(step), stream \(stream), "
      + "tokens \(hidden.dim(1)), load \(String(format: "%.1f", loadSeconds)) s, pass \(String(format: "%.1f", passSeconds)) s")
print("stage                       cosine      rel L2   left out")
for (name, c, e, n) in rows {
    print(name.padding(toLength: 26, withPad: " ", startingAt: 0)
          + String(format: "  %.6f  %.5f  %d", c, e, n))
}
