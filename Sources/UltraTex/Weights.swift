// Loading the klein 9B DiT and merging the UltraTex LoRA into it.
//
// The base comes from the diffusers tree BFL publishes (`<repo>/transformer/*.safetensors`), whose keys match the
// vendored module but for two renames. The UltraTex LoRA is PEFT-format on BFL module names
// (`base_model.model.double_blocks.N.img_attn.qkv.lora_A.default.weight`), rank 64 with alpha = rank, so its scale
// is 1 (AB-R-0874). BFL fuses q, k and v into one projection; diffusers splits them, so a fused `qkv` update is
// cut into thirds along its output rows. The name map is the one `KleinLoRA` in flux2-klein-swift uses.
//
// Upstream merges with PEFT on a bf16 model: A and B are held in bf16, their product is rounded to bf16 and added
// to the bf16 weight. This follows the same rounding points.

import Foundation
import MLX
import MLXNN

public enum UltraTexWeights {
    public enum WeightError: Error, CustomStringConvertible {
        case noShards(String)
        case unmappedLoRA([String])
        case missingTarget(String)
        case shape(String)

        public var description: String {
            switch self {
            case .noShards(let p): "no .safetensors in \(p)"
            case .unmappedLoRA(let k): "LoRA tensors with no target in the DiT: \(k.prefix(5))"
            case .missingTarget(let k): "LoRA target \(k) is not a DiT weight"
            case .shape(let m): m
            }
        }
    }

    public struct Report: Sendable {
        public var baseTensors = 0
        public var loraModules = 0
        public var mergedWeights = 0
    }

    /// diffusers key -> Swift module path (as KleinWeights.remapDiTKey).
    static func remapDiTKey(_ key: String) -> String {
        key.replacingOccurrences(of: "time_guidance_embed.timestep_embedder.", with: "time_guidance_embed.")
            .replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")
    }

    /// BFL module path -> the DiT weights it updates. `third` selects the q, k or v rows of a fused update.
    static func loraTargets(_ module: String) -> [(path: String, third: Int?)]? {
        let parts = module.split(separator: ".").map(String.init)
        guard parts.count >= 3, let index = Int(parts[1]) else { return nil }
        let rest = parts[2...].joined(separator: ".")
        switch parts[0] {
        case "double_blocks":
            let block = "transformer_blocks.\(index).attn."
            switch rest {
            case "img_attn.qkv": return [(block + "to_q", 0), (block + "to_k", 1), (block + "to_v", 2)]
            case "img_attn.proj": return [(block + "to_out", nil)]
            case "txt_attn.qkv": return [(block + "add_q_proj", 0), (block + "add_k_proj", 1), (block + "add_v_proj", 2)]
            case "txt_attn.proj": return [(block + "to_add_out", nil)]
            default: return nil
            }
        case "single_blocks":
            let block = "single_transformer_blocks.\(index).attn."
            switch rest {
            case "linear1": return [(block + "to_qkv_mlp_proj", nil)]
            case "linear2": return [(block + "to_out", nil)]
            default: return nil
            }
        default:
            return nil
        }
    }

    static func loadShards(_ directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw WeightError.noShards(directory.path) }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        return weights
    }

    /// The DiT from `<directory>/*.safetensors` (a diffusers `transformer/` folder), in bf16, with `lora` merged
    /// when given. Strict: every base tensor must land, and every LoRA tensor must have a target.
    public static func loadTransformer(directory: URL, lora: URL? = nil, config: DiTConfig = .klein9B)
        throws -> (model: Flux2Transformer, report: Report)
    {
        var report = Report()
        // Reads and the merge run on the CPU stream: a multi-GB read on the GPU stream can outlast Metal's
        // command-buffer watchdog.
        let weights: [String: MLXArray] = try Device.withDefaultDevice(.cpu) {
            var weights: [String: MLXArray] = [:]
            for (key, value) in try loadShards(directory) {
                weights[remapDiTKey(key)] = value.asType(.bfloat16)
            }
            report.baseTensors = weights.count
            guard let lora else { return weights }

            var factors: [String: (a: MLXArray?, b: MLXArray?)] = [:]
            var unmapped: [String] = []
            for (key, value) in try MLX.loadArrays(url: lora) {
                var name = key
                if name.hasPrefix("base_model.model.") { name.removeFirst("base_model.model.".count) }
                if name.hasSuffix(".lora_A.default.weight") {
                    factors[String(name.dropLast(".lora_A.default.weight".count)), default: (nil, nil)].a = value
                } else if name.hasSuffix(".lora_B.default.weight") {
                    factors[String(name.dropLast(".lora_B.default.weight".count)), default: (nil, nil)].b = value
                } else {
                    unmapped.append(key)
                }
            }
            for (module, pair) in factors.sorted(by: { $0.key < $1.key }) {
                guard let a = pair.a, let b = pair.b, let targets = loraTargets(module) else {
                    unmapped.append(module)
                    continue
                }
                let a32 = a.asType(.bfloat16).asType(.float32)          // [rank, in], as PEFT holds it on a bf16 model
                let b32 = b.asType(.bfloat16).asType(.float32)          // [out, rank]
                let rows = b32.dim(0)
                for (path, third) in targets {
                    let key = path + ".weight"
                    guard let base = weights[key] else { throw WeightError.missingTarget(key) }
                    let part: MLXArray
                    if let third {
                        guard rows % 3 == 0 else { throw WeightError.shape("\(module): fused update has \(rows) rows") }
                        part = b32[(third * rows / 3) ..< ((third + 1) * rows / 3), 0...]
                    } else {
                        part = b32
                    }
                    let delta = matmul(part, a32).asType(.bfloat16)
                    guard delta.shape == base.shape else {
                        throw WeightError.shape("\(module) -> \(key): update \(delta.shape), weight \(base.shape)")
                    }
                    let merged = base + delta
                    eval(merged)
                    weights[key] = merged
                    report.mergedWeights += 1
                }
                report.loraModules += 1
            }
            guard unmapped.isEmpty else { throw WeightError.unmappedLoRA(unmapped.sorted()) }
            return weights
        }
        let model = Flux2Transformer(config: config)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return (model, report)
    }
}
