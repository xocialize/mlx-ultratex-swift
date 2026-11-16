// A whole UltraTex run: a mesh and a reference picture in, six albedo views out.
//
//   render six normal maps -> prepare inputs -> encode the normals and the reference -> pick the kept tokens ->
//   25 foreground-only Euler steps -> lay the result on the black-latent canvas -> decode each view -> whiten
//   the background
//
// Each stage is gated against the CUDA oracle on its own (AB-R-0892 to AB-R-0895). Baking the six views onto the
// mesh's texture is a later stage and is not here.

import Foundation
import MLX
import MLXRandom

public struct UltraTexPipeline {
    public let model: Flux2Transformer
    public let vae: UltraTexVAE
    /// The cached empty-prompt embedding [512, 12288] (`empty_prompt_qwen3_8b.safetensors`, key `ctx`).
    public let text: MLXArray

    public init(model: Flux2Transformer, vae: UltraTexVAE, text: MLXArray) {
        (self.model, self.vae, self.text) = (model, vae, text)
    }

    public struct Output {
        /// The 2x3 canvas (views 0, 1, 3 over 2, 4, 5), opaque RGB in RGBA8.
        public var canvas: UltraTexImage
        /// The six views, each `resolution` square, in view order.
        public var views: [UltraTexImage]
        public var keptTokens: Int
        public var seconds: [String: Double]
    }

    /// `noise`, when given, is the target's starting noise at its kept positions [n, 128] (for parity gates);
    /// otherwise the whole canvas is drawn from `seed` and the kept positions are taken from it, so a position's
    /// noise does not depend on the mask.
    public func run(mesh: UltraTexMesh, reference: UltraTexImage, resolution: Int, steps: Int = 25, seed: UInt64 = 42,
                    distance: Double = UltraTexRenderer.defaultDistance, noise: MLXArray? = nil,
                    onStep: ((Int) -> Void)? = nil) -> Output {
        let t = Date()
        let prepared = Self.prepare(mesh: mesh, reference: reference, resolution: resolution, distance: distance)
        var out = run(prepared: prepared, steps: steps, seed: seed, noise: noise, onStep: onStep)
        out.seconds["render and prepare"] = Date().timeIntervalSince(t) - out.seconds.values.reduce(0, +)
        return out
    }

    /// The CPU half of a run: render the six normal maps and prepare what the model consumes. No model and no MLX
    /// memory to speak of, so a caller can size the run (see `keptTokens`) before loading anything.
    public static func prepare(mesh: UltraTexMesh, reference: UltraTexImage, resolution: Int,
                               distance: Double = UltraTexRenderer.defaultDistance) -> UltraTexInputs.Prepared {
        let views = UltraTexRenderer.normalMaps(mesh, size: 2048, distance: distance)
            .map { UltraTexImage(width: $0.size, height: $0.size, rgba: $0.rgba) }
        return UltraTexInputs.prepare(views: views, reference: reference, resolution: resolution)
    }

    /// Tokens the DiT will see for these inputs, without the 512 text tokens.
    public static func keptTokens(_ prepared: UltraTexInputs.Prepared) -> Int {
        UltraTexSampler.select(targetMask: prepared.canvasMask, normalsMask: prepared.canvasMask,
                               referenceMask: prepared.referenceMask).keptCount
    }

    /// The model half: encode, sample, decode.
    public func run(prepared: UltraTexInputs.Prepared, steps: Int = 25, seed: UInt64 = 42, noise: MLXArray? = nil,
                    onStep: ((Int) -> Void)? = nil) -> Output {
        let resolution = prepared.resolution
        var seconds: [String: Double] = [:]
        func timed<T>(_ name: String, _ body: () -> T) -> T {
            let t = Date()
            let out = body()
            seconds[name, default: 0] += Date().timeIntervalSince(t)
            return out
        }
        let side = resolution / 16
        let sel = UltraTexSampler.select(targetMask: prepared.canvasMask, normalsMask: prepared.canvasMask,
                                         referenceMask: prepared.referenceMask)

        // Latents. Views are encoded one at a time and laid out 2x3 as on the pixel canvas.
        let (normalTokens, referenceTokens, background) = timed("encode") {
            let tiles = prepared.normalViews.map { vae.encode($0)[0].transposed(1, 2, 0) }           // [side, side, 128]
            let canvas = concatenated([concatenated([tiles[0], tiles[1], tiles[3]], axis: 1),
                                       concatenated([tiles[2], tiles[4], tiles[5]], axis: 1)], axis: 0)
            let black = vae.encode(MLXArray.full([1, 3, resolution, resolution], values: MLXArray(Float(-1))))[0].transposed(1, 2, 0)
            let blackRow = concatenated([black, black, black], axis: 1)
            let bg = concatenated([blackRow, blackRow], axis: 0).reshaped(6 * side * side, 128)
            let ref = vae.encodeTokens(prepared.reference)
            eval(canvas, bg, ref)
            return (canvas.reshaped(6 * side * side, 128), ref, bg)
        }
        let conditions = UltraTexSampler.conditionTokens(sel, normals: normalTokens, reference: referenceTokens)
        let start: MLXArray
        if let noise {
            start = noise
        } else {
            let full = MLXRandom.normal([6 * side * side, 128], key: MLXRandom.key(seed))
            start = full.take(MLXArray(sel.target), axis: 0)
        }
        let result = timed("sample") {
            UltraTexSampler.sample(model: model, noise: start, conditions: conditions,
                                   sequenceIds: UltraTexSampler.sequenceIds(sel), text: text, steps: steps,
                                   resolution: resolution) { i, _ in onStep?(i) }
        }
        let latents = UltraTexSampler.composite(sel, target: result, background: background)
        var views: [UltraTexImage] = []
        timed("decode") {
            // Tiles come in canvas order (views 0, 1, 3, 2, 4, 5); put them back in view order.
            let order = [0, 1, 3, 2, 4, 5]
            var byView = [UltraTexImage?](repeating: nil, count: 6)
            for (slot, tile) in UltraTexSampler.tiles(latents).enumerated() {
                let view = order[slot]
                let rgb = floor(vae.decodeTile(tile)).asType(.uint8).reshaped(-1).asArray(UInt8.self)
                let alpha = prepared.viewAlpha[view]
                var rgba = [UInt8](repeating: 255, count: resolution * resolution * 4)
                for p in 0 ..< resolution * resolution where alpha[p] != 0 {
                    rgba[p * 4] = rgb[p * 3]; rgba[p * 4 + 1] = rgb[p * 3 + 1]; rgba[p * 4 + 2] = rgb[p * 3 + 2]
                }
                byView[view] = UltraTexImage(width: resolution, height: resolution, rgba: rgba)
            }
            views = byView.map { $0! }
        }
        // The canvas: views 0, 1, 3 over 2, 4, 5.
        let layout = [[0, 1, 3], [2, 4, 5]]
        var canvas = [UInt8](repeating: 255, count: 2 * resolution * 3 * resolution * 4)
        for (row, rowViews) in layout.enumerated() {
            for (col, view) in rowViews.enumerated() {
                for y in 0 ..< resolution {
                    let from = y * resolution * 4
                    let to = ((row * resolution + y) * 3 * resolution + col * resolution) * 4
                    canvas.replaceSubrange(to ..< (to + resolution * 4), with: views[view].rgba[from ..< (from + resolution * 4)])
                }
            }
        }
        return Output(canvas: UltraTexImage(width: 3 * resolution, height: 2 * resolution, rgba: canvas), views: views,
                      keptTokens: sel.keptCount, seconds: seconds)
    }
}
