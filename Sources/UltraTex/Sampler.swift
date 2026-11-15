// The UltraTex sampler: which tokens enter the DiT, the timestep schedule, and the foreground-only Euler loop.
// Follows upstream train_flux2.py (`inference`, `encode_conditions`, `foreground_token_indices`, `uno_schedule`)
// at yiboz2001/UltraTex @ a726678, with one pass per step: the release only ever uses the empty prompt, under
// which the second CFG branch repeats the first (AB-R-0878, AB-R-0880).
//
// Layout. The six views sit on a 2x3 canvas; at view size R the latent canvas is (2R/16) x (3R/16) tokens of 128
// channels. Three token sets share one sequence, told apart by the first position-id axis:
//   0   the target (noise, being denoised)      canvas grid
//   10  the normal maps                         canvas grid
//   20  the reference picture                   its own (R/16) x (R/16) grid
// Only tokens whose mask is set are kept; each keeps the position id it had on the full grid.

import Foundation
import MLX

public enum UltraTexSampler {

    // MARK: Schedule

    /// Upstream's `uno_schedule`: `steps + 1` timesteps from 1 to 0, shifted by the canvas size. Computed in
    /// Float, as upstream computes it on a float32 tensor.
    public static func timesteps(steps: Int, resolution: Int) -> [Float] {
        let seqLen = ((resolution * 3) / 8) * ((resolution * 2) / 8) / 256
        let slope = (1.15 - 0.5) / Double(4096 - 256)
        let mu = slope * Double(seqLen) + (0.5 - slope * 256)
        let e = Float(exp(mu))
        return (0 ... steps).map { i in
            let t = 1 - Float(i) / Float(steps)
            return t == 0 ? 0 : e / (e + (1 / t - 1))
        }
    }

    /// The value a bf16 tensor would hold (round to nearest, ties to even). Upstream holds the timestep and the
    /// step size in bf16, so the port rounds at the same points instead of trusting an fp32 value.
    public static func bf16(_ x: Float) -> Float {
        let bits = x.bitPattern
        let rounded = (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000
        return Float(bitPattern: rounded)
    }

    // MARK: Token selection

    public struct Selection {
        /// Kept positions on the canvas grid (row-major), for the target and for the normal maps.
        public var target: [Int32]
        public var normals: [Int32]
        /// Kept positions on the reference grid.
        public var reference: [Int32]
        public var canvasRows: Int
        public var canvasCols: Int
        public var referenceSide: Int

        public var targetCount: Int { target.count }
        public var keptCount: Int { target.count + normals.count + reference.count }
    }

    static func kept(_ mask: MLXArray) -> [Int32] {
        let values = mask.asType(.float32).reshaped(-1).asArray(Float.self)
        var out: [Int32] = []
        out.reserveCapacity(values.count / 3)
        for (i, v) in values.enumerated() where v != 0 { out.append(Int32(i)) }
        return out
    }

    /// Masks as upstream's dataset builds them: [rows, cols] of 0/1 on the token grid.
    public static func select(targetMask: MLXArray, normalsMask: MLXArray, referenceMask: MLXArray) -> Selection {
        precondition(targetMask.shape == normalsMask.shape, "target and normals masks differ in shape")
        precondition(referenceMask.dim(0) == referenceMask.dim(1), "the reference grid is square")
        return Selection(target: kept(targetMask), normals: kept(normalsMask), reference: kept(referenceMask),
                         canvasRows: targetMask.dim(0), canvasCols: targetMask.dim(1),
                         referenceSide: referenceMask.dim(0))
    }

    /// Position ids [n, 4] = (plane, row, col, 0) for kept positions of a grid with `cols` columns.
    static func ids(_ positions: [Int32], cols: Int, plane: Int32) -> MLXArray {
        var flat = [Int32](repeating: 0, count: positions.count * 4)
        for (i, p) in positions.enumerated() {
            flat[i * 4] = plane
            flat[i * 4 + 1] = p / Int32(cols)
            flat[i * 4 + 2] = p % Int32(cols)
        }
        return MLXArray(flat, [positions.count, 4])
    }

    /// Ids of the whole DiT sequence: target, then normals, then reference.
    public static func sequenceIds(_ s: Selection) -> MLXArray {
        concatenated([ids(s.target, cols: s.canvasCols, plane: 0),
                      ids(s.normals, cols: s.canvasCols, plane: 10),
                      ids(s.reference, cols: s.referenceSide, plane: 20)], axis: 0)
    }

    /// The kept condition tokens, normals then reference, from the full latents laid out as tokens:
    /// `normals` [rows*cols, 128] and `reference` [side*side, 128].
    public static func conditionTokens(_ s: Selection, normals: MLXArray, reference: MLXArray) -> MLXArray {
        concatenated([normals.take(MLXArray(s.normals), axis: 0),
                      reference.take(MLXArray(s.reference), axis: 0)], axis: 0)
    }

    // MARK: Sampling

    /// Text ids as upstream's `prc_txt`: (0, 0, 0, i).
    public static func textIds(count: Int) -> MLXArray {
        var flat = [Int32](repeating: 0, count: count * 4)
        for i in 0 ..< count { flat[i * 4 + 3] = Int32(i) }
        return MLXArray(flat, [count, 4])
    }

    /// Foreground-only Euler sampling. `noise` [targetCount, 128] is the target's starting noise at its kept
    /// positions; `conditions` [n, 128] are the kept normals and reference tokens; `text` [tokens, dim] is the
    /// cached empty-prompt embedding. Returns the denoised target tokens [targetCount, 128] in bf16.
    /// `onStep(i, x)` is called before step i with the target tokens entering it (and once more, with i = steps,
    /// with the result).
    public static func sample(
        model: Flux2Transformer, noise: MLXArray, conditions: MLXArray, sequenceIds: MLXArray, text: MLXArray,
        steps: Int = 25, resolution: Int, onStep: ((Int, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let ts = timesteps(steps: steps, resolution: resolution)
        let count = noise.dim(0)
        let cond = conditions.asType(.bfloat16).expandedDimensions(axis: 0)
        let encoder = text.asType(.bfloat16).expandedDimensions(axis: 0)
        let txtIds = textIds(count: text.dim(0))
        var x = noise.asType(.bfloat16)
        eval(x, cond, encoder)
        for i in 0 ..< steps {
            onStep?(i, x)
            // Upstream: the timestep is a bf16 tensor, multiplied by 1000 in bf16 before the sinusoid.
            let tModel = bf16(bf16(ts[i]) * 1000)
            let input = concatenated([x.expandedDimensions(axis: 0), cond], axis: 1)
            let velocity = model(input, encoder: encoder, timestep: MLXArray([tModel]), guidance: nil,
                                 imgIds: sequenceIds, txtIds: txtIds)[0, 0 ..< count, 0...]
            // x + dt * v with upstream's rounding: dt is a Python float turned into a bf16 scalar, the product
            // and the sum are bf16.
            let dt = bf16(ts[i + 1] - ts[i])
            let delta = (velocity.asType(.float32) * dt).asType(.bfloat16).asType(.float32)
            x = (x.asType(.float32) + delta).asType(.bfloat16)
            eval(x)
            MLX.Memory.clearCache()
        }
        onStep?(steps, x)
        return x
    }

    // MARK: After sampling

    /// The latent canvas upstream decodes: the canonical background everywhere, the denoised tokens at the kept
    /// target positions. `background` is the black-picture latent as tokens [rows*cols, 128]. Returns
    /// [rows, cols, 128] in float32.
    public static func composite(_ s: Selection, target: MLXArray, background: MLXArray) -> MLXArray {
        let total = s.canvasRows * s.canvasCols
        var slot = [Int32](repeating: 0, count: total)
        var isKept = [Bool](repeating: false, count: total)
        for (i, p) in s.target.enumerated() {
            slot[Int(p)] = Int32(i)
            isKept[Int(p)] = true
        }
        let spread = target.asType(.float32).take(MLXArray(slot), axis: 0)
        let keep = MLXArray(isKept).reshaped(total, 1)
        return which(keep, spread, background.asType(.float32)).reshaped(s.canvasRows, s.canvasCols, 128)
    }

    /// The six view tiles of a canvas [rows, cols, c], in upstream's decode order: the top row left to right
    /// (views 0, 1, 3), then the bottom row (views 2, 4, 5).
    public static func tiles(_ canvas: MLXArray) -> [MLXArray] {
        let (h, w) = (canvas.dim(0) / 2, canvas.dim(1) / 3)
        var out: [MLXArray] = []
        for r in 0 ..< 2 {
            for c in 0 ..< 3 {
                out.append(canvas[(r * h) ..< ((r + 1) * h), (c * w) ..< ((c + 1) * w), 0...])
            }
        }
        return out
    }
}
