// Vendored from xocialize/flux2-klein-swift (MIT), Sources/Klein/Transformer.swift @ v0.11.0, for the UltraTex
// port (AB-T-0489). Changes, all marked "UltraTex:":
//   - DiTConfig carries the model size (klein 4B or 9B); the layers were already size-generic;
//   - attention goes through AttentionBackend (dense, dense chunked over query rows; block-sparse to come);
//   - `streamDType` keeps the activation stream in bf16 instead of letting fp32 modulation promote it;
//   - the to_out row-chunk workaround for mlx#3797 is off by default (this package needs mlx-swift >= 0.32.3).
// The 9B weights are under BFL's non-commercial licence, which is why this lives outside flux2-klein-swift.
//
// Flux2Transformer — isomorphic Swift/MLX port of the FLUX.2-klein MMDiT
// (mflux src/mflux/models/flux2/model/flux2_transformer/). Same class/method names as the
// Python-MLX reference; op-for-op translation. klein-4B config: 5 double-stream + 20
// single-stream blocks, 24 heads × 128 dim (inner 3072), joint_attention_dim 7680,
// in_channels 128, patch 1, 4-axis RoPE (θ2000, axes 32×4), mlp_ratio 3.0, eps 1e-6.

import Foundation
import MLX
import MLXFast
import MLXNN

/// UltraTex: the two klein sizes. Everything else is shared (see KleinConfig).
public struct DiTConfig: Sendable {
    public var numLayers: Int
    public var numSingleLayers: Int
    public var numHeads: Int
    public var jointAttentionDim: Int
    public static let klein4B = DiTConfig(numLayers: 5, numSingleLayers: 20, numHeads: 24, jointAttentionDim: 7680)
    /// hidden 4096 = 32 heads x 128; context 12288 = 3 x Qwen3-8B hidden 4096 (BFL Klein9BParams).
    public static let klein9B = DiTConfig(numLayers: 8, numSingleLayers: 24, numHeads: 32, jointAttentionDim: 12288)
}

/// UltraTex: how self-attention is computed. Set once before a forward (as upstream's
/// configure_sparse_attention does); not thread-safe by design.
public enum AttentionBackend: Sendable {
    /// One scaled-dot-product call over the whole sequence. Scores are [B, H, S, S].
    case dense
    /// The same result, computed over blocks of query rows so that one block's scores stay under
    /// `maxScoreBytes`. Each block is evaluated before the next is built, which bounds peak memory.
    case denseChunked(maxScoreBytes: Int)
}

public enum KleinConfig {
    public static let inChannels = 128
    public static let numLayers = 5
    public static let numSingleLayers = 20
    public static let numHeads = 24
    public static let headDim = 128
    public static let innerDim = 3072        // numHeads * headDim
    public static let jointAttentionDim = 7680   // 3 × Qwen3 hidden 2560
    public static let mlpRatio: Float = 3.0
    public static let patchSize = 1
    public static let ropeTheta: Double = 2000
    public static let axesDimsRope = [32, 32, 32, 32]
    public static let timestepGuidanceChannels = 256
    public static let eps: Float = 1e-6
}

// MARK: - RoPE (4-axis)

/// Flux2PosEmbed — plain class (tables aren't parameters). Real (cos, sin) per axis,
/// concatenated over the 4 axes → head_dim/2 columns.
public final class Flux2PosEmbed {
    let theta: Double
    let axesDim: [Int]

    public init(theta: Double = KleinConfig.ropeTheta, axesDim: [Int] = KleinConfig.axesDimsRope) {
        self.theta = theta
        self.axesDim = axesDim
    }

    /// ids: [N, nAxes] → (cos, sin) each [N, sum(dim/2)].
    public func callAsFunction(_ ids: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let pos = ids.asType(.float32)
        var cosParts: [MLXArray] = []
        var sinParts: [MLXArray] = []
        for (i, dim) in axesDim.enumerated() {
            let (c, s) = get1DRope(dim: dim, pos: pos[0..., i])
            cosParts.append(c)
            sinParts.append(s)
        }
        return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))
    }

    private func get1DRope(dim: Int, pos: MLXArray) -> (MLXArray, MLXArray) {
        // scale = arange(0, dim, 2)/dim ; omega = 1/theta^scale ; out = pos ⊗ omega
        let half = dim / 2
        let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim)
        let omega = MLX.pow(MLXArray(Float(theta)), -scale)   // 1/theta^scale
        let out = pos[0..., .newAxis] * omega[.newAxis, 0...]
        _ = half
        return (cos(out), sin(out))
    }
}

/// apply_rope_bshd — interleaved-pair rotation. x: [B,H,S,D]; cos/sin: [S, D/2].
/// (real, imag) = (x[..0], x[..1]) → (real·cos − imag·sin, imag·cos + real·sin).
func applyRopeBSHD(_ xq: MLXArray, _ xk: MLXArray, cos: MLXArray, sin: MLXArray)
    -> (MLXArray, MLXArray)
{
    let outDtype = xq.dtype
    let cosB = cos.reshaped(1, 1, cos.shape[0], cos.shape[1])
    let sinB = sin.reshaped(1, 1, sin.shape[0], sin.shape[1])
    func mix(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let x2 = x.asType(.float32).reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
        let real = x2[0..., 0..., 0..., 0..., 0]
        let imag = x2[0..., 0..., 0..., 0..., 1]
        let out0 = real * cosB - imag * sinB
        let out1 = imag * cosB + real * sinB
        return stacked([out0, out1], axis: -1).reshaped(shape)
    }
    return (mix(xq).asType(outDtype), mix(xk).asType(outDtype))
}

// MARK: - Timestep + guidance embedding

public final class Flux2TimestepGuidanceEmbeddings: Module {
    let inChannels: Int
    let guidanceEmbeds: Bool
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "guidance_linear_1") var guidanceLinear1: Linear?
    @ModuleInfo(key: "guidance_linear_2") var guidanceLinear2: Linear?

    public init(inChannels: Int = 256, embeddingDim: Int, guidanceEmbeds: Bool = false) {
        self.inChannels = inChannels
        self.guidanceEmbeds = guidanceEmbeds
        self._linear1.wrappedValue = Linear(inChannels, embeddingDim, bias: false)
        self._linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
        self._guidanceLinear1.wrappedValue = guidanceEmbeds ? Linear(inChannels, embeddingDim, bias: false) : nil
        self._guidanceLinear2.wrappedValue = guidanceEmbeds ? Linear(embeddingDim, embeddingDim, bias: false) : nil
    }

    static func timestepEmbedding(_ timesteps: MLXArray, dim: Int) -> MLXArray {
        // flip_sin_to_cos=True: [cos(second half) , sin(first half)] layout.
        let half = dim / 2
        let freqs = exp(-Float(Foundation.log(10000.0)) * MLXArray(0..<half).asType(.float32) / Float(half))
        let args = timesteps.asType(.float32)[0..., .newAxis] * freqs[.newAxis, 0...]
        var emb = concatenated([sin(args), cos(args)], axis: -1)   // [sin | cos]
        emb = concatenated([emb[0..., half...], emb[0..., ..<half]], axis: -1)   // flip → [cos | sin]
        return emb
    }

    public func callAsFunction(_ timestep: MLXArray, _ guidance: MLXArray?) -> MLXArray {
        let t = Self.timestepEmbedding(timestep.asType(.float32), dim: inChannels)
        var emb = linear2(silu(linear1(t)))
        if let guidance, let g1 = guidanceLinear1, let g2 = guidanceLinear2 {
            let g = Self.timestepEmbedding(guidance.asType(.float32), dim: inChannels)
            emb = emb + g2(silu(g1(g)))
        }
        return emb
    }
}

// MARK: - Modulation

/// Flux2Modulation — SiLU → Linear(dim → dim·3·sets) → grouped (shift, scale, gate) per set.
public final class Flux2Modulation: Module {
    let modParamSets: Int
    @ModuleInfo var linear: Linear

    public init(dim: Int, modParamSets: Int = 2) {
        self.modParamSets = modParamSets
        self._linear.wrappedValue = Linear(dim, dim * 3 * modParamSets, bias: false)
    }

    /// Returns `modParamSets` tuples of (shift, scale, gate), each [B,1,dim].
    public func callAsFunction(_ temb: MLXArray) -> [(shift: MLXArray, scale: MLXArray, gate: MLXArray)] {
        var mod = linear(silu(temb))
        if mod.ndim == 2 { mod = mod[0..., .newAxis, 0...] }
        let parts = split(mod, parts: 3 * modParamSets, axis: -1)
        return (0..<modParamSets).map { i in
            (parts[3 * i], parts[3 * i + 1], parts[3 * i + 2])
        }
    }
}

// MARK: - Attention helpers (mflux AttentionUtils)

public enum AttentionUtils {
    /// UltraTex: the active attention backend.
    public nonisolated(unsafe) static var backend: AttentionBackend = .dense
    /// UltraTex: query-row blocks used by the last chunked call (for reporting).
    public nonisolated(unsafe) static var lastChunkCount = 1

    /// [B,S,dim] → q,k,v each [B,H,S,D] with per-head RMSNorm (fp32) on q,k.
    static func processQKV(
        _ hidden: MLXArray, toQ: Linear, toK: Linear, toV: Linear,
        normQ: RMSNorm, normK: RMSNorm, heads: Int, headDim: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (B, S) = (hidden.shape[0], hidden.shape[1])
        func reshapeHeads(_ x: MLXArray) -> MLXArray {
            x.reshaped(B, S, heads, headDim).transposed(0, 2, 1, 3)
        }
        var q = reshapeHeads(toQ(hidden))
        var k = reshapeHeads(toK(hidden))
        let v = reshapeHeads(toV(hidden))
        q = normQ(q.asType(.float32)).asType(q.dtype)
        k = normK(k.asType(.float32)).asType(k.dtype)
        return (q, k, v)
    }

    /// SDPA + merge heads → [B, S, H*D].
    static func computeAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                                 heads: Int, headDim: Int) -> MLXArray {
        let scale = 1.0 / Float(q.shape[q.ndim - 1]).squareRoot()
        let out: MLXArray
        switch backend {
        case .dense:
            out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        case .denseChunked(let maxScoreBytes):
            // Rows of the output are independent, so chunking the queries is exact.
            let (b, h, s) = (q.shape[0], q.shape[1], q.shape[2])
            let bytesPerRow = b * h * s * 4                      // budget as if the scores were fp32
            let rows = max(1, min(s, maxScoreBytes / max(bytesPerRow, 1)))
            var parts: [MLXArray] = []
            var start = 0
            eval(q, k, v)
            while start < s {
                let end = min(start + rows, s)
                let part = MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., start ..< end, 0...], keys: k, values: v, scale: scale, mask: .none)
                eval(part)
                parts.append(part)
                start = end
            }
            lastChunkCount = parts.count
            out = parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
        }
        let B = out.shape[0]
        return out.transposed(0, 2, 1, 3).reshaped(B, -1, heads * headDim)
    }
}

// MARK: - Feed forward (SwiGLU)

public final class Flux2FeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    public init(dim: Int, mult: Float = KleinConfig.mlpRatio) {
        let inner = Int(Float(dim) * mult)
        self._linearIn.wrappedValue = Linear(dim, inner * 2, bias: false)
        self._linearOut.wrappedValue = Linear(inner, dim, bias: false)
    }

    static func swiglu(_ x: MLXArray) -> MLXArray {
        let parts = split(x, parts: 2, axis: -1)
        return silu(parts[0]) * parts[1]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearOut(Self.swiglu(linearIn(x)))
    }
}

// MARK: - Double-stream attention (joint img + txt)

public final class Flux2Attention: Module {
    let heads: Int
    let dimHead: Int
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    public init(dim: Int, heads: Int, dimHead: Int, addedKVProjDim: Int) {
        self.heads = heads
        self.dimHead = dimHead
        let inner = heads * dimHead
        self._toQ.wrappedValue = Linear(dim, inner, bias: false)
        self._toK.wrappedValue = Linear(dim, inner, bias: false)
        self._toV.wrappedValue = Linear(dim, inner, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._toOut.wrappedValue = Linear(inner, dim, bias: false)
        self._normAddedQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._normAddedK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._addQProj.wrappedValue = Linear(addedKVProjDim, inner, bias: false)
        self._addKProj.wrappedValue = Linear(addedKVProjDim, inner, bias: false)
        self._addVProj.wrappedValue = Linear(addedKVProjDim, inner, bias: false)
        self._toAddOut.wrappedValue = Linear(inner, dim, bias: false)
    }

    /// Returns (img_out, txt_out). Sequence order in attention is [txt, img].
    public func callAsFunction(
        _ hidden: MLXArray, encoder: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        var (q, k, v) = AttentionUtils.processQKV(
            hidden, toQ: toQ, toK: toK, toV: toV, normQ: normQ, normK: normK,
            heads: heads, headDim: dimHead)
        let (eq, ek, ev) = AttentionUtils.processQKV(
            encoder, toQ: addQProj, toK: addKProj, toV: addVProj,
            normQ: normAddedQ, normK: normAddedK, heads: heads, headDim: dimHead)
        q = concatenated([eq, q], axis: 2)   // [txt, img] on seq axis
        k = concatenated([ek, k], axis: 2)
        v = concatenated([ev, v], axis: 2)

        (q, k) = applyRopeBSHD(q, k, cos: cos, sin: sin)
        let attn = AttentionUtils.computeAttention(q, k, v, heads: heads, headDim: dimHead)

        let txtLen = encoder.shape[1]
        let encOut = toAddOut(attn[0..., ..<txtLen, 0...])
        let imgOut = toOut(attn[0..., txtLen..., 0...])
        return (imgOut, encOut)
    }
}

// MARK: - Double-stream block

public final class Flux2TransformerBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo(key: "norm1_context") var norm1Context: LayerNorm
    @ModuleInfo var attn: Flux2Attention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var ff: Flux2FeedForward
    @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm
    @ModuleInfo(key: "ff_context") var ffContext: Flux2FeedForward

    public init(dim: Int, heads: Int, headDim: Int, mlpRatio: Float) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._norm1Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._attn.wrappedValue = Flux2Attention(dim: dim, heads: heads, dimHead: headDim, addedKVProjDim: dim)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._ff.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
        self._norm2Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._ffContext.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
    }

    public func callAsFunction(
        _ hidden: MLXArray, encoder: MLXArray,
        modImg: [(shift: MLXArray, scale: MLXArray, gate: MLXArray)],
        modTxt: [(shift: MLXArray, scale: MLXArray, gate: MLXArray)],
        cos: MLXArray, sin: MLXArray
    ) -> (encoder: MLXArray, hidden: MLXArray) {
        let (msaI, mlpI) = (modImg[0], modImg[1])
        let (msaT, mlpT) = (modTxt[0], modTxt[1])
        var hidden = hidden
        var encoder = encoder

        let nh = (1 + msaI.scale) * norm1(hidden) + msaI.shift
        let ne = (1 + msaT.scale) * norm1Context(encoder) + msaT.shift
        let (attnOut, encAttnOut) = attn(nh, encoder: ne, cos: cos, sin: sin)
        hidden = hidden + msaI.gate * attnOut
        encoder = encoder + msaT.gate * encAttnOut

        let nh2 = (1 + mlpI.scale) * norm2(hidden) + mlpI.shift
        hidden = hidden + mlpI.gate * ff(nh2)
        let ne2 = (1 + mlpT.scale) * norm2Context(encoder) + mlpT.shift
        encoder = encoder + mlpT.gate * ffContext(ne2)
        return (encoder, hidden)
    }
}

// MARK: - Single-stream block (parallel self-attention + fused MLP)

public final class Flux2ParallelSelfAttention: Module {
    let heads: Int
    let dimHead: Int
    let innerDim: Int
    let mlpHiddenDim: Int
    @ModuleInfo(key: "to_qkv_mlp_proj") var toQKVMLP: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear

    public init(dim: Int, heads: Int, dimHead: Int, mlpRatio: Float) {
        self.heads = heads
        self.dimHead = dimHead
        self.innerDim = heads * dimHead
        self.mlpHiddenDim = Int(Float(dim) * mlpRatio)
        self._toQKVMLP.wrappedValue = Linear(dim, innerDim * 3 + mlpHiddenDim * 2, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._toOut.wrappedValue = Linear(innerDim + mlpHiddenDim, dim, bias: false)
    }

    public func callAsFunction(_ hidden: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let proj = toQKVMLP(hidden)
        let qkvMLP = split(proj, indices: [innerDim * 3], axis: -1)
        let qkv = qkvMLP[0]
        var mlpHidden = qkvMLP[1]
        let parts = split(qkv, parts: 3, axis: -1)
        let (B, S) = (hidden.shape[0], hidden.shape[1])
        func heads4(_ x: MLXArray) -> MLXArray { x.reshaped(B, S, heads, dimHead).transposed(0, 2, 1, 3) }
        var q = heads4(parts[0]); var k = heads4(parts[1]); let v = heads4(parts[2])
        q = normQ(q.asType(.float32)).asType(q.dtype)
        k = normK(k.asType(.float32)).asType(k.dtype)
        (q, k) = applyRopeBSHD(q, k, cos: cos, sin: sin)
        var attn = AttentionUtils.computeAttention(q, k, v, heads: heads, headDim: dimHead)
        mlpHidden = Flux2FeedForward.swiglu(mlpHidden)
        attn = concatenated([attn, mlpHidden], axis: -1)
        return outProjected(attn)
    }

    /// mlx-swift ≤0.31.6 JIT-compiles the steel split-K NAX GEMM (dispatched at
    /// M·N ≥ 2048², K ≥ 10240, K ≥ 3·max(M,N), HALF PRECISION ONLY) with the wrong
    /// dtype (mlx#3797) → bf16 garbage/NaN on M5-class GPUs. This to_out
    /// (K = innerDim 3072 + mlpHidden 9216 = 12288, N = 3072) is in the window at
    /// M = txt(512 padded)+img tokens ∈ [1366, 4096] — probe-confirmed NaN/1e21 at
    /// the exact production shape [1, 2816, 12288] in bf16 (768² t2i). 512²–896² t2i
    /// and 512²-class edits (packed target+ref) all land in-band; 1024² t2i
    /// (M = 4608) sits just above it.
    ///
    /// Klein is currently SHIELDED by accident, not by design: temb is computed in
    /// fp32 and MLX promotion (fp32 mod × bf16 stream → fp32) makes the whole
    /// single-block activation stream fp32, which the half-precision-only window
    /// never dispatches. This chunk is therefore dormant armor — it engages
    /// automatically if the activation flow ever becomes true bf16 (e.g. a
    /// Boogu-style bf16-activation speedup). Chunking is mathematically exact
    /// (output rows independent); same workaround as mage-flow-swift's
    /// downProjected. Fixed upstream in ml-explore/mlx#3810 (2026-07-07); on any
    /// mlx-swift bump run `klein-cli --nax-probe` and on PASS remove this chunk
    /// together with the mage-flow-swift / qwen3vl-mlx-swift / boogu-image-swift
    /// ones (boogu-image-swift/tools/check_mlx_swift_3810.sh checks whether a tag
    /// vendors mlx ≥ a8c3e9c).
    func outProjected(_ x: MLXArray) -> MLXArray {
        // KLEIN_NO_CHUNK disables the workaround — for validating a fixed
        // mlx-swift (run `klein-cli --nax-probe` first).
        // UltraTex: off unless ULTRATEX_ROW_CHUNK is set; mlx-swift >= 0.32.3 carries the upstream fix.
        if ProcessInfo.processInfo.environment["ULTRATEX_ROW_CHUNK"] == nil { return toOut(x) }
        let tokens = x.dim(-2)
        let rowLimit = 896
        guard x.dtype != .float32, tokens > rowLimit else { return toOut(x) }
        var parts: [MLXArray] = []
        var start = 0
        while start < tokens {
            let end = min(start + rowLimit, tokens)
            parts.append(toOut(x[.ellipsis, start ..< end, 0...]))
            start = end
        }
        return concatenated(parts, axis: -2)
    }
}

public final class Flux2SingleTransformerBlock: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var attn: Flux2ParallelSelfAttention

    public init(dim: Int, heads: Int, headDim: Int, mlpRatio: Float) {
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._attn.wrappedValue = Flux2ParallelSelfAttention(dim: dim, heads: heads, dimHead: headDim, mlpRatio: mlpRatio)
    }

    public func callAsFunction(
        _ hidden: MLXArray, mod: (shift: MLXArray, scale: MLXArray, gate: MLXArray),
        cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let nh = (1 + mod.scale) * norm(hidden) + mod.shift
        return hidden + mod.gate * attn(nh, cos: cos, sin: sin)
    }
}

// MARK: - AdaLayerNormContinuous (norm_out)

public final class AdaLayerNormContinuous: Module {
    let embeddingDim: Int
    @ModuleInfo var linear: Linear
    @ModuleInfo var norm: LayerNorm

    public init(embeddingDim: Int, conditioningEmbeddingDim: Int) {
        self.embeddingDim = embeddingDim
        self._linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim * 2, bias: false)
        self._norm.wrappedValue = LayerNorm(dimensions: embeddingDim, eps: 1e-6, affine: false)
    }

    public func callAsFunction(_ x: MLXArray, _ temb: MLXArray) -> MLXArray {
        let emb = linear(silu(temb))
        let scale = emb[0..., ..<embeddingDim]
        let shift = emb[0..., embeddingDim...]
        return norm(x) * (1 + scale)[0..., .newAxis, 0...] + shift[0..., .newAxis, 0...]
    }
}

// MARK: - Full transformer

public final class Flux2Transformer: Module {
    public let outChannels: Int
    let innerDim: Int

    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2TimestepGuidanceEmbeddings
    @ModuleInfo(key: "double_stream_modulation_img") var dsModImg: Flux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var dsModTxt: Flux2Modulation
    @ModuleInfo(key: "single_stream_modulation") var ssMod: Flux2Modulation
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Flux2TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux2SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let posEmbed: Flux2PosEmbed

    /// UltraTex: when set, the time embedding is cast to this dtype so the activation stream stays in it
    /// (bf16 weights with an fp32 time embedding otherwise promote every activation to fp32).
    public var streamDType: DType? = nil

    /// UltraTex: called after every block with its name ("double_3", "single_11") and outputs: the image and
    /// text streams of a double block, or the joint [text, image] sequence of a single block. For parity gates.
    public var blockTap: ((String, MLXArray, MLXArray?) -> Void)? = nil

    /// UltraTex: a klein DiT of the given size.
    public convenience init(config: DiTConfig) {
        self.init(numLayers: config.numLayers, numSingleLayers: config.numSingleLayers, numHeads: config.numHeads,
                  jointAttentionDim: config.jointAttentionDim)
    }

    public init(
        inChannels: Int = KleinConfig.inChannels,
        outChannels: Int? = nil,
        numLayers: Int = KleinConfig.numLayers,
        numSingleLayers: Int = KleinConfig.numSingleLayers,
        numHeads: Int = KleinConfig.numHeads,
        headDim: Int = KleinConfig.headDim,
        jointAttentionDim: Int = KleinConfig.jointAttentionDim,
        mlpRatio: Float = KleinConfig.mlpRatio,
        patchSize: Int = KleinConfig.patchSize,
        guidanceEmbeds: Bool = false
    ) {
        let out = outChannels ?? inChannels
        let inner = numHeads * headDim
        self.outChannels = out
        self.innerDim = inner
        self.posEmbed = Flux2PosEmbed()
        self._timeGuidanceEmbed.wrappedValue = Flux2TimestepGuidanceEmbeddings(
            inChannels: KleinConfig.timestepGuidanceChannels, embeddingDim: inner,
            guidanceEmbeds: guidanceEmbeds)
        self._dsModImg.wrappedValue = Flux2Modulation(dim: inner, modParamSets: 2)
        self._dsModTxt.wrappedValue = Flux2Modulation(dim: inner, modParamSets: 2)
        self._ssMod.wrappedValue = Flux2Modulation(dim: inner, modParamSets: 1)
        self._xEmbedder.wrappedValue = Linear(inChannels, inner, bias: false)
        self._contextEmbedder.wrappedValue = Linear(jointAttentionDim, inner, bias: false)
        self._transformerBlocks.wrappedValue = (0..<numLayers).map { _ in
            Flux2TransformerBlock(dim: inner, heads: numHeads, headDim: headDim, mlpRatio: mlpRatio)
        }
        self._singleTransformerBlocks.wrappedValue = (0..<numSingleLayers).map { _ in
            Flux2SingleTransformerBlock(dim: inner, heads: numHeads, headDim: headDim, mlpRatio: mlpRatio)
        }
        self._normOut.wrappedValue = AdaLayerNormContinuous(embeddingDim: inner, conditioningEmbeddingDim: inner)
        self._projOut.wrappedValue = Linear(inner, patchSize * patchSize * out, bias: false)
    }

    /// hidden: [B, imgLen, inCh]; encoder: [B, txtLen, jointDim]; timestep/guidance: [B];
    /// imgIds/txtIds: [len, 4]. Returns [B, imgLen, patch²·outCh].
    public func callAsFunction(
        _ hidden: MLXArray, encoder: MLXArray,
        timestep: MLXArray, guidance: MLXArray?,
        imgIds: MLXArray, txtIds: MLXArray
    ) -> MLXArray {
        // timestep/guidance auto-scale ×1000 when ≤1 (verbatim from mflux).
        var t = timestep.asType(.float32)
        let tScale: Float = t.max().item(Float.self) <= 1.0 ? 1000.0 : 1.0
        t = t * tScale
        var g: MLXArray? = nil
        if let guidance {
            var gg = guidance.asType(.float32)
            let gScale: Float = gg.max().item(Float.self) <= 1.0 ? 1000.0 : 1.0
            gg = gg * gScale
            g = gg
        }
        var temb = timeGuidanceEmbed(t, g)
        if let streamDType { temb = temb.asType(streamDType) }

        var hidden = xEmbedder(hidden)
        var encoder = contextEmbedder(encoder)

        let (imgCos, imgSin) = posEmbed(imgIds)
        let (txtCos, txtSin) = posEmbed(txtIds)
        let cos = concatenated([txtCos, imgCos], axis: 0)   // [txt, img]
        let sin = concatenated([txtSin, imgSin], axis: 0)

        let modImg = dsModImg(temb)
        let modTxt = dsModTxt(temb)
        for (i, block) in transformerBlocks.enumerated() {
            (encoder, hidden) = block(hidden, encoder: encoder, modImg: modImg, modTxt: modTxt, cos: cos, sin: sin)
            blockTap?("double_\(i)", hidden, encoder)
        }

        var combined = concatenated([encoder, hidden], axis: 1)
        let modSingle = ssMod(temb)[0]
        for (i, block) in singleTransformerBlocks.enumerated() {
            combined = block(combined, mod: modSingle, cos: cos, sin: sin)
            blockTap?("single_\(i)", combined, nil)
        }

        var out = combined[0..., encoder.shape[1]..., 0...]
        out = normOut(out, temb)
        out = projOut(out)
        return out
    }
}
