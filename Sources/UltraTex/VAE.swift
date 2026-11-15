// The FLUX.2 autoencoder as UltraTex uses it: the stock encoder for the normal maps and the reference, and the
// foreground-aware decoder UltraTex fine-tuned (converted to the diffusers layout by
// DEV/UltraTexOracle/tools/convert_decoder.py). Each view is encoded and decoded on its own, as upstream does.
// The network itself is the fleet's Flux2VAE package; this file only adapts layouts.
//
// Every call goes through that package's row-band path. Untiled, one 2048² view costs a 17 GB attention score
// matrix to encode and about 42 GB to decode; in row bands the floor is about 4 GB and 7 GB. The banded result
// equals the untiled one up to float reassociation.

import Flux2VAE
import Foundation
import MLX

public final class UltraTexVAE {
    public let encoder: Flux2VAEEncoder
    public let decoder: Flux2VAE
    /// Row-band size for every encode and decode.
    public var tile: Flux2VAETile = .standard

    public init(encoder: Flux2VAEEncoder, decoder: Flux2VAE) {
        self.encoder = encoder
        self.decoder = decoder
    }

    /// `directory` is a diffusers `vae/` folder: the stock one, or the converted one with UltraTex's decoder.
    /// float32, as upstream runs its autoencoder. `exactConvolutions` takes the decoder's exact 3x3 route (for
    /// parity gates) instead of the faster default.
    public static func load(directory: URL, exactConvolutions: Bool = false) throws -> UltraTexVAE {
        let vae = UltraTexVAE(encoder: try Flux2VAEWeights.loadEncoder(directory: directory, dtype: .float32),
                              decoder: try Flux2VAEWeights.loadVAE(directory: directory, dtype: .float32))
        if exactConvolutions { vae.decoder.convRoute = .conv3d }
        return vae
    }

    /// One view [1, 3, R, R] in [-1, 1] -> its latent as BFL's `AutoEncoder.encode` gives it: [1, 128, R/16, R/16].
    public func encode(_ image: MLXArray) -> MLXArray {
        let z = encoder.encodePacked(image.asType(.float32), tile: tile)
        eval(z)
        MLX.Memory.clearCache()
        return z
    }

    /// The latent of one view as tokens, row-major: [(R/16)^2, 128].
    public func encodeTokens(_ image: MLXArray) -> MLXArray {
        let z = encode(image)
        return z[0].transposed(1, 2, 0).reshaped(z.dim(2) * z.dim(3), z.dim(1))
    }

    /// One view's latent [1, 128, h, w] -> the picture [1, 3, 16h, 16w], not clamped.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let out = decoder.decodePackedLatents(latent.asType(.float32), tile: tile)
        eval(out)
        MLX.Memory.clearCache()
        return out
    }

    /// One tile of the latent canvas [h, w, 128] -> the picture [16h, 16w, 3] in 0...255 (uint8), as upstream
    /// converts it: clamp to [-1, 1], then 127.5 * (x + 1), truncated.
    public func decodeTile(_ tile: MLXArray) -> MLXArray {
        let out = decode(tile.transposed(2, 0, 1).expandedDimensions(axis: 0))[0].transposed(1, 2, 0)
        return (clip(out, min: -1, max: 1) + 1) * 127.5
    }
}
