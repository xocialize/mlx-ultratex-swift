// From rendered views and a reference picture to what the model consumes, the way upstream's dataset does it
// (ultratex/data/dataset.py @ a726678, which uses Pillow and OpenCV):
//
//   - a view or the reference is resized to the working resolution R with Pillow's BILINEAR filter (an RGBA picture
//     is premultiplied, resampled in two 8-bit passes, horizontal then vertical, and un-premultiplied), then pasted
//     on white through its alpha;
//   - a token mask is the alpha resized to R/16 the same way, thresholded at > 0, and dilated by a 2x2 box;
//   - the six views are tiled 2x3 as views 0, 1, 3 over 2, 4, 5.
//
// The arithmetic below follows Pillow's fixed-point code (Resample.c, Convert.c, Paste.c) so that a Swift run sees
// the same bytes the CUDA oracle did; that is what the gate checks.

import Foundation
import MLX

public enum UltraTexInputs {

    // MARK: Pillow-compatible pieces

    /// Pillow's coefficients for one axis of a BILINEAR resize from `inSize` to `outSize`.
    static func coefficients(inSize: Int, outSize: Int) -> (bounds: [(Int, Int)], weights: [[Int]]) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = 1.0 * filterScale
        var bounds: [(Int, Int)] = [], weights: [[Int]] = []
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            let xmin = max(0, Int(center - support + 0.5))
            let xmax = min(inSize, Int(center + support + 0.5))
            var k = [Double](repeating: 0, count: xmax - xmin)
            var total = 0.0
            for x in 0 ..< k.count {
                let d = abs((Double(x + xmin) - center + 0.5) / filterScale)
                k[x] = d < 1 ? 1 - d : 0
                total += k[x]
            }
            bounds.append((xmin, k.count))
            weights.append(k.map { w in
                let v = total != 0 ? w / total : w
                return Int(v * Double(1 << 22) + (v < 0 ? -0.5 : 0.5))            // PRECISION_BITS = 32 - 8 - 2
            })
        }
        return (bounds, weights)
    }

    @inline(__always) static func clip8(_ v: Int) -> UInt8 { UInt8(max(0, min(255, v))) }

    /// Pillow's two-pass 8-bit BILINEAR resize of an interleaved image with `channels` bands.
    static func resize(_ src: [UInt8], width: Int, height: Int, channels: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        if width == toWidth, height == toHeight { return src }
        var current = src, w = width
        if width != toWidth {
            let (bounds, weights) = coefficients(inSize: width, outSize: toWidth)
            var out = [UInt8](repeating: 0, count: height * toWidth * channels)
            for y in 0 ..< height {
                for xx in 0 ..< toWidth {
                    let (xmin, n) = bounds[xx]
                    let k = weights[xx]
                    for c in 0 ..< channels {
                        var ss = 1 << 21
                        for x in 0 ..< n { ss += Int(current[(y * width + xmin + x) * channels + c]) * k[x] }
                        out[(y * toWidth + xx) * channels + c] = clip8(ss >> 22)
                    }
                }
            }
            current = out
            w = toWidth
        }
        if height != toHeight {
            let (bounds, weights) = coefficients(inSize: height, outSize: toHeight)
            var out = [UInt8](repeating: 0, count: toHeight * w * channels)
            for yy in 0 ..< toHeight {
                let (ymin, n) = bounds[yy]
                let k = weights[yy]
                for x in 0 ..< w {
                    for c in 0 ..< channels {
                        var ss = 1 << 21
                        for y in 0 ..< n { ss += Int(current[((ymin + y) * w + x) * channels + c]) * k[y] }
                        out[(yy * w + x) * channels + c] = clip8(ss >> 22)
                    }
                }
            }
            current = out
        }
        return current
    }

    /// Pillow's MULDIV255 / DIV255: a * b / 255, rounded.
    @inline(__always) static func mulDiv255(_ a: Int, _ b: Int) -> Int {
        let t = a * b + 128
        return ((t >> 8) + t) >> 8
    }

    /// `Image.resize(size, BILINEAR)` on a straight-alpha RGBA picture: RGBA -> RGBa, resample, RGBa -> RGBA.
    static func resizeRGBA(_ rgba: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        if width == toWidth, height == toHeight { return rgba }
        var pre = rgba
        for p in 0 ..< width * height {
            let a = Int(rgba[p * 4 + 3])
            for c in 0 ..< 3 { pre[p * 4 + c] = UInt8(mulDiv255(Int(rgba[p * 4 + c]), a)) }
        }
        var out = resize(pre, width: width, height: height, channels: 4, toWidth: toWidth, toHeight: toHeight)
        for p in 0 ..< toWidth * toHeight {
            let a = Int(out[p * 4 + 3])
            if a == 255 || a == 0 { continue }
            for c in 0 ..< 3 { out[p * 4 + c] = clip8(255 * Int(out[p * 4 + c]) / a) }
        }
        return out
    }

    /// `Image.new("RGB", size, white).paste(img, mask=alpha)`: interleaved RGB.
    static func onWhite(_ rgba: [UInt8], pixels: Int) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: pixels * 3)
        for p in 0 ..< pixels {
            let a = Int(rgba[p * 4 + 3])
            for c in 0 ..< 3 {
                let t = 255 * (255 - a) + Int(rgba[p * 4 + c]) * a + 128
                rgb[p * 3 + c] = UInt8(((t >> 8) + t) >> 8)
            }
        }
        return rgb
    }

    /// `(alpha > 0)` dilated by OpenCV's 2x2 box (anchor at its lower-right cell): a cell is set when it, its left,
    /// upper or upper-left neighbour is.
    static func tokenMask(alpha: [UInt8], side: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: side * side)
        for y in 0 ..< side {
            for x in 0 ..< side {
                var on = false
                for dy in -1 ... 0 where y + dy >= 0 {
                    for dx in -1 ... 0 where x + dx >= 0 && alpha[(y + dy) * side + x + dx] > 0 { on = true }
                }
                out[y * side + x] = on ? 1 : 0
            }
        }
        return out
    }

    static func channel(_ interleaved: [UInt8], _ index: Int, of channels: Int) -> [UInt8] {
        stride(from: index, to: interleaved.count, by: channels).map { interleaved[$0] }
    }

    // MARK: What the pipeline consumes

    public struct Prepared {
        /// The six normal maps on white, [-1, 1], one per view: [1, 3, R, R] float32, in view order 0...5.
        public var normalViews: [MLXArray]
        /// The reference on white, [-1, 1]: [1, 3, R, R] float32.
        public var reference: MLXArray
        /// Token masks, 0/1: the 2x3 canvas [2R/16, 3R/16] (target and normals share it) and the reference [R/16, R/16].
        public var canvasMask: MLXArray
        public var referenceMask: MLXArray
        /// Per view, the alpha at the working resolution (R * R bytes): where it is 0 the output is painted white.
        public var viewAlpha: [[UInt8]]
        public var resolution: Int
    }

    static func tensor(_ rgb: [UInt8], side: Int) -> MLXArray {
        // ToTensor then Normalize(0.5, 0.5): x / 255 * 2 - 1, as float32
        (MLXArray(rgb, [side, side, 3]).asType(.float32) / 255 - 0.5) / 0.5
    }

    /// `views` are the six rendered normal maps and `reference` the reference picture, both straight-alpha RGBA8.
    public static func prepare(views: [UltraTexImage], reference: UltraTexImage, resolution r: Int) -> Prepared {
        precondition(views.count == 6 && r % 16 == 0)
        let small = r / 16
        var tensors: [MLXArray] = [], masks: [[UInt8]] = [], alphas: [[UInt8]] = []
        for v in views {
            let big = resizeRGBA(v.rgba, width: v.width, height: v.height, toWidth: r, toHeight: r)
            alphas.append(channel(big, 3, of: 4))
            tensors.append(tensor(onWhite(big, pixels: r * r), side: r).transposed(2, 0, 1).expandedDimensions(axis: 0))
            let tiny = resizeRGBA(v.rgba, width: v.width, height: v.height, toWidth: small, toHeight: small)
            masks.append(tokenMask(alpha: channel(tiny, 3, of: 4), side: small))
        }
        // Tile the masks 2x3: views 0, 1, 3 over 2, 4, 5.
        let order = [[0, 1, 3], [2, 4, 5]]
        var canvas = [UInt8](repeating: 0, count: 2 * small * 3 * small)
        for (row, viewsInRow) in order.enumerated() {
            for (col, view) in viewsInRow.enumerated() {
                for y in 0 ..< small {
                    for x in 0 ..< small {
                        canvas[(row * small + y) * 3 * small + col * small + x] = masks[view][y * small + x]
                    }
                }
            }
        }
        let ref = resizeRGBA(reference.rgba, width: reference.width, height: reference.height, toWidth: r, toHeight: r)
        let refAlpha = channel(ref, 3, of: 4)
        let refSmall = resize(refAlpha, width: r, height: r, channels: 1, toWidth: small, toHeight: small)
        return Prepared(
            normalViews: tensors,
            reference: tensor(onWhite(ref, pixels: r * r), side: r).transposed(2, 0, 1).expandedDimensions(axis: 0),
            canvasMask: MLXArray(canvas, [2 * small, 3 * small]),
            referenceMask: MLXArray(tokenMask(alpha: refSmall, side: small), [small, small]),
            viewAlpha: alphas, resolution: r)
    }

    /// The six view tensors as upstream's canvas [3, 2R, 3R] (views 0, 1, 3 over 2, 4, 5), for comparison.
    public static func canvas(_ views: [MLXArray]) -> MLXArray {
        let top = concatenated([views[0], views[1], views[3]], axis: 3), bottom = concatenated([views[2], views[4], views[5]], axis: 3)
        return concatenated([top, bottom], axis: 2)[0]
    }
}
