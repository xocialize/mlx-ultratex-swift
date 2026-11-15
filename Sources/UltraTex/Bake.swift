// Six generated views onto the mesh's own texture atlas.
//
// A back-projection bake. Upstream publishes none (AB-R-0874), so this follows the tool used to judge the oracle
// runs (DEV/UltraTexOracle/tools/bake.py) and adds the two things that tool lacks: a fill for texels no view
// reaches and padding around the charts.
//
//   1. Rasterize the atlas: every texel inside a triangle's UV footprint gets that triangle, a 3D point and a
//      smooth normal.
//   2. Per view: project the point; keep it if the view's depth buffer says nothing is in front of it, if it faces
//      the camera, and if it is not on the silhouette (there the generated pixel is mixed with the white
//      background). Sample the view bilinearly and weight by cos(angle)^power.
//   3. Blend by weight.
//   4. Fill: a texel no view reached takes the colour of the nearest painted texels in 3D (inverse-distance
//      weighted), so the inside of a thin shell takes its outside, not the mesh's old texture. On the corpus dress
//      a third of the atlas is in that state (AB-R-0880).
//   5. Pad: texels outside every chart take their neighbours' colour, so filtering does not pull in black.
//
// CPU only: no GPU and no MLX memory.

import Foundation

public enum UltraTexBake {
    public struct Result: Sendable {
        /// The atlas, opaque RGBA8, `size` square, row 0 at v = 0 (glTF).
        public var atlas: UltraTexImage
        /// Where a texel's colour came from: 0 outside every chart, 1 painted by a view, 2 filled from neighbours.
        public var origin: [UInt8]
        public var texelsOnMesh: Int
        public var paintedFraction: Double
        public var filledFraction: Double
        /// Per view, the fraction of on-mesh texels it painted.
        public var perView: [Double]
    }

    /// `views` are the six generated pictures in view order, any square size.
    public static func bake(mesh: UltraTexMesh, views: [UltraTexImage], atlasSize size: Int = 2048,
                            distance: Double = UltraTexRenderer.defaultDistance, power: Double = 4,
                            depthTolerance: Double = 3e-3) -> Result {
        precondition(views.count == 6 && !mesh.uvs.isEmpty, "the bake needs six views and a mesh with UVs")
        let total = size * size

        // 1. The atlas: which triangle owns each texel, and where on it.
        var face = [Int32](repeating: -1, count: total)
        var b1 = [Float](repeating: 0, count: total), b2 = b1
        let scale = Double(size)
        for (index, t) in mesh.faces.enumerated() {
            let (p0, p1, p2) = (mesh.uvs[Int(t.x)] * scale, mesh.uvs[Int(t.y)] * scale, mesh.uvs[Int(t.z)] * scale)
            let area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)
            if area == 0 { continue }
            let minX = max(0, Int((min(p0.x, p1.x, p2.x) - 0.5).rounded(.up))), maxX = min(size - 1, Int((max(p0.x, p1.x, p2.x) - 0.5).rounded(.down)))
            let minY = max(0, Int((min(p0.y, p1.y, p2.y) - 0.5).rounded(.up))), maxY = min(size - 1, Int((max(p0.y, p1.y, p2.y) - 0.5).rounded(.down)))
            if minX > maxX || minY > maxY { continue }
            for row in minY ... maxY {
                let y = Double(row) + 0.5
                for col in minX ... maxX {
                    let x = Double(col) + 0.5
                    let l1 = ((x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (y - p0.y)) / area
                    let l2 = ((p1.x - p0.x) * (y - p0.y) - (x - p0.x) * (p1.y - p0.y)) / area
                    if l1 < 0 || l2 < 0 || l1 + l2 > 1 { continue }
                    let at = row * size + col
                    face[at] = Int32(index); b1[at] = Float(l1); b2[at] = Float(l2)
                }
            }
        }
        let texels = (0 ..< total).filter { face[$0] >= 0 }
        var position = [SIMD3<Double>](repeating: .zero, count: texels.count), normal = position
        for (i, at) in texels.enumerated() {
            let t = mesh.faces[Int(face[at])]
            let (w1, w2) = (Double(b1[at]), Double(b2[at])), w0 = 1 - w1 - w2
            position[i] = mesh.positions[Int(t.x)] * w0 + mesh.positions[Int(t.y)] * w1 + mesh.positions[Int(t.z)] * w2
            var n = mesh.normals[Int(t.x)] * w0 + mesh.normals[Int(t.y)] * w1 + mesh.normals[Int(t.z)] * w2
            n /= max((n * n).sum().squareRoot(), 1e-9)
            normal[i] = n
        }

        // 2-3. Views, in parallel; each returns its weighted colours.
        let cameras = UltraTexRenderer.cameras(distance: distance)
        let bufferSize = 2048
        let lock = NSLock()
        var sums = [SIMD3<Double>](repeating: .zero, count: texels.count)
        var weights = [Double](repeating: 0, count: texels.count)
        var perView = [Double](repeating: 0, count: 6)
        DispatchQueue.concurrentPerform(iterations: 6) { v in
            let camera = cameras[v], view = views[v]
            let raster = UltraTexRenderer.rasterize(mesh, camera: camera, size: bufferSize)
            let toView = Double(view.width) / Double(bufferSize)
            let margin = 1.5 / toView                              // 1.5 view pixels, in buffer pixels
            var colour = [SIMD3<Double>](repeating: .zero, count: texels.count)
            var weight = [Double](repeating: 0, count: texels.count)
            var painted = 0
            func covered(_ x: Double, _ y: Double) -> Bool {
                let (ix, iy) = (Int(x.rounded(.down)), Int(y.rounded(.down)))
                return ix >= 0 && iy >= 0 && ix < bufferSize && iy < bufferSize && raster.face[iy * bufferSize + ix] >= 0
            }
            for i in 0 ..< texels.count {
                let toCamera = camera.position - position[i]
                let cosine = (normal[i] * toCamera).sum() / (toCamera * toCamera).sum().squareRoot()
                if cosine <= 0.05 { continue }
                let p = UltraTexRenderer.project(position[i], camera: camera, size: bufferSize)
                let (ix, iy) = (Int(p.x.rounded(.down)), Int(p.y.rounded(.down)))
                if ix < 0 || iy < 0 || ix >= bufferSize || iy >= bufferSize { continue }
                let nearest = raster.inverseDepth[iy * bufferSize + ix]
                if nearest == 0 || p.z > 1 / nearest + depthTolerance { continue }
                if !(covered(p.x - margin, p.y - margin) && covered(p.x + margin, p.y - margin)
                     && covered(p.x - margin, p.y + margin) && covered(p.x + margin, p.y + margin)) { continue }
                // bilinear sample of the view at the same point
                let (x, y) = (max(0, min(Double(view.width) - 1.001, p.x * toView - 0.5)), max(0, min(Double(view.height) - 1.001, p.y * toView - 0.5)))
                let (x0, y0) = (Int(x), Int(y)), (fx, fy) = (x - Double(x0), y - Double(y0))
                func px(_ xx: Int, _ yy: Int) -> SIMD3<Double> {
                    let o = (yy * view.width + xx) * 4
                    return SIMD3(Double(view.rgba[o]), Double(view.rgba[o + 1]), Double(view.rgba[o + 2]))
                }
                let sample = (px(x0, y0) * (1 - fx) + px(x0 + 1, y0) * fx) * (1 - fy) + (px(x0, y0 + 1) * (1 - fx) + px(x0 + 1, y0 + 1) * fx) * fy
                let w = pow(cosine, power)
                colour[i] = sample * w; weight[i] = w
                painted += 1
            }
            lock.lock()
            for i in 0 ..< texels.count where weight[i] > 0 { sums[i] += colour[i]; weights[i] += weight[i] }
            perView[v] = Double(painted) / Double(max(texels.count, 1))
            lock.unlock()
        }

        var rgb = [SIMD3<Double>](repeating: .zero, count: texels.count)
        var isPainted = [Bool](repeating: false, count: texels.count)
        var paintedCount = 0
        for i in 0 ..< texels.count where weights[i] > 1e-6 {
            rgb[i] = sums[i] / weights[i]; isPainted[i] = true; paintedCount += 1
        }

        // 4. Fill from the nearest painted texels in 3D, through a uniform grid over [-0.5, 0.5]^3.
        let cells = 96, cellSize = 1.0 / Double(cells)
        func cell(_ p: SIMD3<Double>) -> (Int, Int, Int) {
            func c(_ v: Double) -> Int { max(0, min(cells - 1, Int((v + 0.5) / cellSize))) }
            return (c(p.x), c(p.y), c(p.z))
        }
        var starts = [Int32](repeating: 0, count: cells * cells * cells + 1)
        for i in 0 ..< texels.count where isPainted[i] { let (a, b, c) = cell(position[i]); starts[(a * cells + b) * cells + c + 1] += 1 }
        for k in 1 ..< starts.count { starts[k] += starts[k - 1] }
        var cursor = starts, members = [Int32](repeating: 0, count: paintedCount)
        for i in 0 ..< texels.count where isPainted[i] {
            let (a, b, c) = cell(position[i]), k = (a * cells + b) * cells + c
            members[Int(cursor[k])] = Int32(i); cursor[k] += 1
        }
        var meanColour = SIMD3<Double>.zero
        for i in 0 ..< texels.count where isPainted[i] { meanColour += rgb[i] }
        if paintedCount > 0 { meanColour /= Double(paintedCount) }
        let unpainted = (0 ..< texels.count).filter { !isPainted[$0] }
        var filled = [SIMD3<Double>](repeating: meanColour, count: unpainted.count)
        DispatchQueue.concurrentPerform(iterations: unpainted.count) { u in
            let p = position[unpainted[u]]
            let (ca, cb, cc) = cell(p)
            var found: [(Double, Int32)] = []
            var radius = 0
            while radius <= 12 {
                for a in max(0, ca - radius) ... min(cells - 1, ca + radius) {
                    for b in max(0, cb - radius) ... min(cells - 1, cb + radius) {
                        for c in max(0, cc - radius) ... min(cells - 1, cc + radius) {
                            if max(abs(a - ca), abs(b - cb), abs(c - cc)) != radius { continue }      // this shell only
                            let k = (a * cells + b) * cells + c
                            for m in Int(starts[k]) ..< Int(starts[k + 1]) {
                                let j = members[m], d = position[Int(j)] - p
                                found.append(((d * d).sum(), j))
                            }
                        }
                    }
                }
                // A hit in shell r can be beaten only by points within r + 1 cells; one more shell settles it.
                if let best = found.map(\.0).min(), best.squareRoot() <= Double(radius) * cellSize { break }
                radius += 1
            }
            guard let best = found.map(\.0).min() else { return }
            var colour = SIMD3<Double>.zero, weight = 0.0
            for (d2, j) in found where d2 <= best * 4 + 1e-12 {
                let w = 1 / (d2 + 1e-8)
                colour += rgb[Int(j)] * w; weight += w
            }
            filled[u] = colour / weight
        }
        for (u, i) in unpainted.enumerated() { rgb[i] = filled[u] }

        // 5. Write the atlas and pad the charts outward.
        var rgba = [UInt8](repeating: 0, count: total * 4)
        var origin = [UInt8](repeating: 0, count: total)
        for (i, at) in texels.enumerated() {
            for k in 0 ..< 3 { rgba[at * 4 + k] = UInt8(max(0, min(255, rgb[i][k].rounded()))) }
            rgba[at * 4 + 3] = 255
            origin[at] = isPainted[i] ? 1 : 2
        }
        var known = origin.map { $0 != 0 }
        for _ in 0 ..< 16 {
            var next = known, out = rgba
            for y in 0 ..< size {
                for x in 0 ..< size where !known[y * size + x] {
                    var sum = SIMD3<Int>.zero, n = 0
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                        let (xx, yy) = (x + dx, y + dy)
                        if xx < 0 || yy < 0 || xx >= size || yy >= size || !known[yy * size + xx] { continue }
                        let o = (yy * size + xx) * 4
                        sum &+= SIMD3(Int(rgba[o]), Int(rgba[o + 1]), Int(rgba[o + 2])); n += 1
                    }
                    if n > 0 {
                        let o = (y * size + x) * 4
                        out[o] = UInt8(sum.x / n); out[o + 1] = UInt8(sum.y / n); out[o + 2] = UInt8(sum.z / n); out[o + 3] = 255
                        next[y * size + x] = true
                    }
                }
            }
            known = next; rgba = out
        }
        for at in 0 ..< total where !known[at] {                    // far from any chart: the mean, opaque
            for k in 0 ..< 3 { rgba[at * 4 + k] = UInt8(max(0, min(255, meanColour[k].rounded()))) }
            rgba[at * 4 + 3] = 255
        }
        return Result(atlas: UltraTexImage(width: size, height: size, rgba: rgba), origin: origin, texelsOnMesh: texels.count,
                      paintedFraction: Double(paintedCount) / Double(max(texels.count, 1)),
                      filledFraction: Double(unpainted.count) / Double(max(texels.count, 1)), perView: perView)
    }
}
