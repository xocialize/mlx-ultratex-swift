// The six views UltraTex is conditioned on: world-space normal maps of the mesh with alpha, from the canonical
// cameras. A CPU rasterizer: nothing here touches the GPU or MLX, so it costs no model memory.
//
// Camera rule (fitted in P0, AB-R-0874): six cameras at distance `distance` on -Y, -X, +Y, +X, +Z, -Z, looking
// at the origin (3x4 camera-to-world, Blender convention: the camera looks down its -Z, +Y is up), principal point
// at the centre, focal length 6400 px at 4096 px (35.49 deg). Normals are stored as rgb = 0.5 + 0.5 n in the
// camera world frame, which the pilot showed the model accepts (AB-R-0878).
//
// A pixel is covered when its centre lies inside a projected triangle; the nearest triangle wins and its vertex
// normals are interpolated with perspective-correct weights. That is the same sample the ray caster in
// DEV/UltraTexOracle/tools/render_gbuffer.py takes, so the two agree except where a ray grazes an edge.

import Foundation

public enum UltraTexRenderer {
    public static let focalAt4096 = 6400.0
    public static let defaultDistance = 1.60

    public struct Camera: Sendable {
        /// Camera axes in world coordinates and its position.
        public var x: SIMD3<Double>, y: SIMD3<Double>, z: SIMD3<Double>, position: SIMD3<Double>
    }

    /// The six canonical cameras, in upstream's view order.
    public static func cameras(distance: Double = defaultDistance) -> [Camera] {
        let axes: [(SIMD3<Double>, SIMD3<Double>, SIMD3<Double>)] = [
            (SIMD3(1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, -1, 0)),
            (SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(-1, 0, 0)),
            (SIMD3(-1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 1, 0)),
            (SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 0, 0)),
            (SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)),
            (SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, -1)),
        ]
        return axes.map { Camera(x: $0.0, y: $0.1, z: $0.2, position: $0.2 * distance) }
    }

    /// One view: RGBA8, row-major, `size * size * 4` bytes. Transparent black where nothing is hit.
    public struct View: Sendable {
        public var size: Int
        public var rgba: [UInt8]
        /// Fraction of pixels covered.
        public var coverage: Double
    }

    /// What a camera sees, per pixel: the nearest triangle, its perspective-correct weights and its depth.
    public struct Raster: Sendable {
        public var size: Int
        /// Index into `mesh.faces`, or -1 where nothing is hit.
        public var face: [Int32]
        /// Weights of the triangle's second and third vertex (the first is 1 - w1 - w2).
        public var w1: [Float]
        public var w2: [Float]
        /// 1 / depth along the camera's view axis; 0 where nothing is hit.
        public var inverseDepth: [Double]
    }

    /// Pixel coordinates and depth of a point for a camera of this size: (x, y, depth along the view axis).
    public static func project(_ point: SIMD3<Double>, camera: Camera, size: Int) -> SIMD3<Double> {
        let f = focalAt4096 * Double(size) / 4096, c = Double(size) / 2
        let d = point - camera.position
        let depth = -(d * camera.z).sum()
        return SIMD3(c + f * (d * camera.x).sum() / depth, c - f * (d * camera.y).sum() / depth, depth)
    }

    /// The world-space normal map of `mesh` seen from `camera`, at `size` x `size`.
    public static func normalMap(_ mesh: UltraTexMesh, camera: Camera, size: Int) -> View {
        let r = rasterize(mesh, camera: camera, size: size)
        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        var covered = 0
        for at in 0 ..< size * size where r.face[at] >= 0 {
            let t = mesh.faces[Int(r.face[at])]
            let (b1, b2) = (Double(r.w1[at]), Double(r.w2[at]))
            var n = mesh.normals[Int(t.x)] * (1 - b1 - b2) + mesh.normals[Int(t.y)] * b1 + mesh.normals[Int(t.z)] * b2
            n /= max((n * n).sum().squareRoot(), 1e-9)
            for k in 0 ..< 3 {
                rgba[at * 4 + k] = UInt8(max(0, min(255, ((0.5 + 0.5 * n[k]) * 255).rounded(.toNearestOrEven))))
            }
            rgba[at * 4 + 3] = 255
            covered += 1
        }
        return View(size: size, rgba: rgba, coverage: Double(covered) / Double(size * size))
    }

    /// Rasterize `mesh` from `camera`: one sample at each pixel centre, nearest triangle wins.
    public static func rasterize(_ mesh: UltraTexMesh, camera: Camera, size: Int) -> Raster {
        let f = focalAt4096 * Double(size) / 4096, c = Double(size) / 2
        // Project every vertex once: pixel coordinates and inverse depth.
        let count = mesh.positions.count
        var px = [Double](repeating: 0, count: count), py = px, inv = px
        for i in 0 ..< count {
            let d = mesh.positions[i] - camera.position
            let depth = -(d * camera.z).sum()
            if depth > 1e-6 {
                inv[i] = 1 / depth
                px[i] = c + f * (d * camera.x).sum() / depth
                py[i] = c - f * (d * camera.y).sum() / depth
            } else {
                inv[i] = -1                                   // behind the camera: its triangles are skipped
            }
        }
        var best = [Double](repeating: 0, count: size * size)             // inverse depth of the nearest hit; 0 = none
        var face = [Int32](repeating: -1, count: size * size)
        var w1 = [Float](repeating: 0, count: size * size), w2 = w1         // perspective-correct weights of b and c
        for (index, t) in mesh.faces.enumerated() {
            let (a, b, d) = (Int(t.x), Int(t.y), Int(t.z))
            if inv[a] < 0 || inv[b] < 0 || inv[d] < 0 { continue }
            let (x0, y0, x1, y1, x2, y2) = (px[a], py[a], px[b], py[b], px[d], py[d])
            let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if area == 0 { continue }
            let minX = max(0, Int((min(x0, x1, x2) - 0.5).rounded(.up))), maxX = min(size - 1, Int((max(x0, x1, x2) - 0.5).rounded(.down)))
            let minY = max(0, Int((min(y0, y1, y2) - 0.5).rounded(.up))), maxY = min(size - 1, Int((max(y0, y1, y2) - 0.5).rounded(.down)))
            if minX > maxX || minY > maxY { continue }
            let (i0, i1, i2) = (inv[a], inv[b], inv[d])
            for row in minY ... maxY {
                let y = Double(row) + 0.5
                for col in minX ... maxX {
                    let x = Double(col) + 0.5
                    let l1 = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / area
                    let l2 = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / area
                    let l0 = 1 - l1 - l2
                    if l0 < 0 || l1 < 0 || l2 < 0 { continue }
                    let depthInv = l0 * i0 + l1 * i1 + l2 * i2
                    let at = row * size + col
                    if depthInv > best[at] {
                        best[at] = depthInv
                        face[at] = Int32(index)
                        w1[at] = Float(l1 * i1 / depthInv)
                        w2[at] = Float(l2 * i2 / depthInv)
                    }
                }
            }
        }
        return Raster(size: size, face: face, w1: w1, w2: w2, inverseDepth: best)
    }

    /// The mesh with a texture, unlit, seen from `camera`: RGBA8 on transparent black. `texture` is sampled at
    /// the nearest texel; UVs follow glTF (v down).
    public static func textured(_ mesh: UltraTexMesh, texture: UltraTexImage, camera: Camera, size: Int) -> UltraTexImage {
        let r = rasterize(mesh, camera: camera, size: size)
        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        for at in 0 ..< size * size where r.face[at] >= 0 {
            let t = mesh.faces[Int(r.face[at])]
            let (b1, b2) = (Double(r.w1[at]), Double(r.w2[at]))
            let uv = mesh.uvs[Int(t.x)] * (1 - b1 - b2) + mesh.uvs[Int(t.y)] * b1 + mesh.uvs[Int(t.z)] * b2
            let x = max(0, min(texture.width - 1, Int(uv.x * Double(texture.width))))
            let y = max(0, min(texture.height - 1, Int(uv.y * Double(texture.height))))
            let from = (y * texture.width + x) * 4
            rgba[at * 4] = texture.rgba[from]; rgba[at * 4 + 1] = texture.rgba[from + 1]; rgba[at * 4 + 2] = texture.rgba[from + 2]
            rgba[at * 4 + 3] = 255
        }
        return UltraTexImage(width: size, height: size, rgba: rgba)
    }

    /// A camera on the -Y side, turned by `azimuth` about Z and raised by `elevation` (degrees), looking at the
    /// origin. For turnaround checks; azimuth 0, elevation 0 is canonical view 0.
    public static func orbitCamera(azimuth: Double, elevation: Double, distance: Double = defaultDistance) -> Camera {
        let (az, el) = (azimuth * .pi / 180, elevation * .pi / 180)
        let z = SIMD3(sin(az) * cos(el), -cos(az) * cos(el), sin(el))
        var x = SIMD3(-z.y, z.x, 0)                              // (0, 0, 1) cross z
        x /= (x * x).sum().squareRoot()
        let y = SIMD3(z.y * x.z - z.z * x.y, z.z * x.x - z.x * x.z, z.x * x.y - z.y * x.x)
        return Camera(x: x, y: y, z: z, position: z * distance)
    }

    /// All six normal maps, rendered in parallel.
    public static func normalMaps(_ mesh: UltraTexMesh, size: Int, distance: Double = defaultDistance) -> [View] {
        let cams = cameras(distance: distance)
        let lock = NSLock()
        var out = [View?](repeating: nil, count: cams.count)
        DispatchQueue.concurrentPerform(iterations: cams.count) { i in
            let view = normalMap(mesh, camera: cams[i], size: size)
            lock.lock(); out[i] = view; lock.unlock()
        }
        return out.map { $0! }
    }
}
