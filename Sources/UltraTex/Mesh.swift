// The mesh UltraTex textures: read from a GLB, put in the frame and scale the model was trained on, with smooth
// normals. Mirrors DEV/UltraTexOracle/tools/render_gbuffer.py, which made the oracle's inputs (AB-R-0874 for the
// camera rule it follows; upstream's render script is not published).
//
//   - frame: Z up, the mesh's front toward -Y. A GLB is Y up; TRELLIS.2 / Pixal3D exports map to it by
//     (x, y, z) -> (-x, z, y).
//   - scale: centred on its bounding box and scaled into a sphere of radius 0.5.
//   - normals: area-weighted, accumulated on vertices welded by position, so UV seams (duplicated vertices) do not
//     show as shading seams. The GLB's own NORMAL attribute is not used: it need not exist, and where it does it
//     agrees with these to a mean cosine of 0.9987 on the corpus dress.

import Foundation

public struct UltraTexMesh: Sendable {
    /// Per vertex, in the normalized canonical frame.
    public var positions: [SIMD3<Double>]
    public var normals: [SIMD3<Double>]
    /// Per vertex, glTF convention (v down), or empty when the mesh has no UVs.
    public var uvs: [SIMD2<Double>]
    public var faces: [SIMD3<Int32>]
    /// What normalization did: `canonical = (raw - centre) * scale`.
    public var centre: SIMD3<Double>
    public var scale: Double
    /// The encoded base-colour image (PNG, JPEG or WebP bytes), when the GLB carries one.
    public var baseColourImage: Data?

    public enum MeshError: Error, CustomStringConvertible {
        case malformed(String)
        case unsupported(String)
        public var description: String {
            switch self {
            case .malformed(let m): "GLB malformed: \(m)"
            case .unsupported(let m): "GLB unsupported: \(m)"
            }
        }
    }

    // MARK: GLB

    /// Reads every triangle primitive of the default scene, with node transforms applied.
    public static func load(glb url: URL) throws -> UltraTexMesh {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        func u32(_ o: Int) -> Int { Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) }) }
        guard data.count > 20, u32(0) == 0x4654_6C67 else { throw MeshError.malformed("not a GLB") }
        var offset = 12
        var json: [String: Any]?
        var binStart = 0
        while offset + 8 <= data.count {
            let (length, type) = (u32(offset), u32(offset + 4))
            if type == 0x4E4F_534A {
                json = try JSONSerialization.jsonObject(with: data.subdata(in: (offset + 8) ..< (offset + 8 + length))) as? [String: Any]
            } else if type == 0x004E_4942 {
                binStart = offset + 8
            }
            offset += 8 + length
        }
        guard let doc = json, binStart > 0,
              let accessors = doc["accessors"] as? [[String: Any]],
              let views = doc["bufferViews"] as? [[String: Any]],
              let meshes = doc["meshes"] as? [[String: Any]],
              let nodes = doc["nodes"] as? [[String: Any]]
        else { throw MeshError.malformed("missing JSON or BIN chunk") }

        func viewRange(_ index: Int) -> Range<Int> {
            let v = views[index]
            let start = binStart + (v["byteOffset"] as? Int ?? 0)
            return start ..< (start + (v["byteLength"] as? Int ?? 0))
        }
        func floats(_ index: Int, components: Int) throws -> [Double] {
            let a = accessors[index]
            guard let view = a["bufferView"] as? Int, (a["componentType"] as? Int) == 5126 else {
                throw MeshError.unsupported("accessor \(index) is not a float buffer")
            }
            let count = a["count"] as? Int ?? 0
            let stride = (views[view]["byteStride"] as? Int) ?? components * 4
            let start = viewRange(view).lowerBound + (a["byteOffset"] as? Int ?? 0)
            var out = [Double](repeating: 0, count: count * components)
            data.withUnsafeBytes { raw in
                for i in 0 ..< count {
                    for c in 0 ..< components {
                        out[i * components + c] = Double(raw.loadUnaligned(fromByteOffset: start + i * stride + c * 4, as: Float.self))
                    }
                }
            }
            return out
        }
        func indices(_ index: Int) throws -> [Int32] {
            let a = accessors[index]
            guard let view = a["bufferView"] as? Int else { throw MeshError.unsupported("sparse index accessor") }
            let count = a["count"] as? Int ?? 0
            let start = viewRange(view).lowerBound + (a["byteOffset"] as? Int ?? 0)
            let type = a["componentType"] as? Int ?? 0
            return try data.withUnsafeBytes { raw in
                try (0 ..< count).map { i in
                    switch type {
                    case 5121: Int32(raw.loadUnaligned(fromByteOffset: start + i, as: UInt8.self))
                    case 5123: Int32(raw.loadUnaligned(fromByteOffset: start + i * 2, as: UInt16.self))
                    case 5125: Int32(bitPattern: raw.loadUnaligned(fromByteOffset: start + i * 4, as: UInt32.self))
                    default: throw MeshError.unsupported("index component type \(type)")
                    }
                }
            }
        }
        /// Column-major 4x4 of a node: `matrix`, or translation * rotation * scale.
        func matrix(_ node: [String: Any]) -> [Double] {
            if let m = node["matrix"] as? [NSNumber], m.count == 16 { return m.map(\.doubleValue) }
            let t = (node["translation"] as? [NSNumber])?.map(\.doubleValue) ?? [0, 0, 0]
            let q = (node["rotation"] as? [NSNumber])?.map(\.doubleValue) ?? [0, 0, 0, 1]
            let s = (node["scale"] as? [NSNumber])?.map(\.doubleValue) ?? [1, 1, 1]
            let (x, y, z, w) = (q[0], q[1], q[2], q[3])
            let r: [Double] = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
                               2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
                               2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]
            return [r[0] * s[0], r[1] * s[0], r[2] * s[0], 0, r[3] * s[1], r[4] * s[1], r[5] * s[1], 0,
                    r[6] * s[2], r[7] * s[2], r[8] * s[2], 0, t[0], t[1], t[2], 1]
        }
        func multiply(_ a: [Double], _ b: [Double]) -> [Double] {
            var out = [Double](repeating: 0, count: 16)
            for c in 0 ..< 4 { for r in 0 ..< 4 { for k in 0 ..< 4 { out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k] } } }
            return out
        }

        var positions: [SIMD3<Double>] = [], uvs: [SIMD2<Double>] = [], faces: [SIMD3<Int32>] = []
        var material: Int?
        var allHaveUVs = true
        func walk(_ index: Int, _ parent: [Double]) throws {
            let node = nodes[index]
            let m = multiply(parent, matrix(node))
            if let meshIndex = node["mesh"] as? Int {
                for prim in meshes[meshIndex]["primitives"] as? [[String: Any]] ?? [] {
                    if let mode = prim["mode"] as? Int, mode != 4 { continue }
                    guard let attributes = prim["attributes"] as? [String: Any], let pos = attributes["POSITION"] as? Int,
                          let idx = prim["indices"] as? Int else { continue }
                    let base = Int32(positions.count)
                    let p = try floats(pos, components: 3)
                    for i in 0 ..< p.count / 3 {
                        let (x, y, z) = (p[i * 3], p[i * 3 + 1], p[i * 3 + 2])
                        positions.append(SIMD3(m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13],
                                               m[2] * x + m[6] * y + m[10] * z + m[14]))
                    }
                    if let uv = attributes["TEXCOORD_0"] as? Int {
                        let t = try floats(uv, components: 2)
                        for i in 0 ..< t.count / 2 { uvs.append(SIMD2(t[i * 2], t[i * 2 + 1])) }
                    } else {
                        allHaveUVs = false
                    }
                    let ix = try indices(idx)
                    for i in 0 ..< ix.count / 3 { faces.append(SIMD3(base + ix[i * 3], base + ix[i * 3 + 1], base + ix[i * 3 + 2])) }
                    if material == nil { material = prim["material"] as? Int }
                }
            }
            for child in node["children"] as? [Int] ?? [] { try walk(child, m) }
        }
        let scenes = doc["scenes"] as? [[String: Any]] ?? []
        let roots = scenes.isEmpty ? Array(nodes.indices) : (scenes[doc["scene"] as? Int ?? 0]["nodes"] as? [Int] ?? [])
        let identity: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        for root in roots { try walk(root, identity) }
        guard !positions.isEmpty, !faces.isEmpty else { throw MeshError.malformed("no triangles") }

        var image: Data?
        if let material, let materials = doc["materials"] as? [[String: Any]],
           let pbr = materials[material]["pbrMetallicRoughness"] as? [String: Any],
           let texture = (pbr["baseColorTexture"] as? [String: Any])?["index"] as? Int,
           let textures = doc["textures"] as? [[String: Any]], let images = doc["images"] as? [[String: Any]] {
            let t = textures[texture]
            let webp = ((t["extensions"] as? [String: Any])?["EXT_texture_webp"] as? [String: Any])?["source"] as? Int
            if let source = webp ?? (t["source"] as? Int), let view = images[source]["bufferView"] as? Int {
                image = data.subdata(in: viewRange(view))
            }
        }
        return normalized(rawPositions: positions, uvs: allHaveUVs ? uvs : [], faces: faces, baseColourImage: image)
    }

    // MARK: Frame, scale, normals

    /// glTF (Y up) -> canonical (Z up, front toward -Y), then smooth normals, then centre and scale.
    static func normalized(rawPositions: [SIMD3<Double>], uvs: [SIMD2<Double>], faces: [SIMD3<Int32>],
                           baseColourImage: Data?) -> UltraTexMesh {
        var v = rawPositions.map { SIMD3(-$0.x, $0.z, $0.y) }
        let normals = smoothNormals(positions: v, faces: faces)
        var lo = v[0], hi = v[0]
        for p in v { lo = pointwiseMin(lo, p); hi = pointwiseMax(hi, p) }
        let centre = (lo + hi) / 2
        var radius = 0.0
        for p in v { radius = max(radius, ((p - centre) * (p - centre)).sum().squareRoot()) }
        let scale = 0.5 / radius
        for i in v.indices { v[i] = (v[i] - centre) * scale }
        return UltraTexMesh(positions: v, normals: normals, uvs: uvs, faces: faces, centre: centre, scale: scale,
                            baseColourImage: baseColourImage)
    }

    /// Area-weighted vertex normals over vertices welded by position (rounded to 1e-7).
    static func smoothNormals(positions v: [SIMD3<Double>], faces: [SIMD3<Int32>]) -> [SIMD3<Double>] {
        var weld = [Int32](repeating: 0, count: v.count)
        var seen: [SIMD3<Int64>: Int32] = [:]
        seen.reserveCapacity(v.count)
        for (i, p) in v.enumerated() {
            let key = SIMD3(Int64((p.x * 1e7).rounded(.toNearestOrEven)), Int64((p.y * 1e7).rounded(.toNearestOrEven)),
                            Int64((p.z * 1e7).rounded(.toNearestOrEven)))
            if let id = seen[key] { weld[i] = id } else { let id = Int32(seen.count); seen[key] = id; weld[i] = id }
        }
        var sums = [SIMD3<Double>](repeating: .zero, count: seen.count)
        for f in faces {
            let (a, b, c) = (v[Int(f.x)], v[Int(f.y)], v[Int(f.z)])
            let e1 = b - a, e2 = c - a
            let n = SIMD3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x)
            sums[Int(weld[Int(f.x)])] += n
            sums[Int(weld[Int(f.y)])] += n
            sums[Int(weld[Int(f.z)])] += n
        }
        return weld.map { id in
            let n = sums[Int(id)]
            return n / max((n * n).sum().squareRoot(), 1e-12)
        }
    }
}
