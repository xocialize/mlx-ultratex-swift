// Writing the baked texture back: a copy of a GLB with its base-colour image replaced.
//
// Everything else is kept byte for byte: geometry, UVs, the other textures, materials, nodes. The new image is a
// PNG (ImageIO does not encode WebP), so if the base-colour texture pointed at its image through
// EXT_texture_webp, that texture is repointed with the core `source` field. The binary chunk is rebuilt with
// every buffer view re-laid at a 4-byte boundary.

import Foundation

public enum UltraTexGLB {
    public enum GLBError: Error, CustomStringConvertible {
        case malformed(String)
        case noBaseColour
        public var description: String {
            switch self {
            case .malformed(let m): "GLB malformed: \(m)"
            case .noBaseColour: "the GLB's first material has no base-colour texture stored in the file"
            }
        }
    }

    /// Copies `source` to `destination` with the base-colour texture of the first material that has one replaced by
    /// `png`.
    public static func replaceBaseColour(in source: URL, with png: Data, to destination: URL) throws {
        let data = try Data(contentsOf: source, options: .mappedIfSafe)
        func u32(_ o: Int) -> Int { Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) }) }
        guard data.count > 20, u32(0) == 0x4654_6C67 else { throw GLBError.malformed("not a GLB") }
        var offset = 12
        var jsonRange: Range<Int>?, binRange: Range<Int>?
        while offset + 8 <= data.count {
            let (length, type) = (u32(offset), u32(offset + 4))
            if type == 0x4E4F_534A { jsonRange = (offset + 8) ..< (offset + 8 + length) }
            if type == 0x004E_4942 { binRange = (offset + 8) ..< (offset + 8 + length) }
            offset += 8 + length
        }
        guard let jsonRange, let binRange,
              var doc = try JSONSerialization.jsonObject(with: data.subdata(in: jsonRange)) as? [String: Any],
              var views = doc["bufferViews"] as? [[String: Any]]
        else { throw GLBError.malformed("missing JSON or BIN chunk") }

        // Which image is the base colour?
        guard let materials = doc["materials"] as? [[String: Any]], var textures = doc["textures"] as? [[String: Any]],
              var images = doc["images"] as? [[String: Any]] else { throw GLBError.noBaseColour }
        var textureIndex: Int?
        for m in materials {
            if let i = ((m["pbrMetallicRoughness"] as? [String: Any])?["baseColorTexture"] as? [String: Any])?["index"] as? Int {
                textureIndex = i
                break
            }
        }
        guard let textureIndex else { throw GLBError.noBaseColour }
        var texture = textures[textureIndex]
        var extensions = texture["extensions"] as? [String: Any] ?? [:]
        let webp = (extensions["EXT_texture_webp"] as? [String: Any])?["source"] as? Int
        guard let imageIndex = webp ?? (texture["source"] as? Int), let imageView = images[imageIndex]["bufferView"] as? Int
        else { throw GLBError.noBaseColour }
        extensions["EXT_texture_webp"] = nil
        texture["extensions"] = extensions.isEmpty ? nil : extensions
        texture["source"] = imageIndex
        textures[textureIndex] = texture
        images[imageIndex]["mimeType"] = "image/png"

        // Re-lay the binary chunk.
        var bin = Data()
        for i in views.indices {
            while bin.count % 4 != 0 { bin.append(0) }
            let start = binRange.lowerBound + (views[i]["byteOffset"] as? Int ?? 0)
            let bytes = i == imageView ? png : data.subdata(in: start ..< (start + (views[i]["byteLength"] as? Int ?? 0)))
            views[i]["byteOffset"] = bin.count
            views[i]["byteLength"] = bytes.count
            bin.append(bytes)
        }
        while bin.count % 4 != 0 { bin.append(0) }
        doc["bufferViews"] = views
        doc["textures"] = textures
        doc["images"] = images
        if var buffers = doc["buffers"] as? [[String: Any]], !buffers.isEmpty {
            buffers[0]["byteLength"] = bin.count
            doc["buffers"] = buffers
        }
        var json = try JSONSerialization.data(withJSONObject: doc, options: [.sortedKeys, .withoutEscapingSlashes])
        while json.count % 4 != 0 { json.append(0x20) }

        var out = Data()
        func put(_ v: Int) { var x = UInt32(v).littleEndian; withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
        put(0x4654_6C67); put(2); put(12 + 8 + json.count + 8 + bin.count)
        put(json.count); put(0x4E4F_534A); out.append(json)
        put(bin.count); put(0x004E_4942); out.append(bin)
        try out.write(to: destination)
    }
}
