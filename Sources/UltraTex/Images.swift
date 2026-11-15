// Reading and writing 8-bit RGBA pictures through ImageIO (PNG, JPEG, WebP in; PNG out).

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct UltraTexImage: Sendable {
    public var width: Int
    public var height: Int
    /// Row-major RGBA8. Colour is PREMULTIPLIED by alpha (what CoreGraphics decodes to): for a picture whose
    /// alpha is only 0 or 255 that is the picture itself, and for a soft-edged one `rgb + 255 * (1 - a / 255)`
    /// is the composite on white.
    public var rgba: [UInt8]

    public init(width: Int, height: Int, rgba: [UInt8]) {
        (self.width, self.height, self.rgba) = (width, height, rgba)
    }

    public enum ImageError: Error { case unreadable(String), unwritable(String) }

    public static func decode(_ data: Data) throws -> UltraTexImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw ImageError.unreadable("\(data.count) bytes") }
        let (w, h) = (image.width, image.height)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ok = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.setBlendMode(.copy)
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw ImageError.unreadable("no bitmap context") }
        return UltraTexImage(width: w, height: h, rgba: rgba)
    }

    /// The picture with STRAIGHT (not premultiplied) alpha, as Pillow reads it. Uses the file's own bytes when it is
    /// plain 8-bit RGBA with straight alpha (a PNG); otherwise un-premultiplies the decoded picture, which can be
    /// off by one level where alpha is partial.
    public static func decodeStraight(_ data: Data) throws -> UltraTexImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw ImageError.unreadable("\(data.count) bytes") }
        let (w, h) = (image.width, image.height)
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        if image.bitsPerComponent == 8, image.bitsPerPixel == 32, image.alphaInfo == .last,
           order == [] || order == .byteOrder32Big, let raw = image.dataProvider?.data as Data? {
            var rgba = [UInt8](repeating: 0, count: w * h * 4)
            raw.withUnsafeBytes { bytes in
                for y in 0 ..< h {
                    for i in 0 ..< w * 4 { rgba[y * w * 4 + i] = bytes[y * image.bytesPerRow + i] }
                }
            }
            return UltraTexImage(width: w, height: h, rgba: rgba)
        }
        var out = try decode(data)
        for p in 0 ..< w * h {
            let a = Int(out.rgba[p * 4 + 3])
            if a == 0 || a == 255 { continue }
            for c in 0 ..< 3 { out.rgba[p * 4 + c] = UInt8(min(255, (Int(out.rgba[p * 4 + c]) * 255 + a / 2) / a)) }
        }
        return out
    }

    public static func loadStraight(_ path: String) throws -> UltraTexImage {
        try decodeStraight(Data(contentsOf: URL(fileURLWithPath: path)))
    }

    public static func load(_ path: String) throws -> UltraTexImage {
        try decode(Data(contentsOf: URL(fileURLWithPath: path)))
    }

    /// The picture as PNG bytes.
    public func pngData() throws -> Data {
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let out = NSMutableData()
        guard let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
        else { throw ImageError.unwritable("PNG encoder") }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw ImageError.unwritable("PNG encoder") }
        return out as Data
    }

    public func writePNG(_ path: String) throws {
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        guard let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw ImageError.unwritable(path) }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw ImageError.unwritable(path) }
    }
}
