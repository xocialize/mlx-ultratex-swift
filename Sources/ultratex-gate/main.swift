// ultratex-gate — the Swift port against the CUDA goldens (DEV/UltraTexOracle, --plan golden).
//
//   swift run -c release ultratex-gate --mode step|select|sample --golden <golden run dir>/st
//        [--transformer <diffusers transformer dir>] [--lora <dit_lora.safetensors>]
//        [--step 0] [--stream bf16|fp32] [--attn chunked|dense] [--resolution 1024] [--max-gb N]
//
// Every mode runs under a hard MLX memory cap (--max-gb; 16 GB for select and vae, 32 GB otherwise) and refuses
// to start unless the machine has that much plus 16 GB available.
//
//   step    one DiT pass on a golden step's inputs: velocity, and at step 0 the dumped block outputs.
//   select  no model: token selection from the masks, position ids, kept condition tokens and the schedule,
//           each of which must equal the golden exactly.
//   sample  `select`, then 25 steps from the golden's noise: the target tokens entering steps 12 and 24 and the
//           six composite latent tiles upstream decodes. `--save-latents <file>` keeps the Swift canvas.
//   render  --glb <mesh.glb> --item <DEV/UltraTexOracle/stage/items/NAME> [--size 2048] [--out <dir>]: no model
//           and no golden. The six normal maps from the Swift renderer against the ones
//           DEV/UltraTexOracle/tools/render_gbuffer.py made for the oracle run.
//   inputs  --item <stage item dir> [--glb <mesh.glb>] --golden <golden run dir>/st --resolution R: no model. The
//           normals canvas, the reference tensor and the three token masks built in Swift against the golden's.
//           With --glb the views come from the Swift renderer; without it, from the item's Python renders, which
//           isolates the resize / paste / mask arithmetic.
//   pipeline  --glb <mesh.glb> --reference <picture.png> --transformer <dir> --lora <file> --vae <dir> --text
//           <empty_prompt_qwen3_8b.safetensors> --resolution R --png <out.png> [--seed N] [--golden <dir>/st
//           --golden-png <result.png>]: a whole run from a mesh and a picture. With --golden the run starts from
//           the golden's noise and the picture is compared with the CUDA run's; without it the noise comes
//           from --seed and there is nothing to compare with.
//   bake    --glb <mesh.glb> --views <2x3 canvas png> [--distance 1.6] [--atlas 2048] [--out-glb <file>]
//           [--out-atlas <png>] [--turn <png>] [--python-atlas <dir with atlas.png and coverage.png>]: no model.
//           The six views onto the mesh's texture with a fill, a GLB out, a turnaround sheet (the old texture
//           over the new), and a comparison with DEV/UltraTexOracle/tools/bake.py on the same canvas.
//   vae     --vae <converted vae dir> [--exact] [--latents <file from sample> --png <out.png>]: the encoder on
//           the golden's normals tile and reference, the fine-tuned decoder on the golden's six latent tiles,
//           and, with --latents, the decoder on the Swift run's own latents against the golden's pictures.
//
// Cosine and relative L2 are over all elements. The goldens' block tensors are fp16 on disk, so the few elements
// that overflowed to infinity there are left out of both sides. Use the DENSE golden for step and sample: the
// sparse golden differs by design. Leave out --lora for a control run on the bare base model.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import UltraTex
import UniformTypeIdentifiers

func arg(_ name: String, _ fallback: String) -> String {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: "--" + name), i + 1 < a.count else { return fallback }
    return a[i + 1]
}

let mode = arg("mode", "step")
// A hard cap on this process's MLX memory (see UltraTexMemory). The defaults cover the measured peaks with room:
// 22 GB for a 1024² sampling run, under 10 GB for the autoencoder.
let capGB = Double(arg("max-gb", ["vae", "select", "render", "inputs", "bake"].contains(mode) ? "16" : "32"))!
let availableGB = UltraTexMemory.guardProcess(capGB: capGB, label: "ultratex-gate")
FileHandle.standardError.write("memory: \(String(format: "%.0f", availableGB)) GB available, cap \(Int(capGB)) GB\n".data(using: .utf8)!)
let golden = URL(fileURLWithPath: arg("golden", ""))
let stream = arg("stream", "bf16")
AttentionUtils.backend = arg("attn", "chunked") == "dense" ? .dense : .denseChunked(maxScoreBytes: 4 << 30)

func load(_ name: String) throws -> [String: MLXArray] {
    try Device.withDefaultDevice(.cpu) {
        try MLX.loadArrays(url: golden.appendingPathComponent(name + ".safetensors"))
    }
}

/// (cosine, relative L2 error, elements left out because the reference is not finite).
func compare(_ ours: MLXArray, _ reference: MLXArray) -> (Float, Float, Int) {
    let r = reference.asType(.float32)
    let o = ours.asType(.float32)
    let bad = MLX.logicalNot(MLX.isFinite(r))
    let zero = MLXArray(Float(0))
    let rr = which(bad, zero, r)
    let oo = which(bad, zero, o)
    let dot = (rr * oo).sum().item(Float.self)
    let nr = (rr * rr).sum().sqrt().item(Float.self)
    let no = (oo * oo).sum().sqrt().item(Float.self)
    let diff = ((oo - rr) * (oo - rr)).sum().sqrt().item(Float.self)
    return (dot / (nr * no), diff / nr, bad.sum().item(Int.self))
}

nonisolated(unsafe) var rows: [(String, String)] = []
func record(_ name: String, _ ours: MLXArray, _ reference: MLXArray) {
    precondition(ours.shape == reference.shape, "\(name): ours \(ours.shape), golden \(reference.shape)")
    let (c, e, n) = compare(ours, reference)
    rows.append((name, String(format: "%.6f  %.5f  %d", c, e, n)))
    FileHandle.standardError.write("\(name): cos \(c) rel \(e)\n".data(using: .utf8)!)
}
/// For things that must be identical: the largest absolute difference.
func exact(_ name: String, _ ours: MLXArray, _ reference: MLXArray) {
    precondition(ours.shape == reference.shape, "\(name): ours \(ours.shape), golden \(reference.shape)")
    let d = MLX.abs(ours.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
    rows.append((name, d == 0 ? "identical" : "MAX DIFFERENCE \(d)"))
}
func report(_ title: String) {
    print(title)
    for (name, value) in rows {
        print(name.padding(toLength: 40, withPad: " ", startingAt: 0) + "  " + value)
    }
}

func loadModel() throws -> (Flux2Transformer, Double) {
    let t0 = Date()
    let loraPath = arg("lora", "")
    let loaded = try UltraTexWeights.loadTransformer(
        directory: URL(fileURLWithPath: arg("transformer", "")),
        lora: loraPath.isEmpty ? nil : URL(fileURLWithPath: loraPath))
    if stream == "bf16" { loaded.model.streamDType = .bfloat16 }
    return (loaded.model, Date().timeIntervalSince(t0))
}

let name = golden.deletingLastPathComponent().lastPathComponent

/// PSNR in 8-bit terms between two pictures in 0...255.
func psnr(_ a: MLXArray, _ b: MLXArray) -> Float {
    let d = a.asType(.float32) - b.asType(.float32)
    let mse = (d * d).mean().item(Float.self)
    return 10 * log10(255 * 255 / max(mse, 1e-9))
}

/// Upstream's conversion of a decoded tile [1, 3, H, W] to 0...255: clamp, 127.5 * (x + 1), truncate.
func toBytes(_ nchw: MLXArray) -> MLXArray {
    floor((clip(nchw[0].transposed(1, 2, 0), min: -1, max: 1) + 1) * 127.5)
}

func writePNG(_ rgb: MLXArray, to path: String) {
    let (h, w) = (rgb.dim(0), rgb.dim(1))
    let bytes = rgb.asType(.uint8).asArray(UInt8.self)
    let provider = CGDataProvider(data: Data(bytes) as CFData)!
    let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: w * 3,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: 0),
                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

if mode == "render" {
    let item = arg("item", "")
    let size = Int(arg("size", "2048"))!
    let t0 = Date()
    let mesh = try UltraTexMesh.load(glb: URL(fileURLWithPath: arg("glb", "")))
    let loadSeconds = Date().timeIntervalSince(t0)
    let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: item + "/meta.json"))) as! [String: Any]
    let distance = meta["dist"] as! Double
    let centre = (meta["centre"] as! [NSNumber]).map(\.doubleValue)
    rows.append(("vertices / faces", "\(mesh.positions.count) / \(mesh.faces.count) (python \(meta["vertices"]!) / \(meta["faces"]!))"))
    rows.append(("scale", String(format: "%.12f (python %.12f)", mesh.scale, meta["scale"] as! Double)))
    rows.append(("centre, largest difference", String(format: "%.3g", (0 ..< 3).map { abs(mesh.centre[$0] - centre[$0]) }.max()!)))
    let t1 = Date()
    let views = UltraTexRenderer.normalMaps(mesh, size: size, distance: distance)
    let renderSeconds = Date().timeIntervalSince(t1)
    let outDir = arg("out", "")
    var totalUnion = 0, totalBoth = 0, totalExact = 0, totalWithin1 = 0
    for (i, view) in views.enumerated() {
        if !outDir.isEmpty {
            try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
            try UltraTexImage(width: size, height: size, rgba: view.rgba).writePNG(String(format: "%@/%03d.png", outDir, i))
        }
        let ref = try UltraTexImage.load(String(format: "%@/normal_cam/%03d.png", item, i))
        precondition(ref.width == size && ref.height == size, "reference is \(ref.width)x\(ref.height)")
        var both = 0, union = 0, exact = 0, within1 = 0, sumAbs = 0, worst = 0
        for p in 0 ..< size * size {
            let (a, b) = (view.rgba[p * 4 + 3] != 0, ref.rgba[p * 4 + 3] != 0)
            if a || b { union += 1 }
            guard a && b else { continue }
            both += 1
            var m = 0
            for k in 0 ..< 3 {
                let d = abs(Int(view.rgba[p * 4 + k]) - Int(ref.rgba[p * 4 + k]))
                m = max(m, d); sumAbs += d
            }
            if m == 0 { exact += 1 }
            if m <= 1 { within1 += 1 }
            worst = max(worst, m)
        }
        totalUnion += union; totalBoth += both; totalExact += exact; totalWithin1 += within1
        rows.append(("view \(i)", String(format: "coverage IoU %.6f (%d px differ), normals identical %.4f, within 1 level %.4f, mean |d| %.3f, max %d",
                                         Double(both) / Double(union), union - both, Double(exact) / Double(both),
                                         Double(within1) / Double(both), Double(sumAbs) / Double(both * 3), worst)))
    }
    rows.append(("all six views", String(format: "coverage IoU %.6f, normals identical %.4f, within 1 level %.4f",
                                        Double(totalBoth) / Double(totalUnion), Double(totalExact) / Double(totalBoth),
                                        Double(totalWithin1) / Double(totalBoth))))
    report("renderer against the Python renders of \(URL(fileURLWithPath: item).lastPathComponent) at \(size)²: "
           + "load \(String(format: "%.1f", loadSeconds)) s, six views in \(String(format: "%.1f", renderSeconds)) s")
    exit(0)
}

if mode == "pipeline" {
    let resolution = Int(arg("resolution", "1024"))!
    let (model, loadSeconds) = try loadModel()
    let vae = try UltraTexVAE.load(directory: URL(fileURLWithPath: arg("vae", "")))
    let text = try MLX.loadArrays(url: URL(fileURLWithPath: arg("text", "")))["ctx"]!
    let pipeline = UltraTexPipeline(model: model, vae: vae, text: text)
    let mesh = try UltraTexMesh.load(glb: URL(fileURLWithPath: arg("glb", "")))
    let reference = try UltraTexImage.loadStraight(arg("reference", ""))
    var noise: MLXArray? = nil
    let hasGolden = !arg("golden", "").isEmpty
    if hasGolden {
        let s0 = try load("step_00")
        let n = try load("inputs")["mask_target"]!.asType(.int32).sum().item(Int.self)
        noise = s0["x"]![0 ..< n, 0...]
    }
    MLX.Memory.peakMemory = 0
    let t0 = Date()
    let out = pipeline.run(mesh: mesh, reference: reference, resolution: resolution, seed: UInt64(arg("seed", "42"))!,
                           noise: noise) { i in
        FileHandle.standardError.write("step \(i) at \(String(format: "%.0f", Date().timeIntervalSince(t0))) s\n".data(using: .utf8)!)
    }
    let total = Date().timeIntervalSince(t0)
    try out.canvas.writePNG(arg("png", ""))
    for (k, v) in out.seconds.sorted(by: { $0.key < $1.key }) { rows.append((k, String(format: "%.1f s", v))) }
    rows.append(("kept tokens", "\(out.keptTokens)"))
    rows.append(("peak MLX memory", "\(MLX.Memory.peakMemory >> 20) MB"))
    let goldenPNG = arg("golden-png", "")
    if !goldenPNG.isEmpty {
        let g = try UltraTexImage.load(goldenPNG)
        precondition(g.width == out.canvas.width && g.height == out.canvas.height, "golden picture is \(g.width)x\(g.height)")
        var sum = 0.0, count = 0, bgDiffer = 0, worst = 0, within8 = 0
        for p in 0 ..< g.width * g.height {
            let white = g.rgba[p * 4] >= 250 && g.rgba[p * 4 + 1] >= 250 && g.rgba[p * 4 + 2] >= 250
            var m = 0
            for c in 0 ..< 3 {
                let d = abs(Int(g.rgba[p * 4 + c]) - Int(out.canvas.rgba[p * 4 + c]))
                m = max(m, d)
                if !white { sum += Double(d * d) }
            }
            if white { if m > 5 { bgDiffer += 1 } } else { count += 1; worst = max(worst, m); if m <= 8 { within8 += 1 } }
        }
        let mse = sum / Double(count * 3)
        rows.append(("picture vs the CUDA run, foreground", String(format: "%.1f dB, %.2f %% of pixels within 8 levels, max %d",
                                                                 10 * log10(255 * 255 / max(mse, 1e-9)), 100 * Double(within8) / Double(count), worst)))
        rows.append(("background pixels that are not white", "\(bgDiffer)"))
    }
    report("pipeline: \(URL(fileURLWithPath: arg("glb", "")).deletingLastPathComponent().lastPathComponent) at \(resolution)², "
           + "\(hasGolden ? "the golden's noise" : "seed \(arg("seed", "42"))"), load \(String(format: "%.1f", loadSeconds)) s, "
           + "run \(String(format: "%.0f", total)) s")
    exit(0)
}

if mode == "bake" {
    let glb = URL(fileURLWithPath: arg("glb", ""))
    let distance = Double(arg("distance", "1.6"))!
    let atlasSize = Int(arg("atlas", "2048"))!
    let mesh = try UltraTexMesh.load(glb: glb)
    let canvas = try UltraTexImage.load(arg("views", ""))
    let r = canvas.width / 3
    precondition(canvas.width == 3 * r && canvas.height == 2 * r, "not a 2x3 canvas: \(canvas.width)x\(canvas.height)")
    let slots = [0: (0, 0), 1: (0, 1), 3: (0, 2), 2: (1, 0), 4: (1, 1), 5: (1, 2)]
    let views: [UltraTexImage] = (0 ..< 6).map { v in
        let (row, col) = slots[v]!
        var rgba = [UInt8](repeating: 0, count: r * r * 4)
        for y in 0 ..< r {
            let from = ((row * r + y) * canvas.width + col * r) * 4
            rgba.replaceSubrange((y * r * 4) ..< ((y + 1) * r * 4), with: canvas.rgba[from ..< (from + r * 4)])
        }
        return UltraTexImage(width: r, height: r, rgba: rgba)
    }
    let t0 = Date()
    let baked = UltraTexBake.bake(mesh: mesh, views: views, atlasSize: atlasSize, distance: distance)
    let seconds = Date().timeIntervalSince(t0)
    rows.append(("texels on the mesh", "\(baked.texelsOnMesh) of \(atlasSize * atlasSize)"))
    rows.append(("painted by a view", String(format: "%.1f %%", baked.paintedFraction * 100)))
    rows.append(("filled from 3D neighbours", String(format: "%.1f %%", baked.filledFraction * 100)))
    rows.append(("painted per view", baked.perView.map { String(format: "%.1f", $0 * 100) }.joined(separator: " / ") + " %"))
    if !arg("out-atlas", "").isEmpty { try baked.atlas.writePNG(arg("out-atlas", "")) }
    if !arg("out-glb", "").isEmpty {
        try UltraTexGLB.replaceBaseColour(in: glb, with: try baked.atlas.pngData(), to: URL(fileURLWithPath: arg("out-glb", "")))
        let reread = try UltraTexMesh.load(glb: URL(fileURLWithPath: arg("out-glb", "")))
        let back = try UltraTexImage.decode(reread.baseColourImage!)
        rows.append(("GLB written and read back", "\(reread.faces.count) faces, texture \(back.width)x\(back.height), "
                     + (back.rgba == baked.atlas.rgba ? "identical to the atlas" : "DIFFERS from the atlas")))
    }
    let python = arg("python-atlas", "")
    if !python.isEmpty {
        let a = try UltraTexImage.load(python + "/atlas.png"), c = try UltraTexImage.load(python + "/coverage.png")
        precondition(a.width == atlasSize, "python atlas is \(a.width)")
        var both = 0, onlyOurs = 0, onlyTheirs = 0, sum = 0.0
        for at in 0 ..< atlasSize * atlasSize {
            let (ours, theirs) = (baked.origin[at] == 1, c.rgba[at * 4] > 127)
            if ours && theirs {
                both += 1
                for k in 0 ..< 3 { let d = Double(Int(a.rgba[at * 4 + k]) - Int(baked.atlas.rgba[at * 4 + k])); sum += d * d }
            } else if ours { onlyOurs += 1 } else if theirs { onlyTheirs += 1 }
        }
        rows.append(("vs the Python bake: painted by both", "\(both) texels; only Swift \(onlyOurs), only Python \(onlyTheirs)"))
        rows.append(("vs the Python bake: colour where both painted", String(format: "%.1f dB", 10 * log10(255 * 255 / max(sum / Double(both * 3), 1e-9)))))
    }
    let turn = arg("turn", "")
    if !turn.isEmpty, let old = mesh.baseColourImage {
        let cell = 384
        var sheet = [UInt8](repeating: 255, count: cell * 8 * cell * 2 * 4)
        for (row, texture) in [try UltraTexImage.decode(old), baked.atlas].enumerated() {
            for col in 0 ..< 8 {
                let camera = UltraTexRenderer.orbitCamera(azimuth: Double(col) * 45, elevation: 15, distance: distance)
                let view = UltraTexRenderer.textured(mesh, texture: texture, camera: camera, size: cell)
                for y in 0 ..< cell {
                    for x in 0 ..< cell where view.rgba[(y * cell + x) * 4 + 3] != 0 {
                        let (from, to) = ((y * cell + x) * 4, ((row * cell + y) * cell * 8 + col * cell + x) * 4)
                        sheet[to] = view.rgba[from]; sheet[to + 1] = view.rgba[from + 1]; sheet[to + 2] = view.rgba[from + 2]
                    }
                }
            }
        }
        try UltraTexImage(width: cell * 8, height: cell * 2, rgba: sheet).writePNG(turn)
    }
    report("bake: \(glb.deletingLastPathComponent().lastPathComponent), views \(r)², atlas \(atlasSize)², \(String(format: "%.1f", seconds)) s")
    exit(0)
}

if mode == "inputs" {
    let item = arg("item", "")
    let resolution = Int(arg("resolution", "1024"))!
    let glb = arg("glb", "")
    var views: [UltraTexImage] = []
    if glb.isEmpty {
        views = try (0 ..< 6).map { try UltraTexImage.loadStraight(String(format: "%@/normal_cam/%03d.png", item, $0)) }
    } else {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: item + "/meta.json"))) as! [String: Any]
        let mesh = try UltraTexMesh.load(glb: URL(fileURLWithPath: glb))
        views = UltraTexRenderer.normalMaps(mesh, size: 2048, distance: meta["dist"] as! Double)
            .map { UltraTexImage(width: $0.size, height: $0.size, rgba: $0.rgba) }
    }
    let t0 = Date()
    let prepared = UltraTexInputs.prepare(views: views, reference: try UltraTexImage.loadStraight(item + "/reference.png"),
                                          resolution: resolution)
    let seconds = Date().timeIntervalSince(t0)
    let g = try load("inputs")
    func levels(_ name: String, _ ours: MLXArray, _ reference: MLXArray) {       // tensors in [-1, 1]: differences in 8-bit levels
        precondition(ours.shape == reference.shape, "\(name): ours \(ours.shape), golden \(reference.shape)")
        let d = MLX.abs(ours.asType(.float32) - reference.asType(.float32)) * 127.5
        let worst = d.max().item(Float.self)
        let off = (d .> 0.5).asType(.float32).mean().item(Float.self)
        rows.append((name, worst < 0.5 ? "identical" : String(format: "%.4f %% of values differ, by at most %.0f levels", off * 100, worst)))
    }
    if let canvas = g["normals_canvas"] { levels("normals canvas \(canvas.shape)", UltraTexInputs.canvas(prepared.normalViews), canvas) }
    if let reference = g["reference"] { levels("reference tensor \(reference.shape)", prepared.reference[0], reference) }
    func mask(_ name: String, _ ours: MLXArray, _ reference: MLXArray) {
        let differ = (ours.asType(.float32) .!= reference.asType(.float32)).asType(.int32).sum().item(Int.self)
        rows.append((name + " (\(ours.asType(.int32).sum().item(Int.self)) kept)", differ == 0 ? "identical" : "\(differ) cells differ"))
    }
    mask("target mask", prepared.canvasMask, g["mask_target"]!)
    mask("normals mask", prepared.canvasMask, g["mask_normals"]!)
    mask("reference mask", prepared.referenceMask, g["mask_reference"]!)
    report("inputs at \(resolution)² from \(glb.isEmpty ? "the Python renders" : "the Swift renderer") of "
           + "\(URL(fileURLWithPath: item).lastPathComponent), prepared in \(String(format: "%.1f", seconds)) s, against golden \(name)")
    exit(0)
}

if mode == "vae" {
    let exactRoute = CommandLine.arguments.contains("--exact")
    let vae = try UltraTexVAE.load(directory: URL(fileURLWithPath: arg("vae", "")), exactConvolutions: exactRoute)
    for (i, what) in [(6, "normals tile"), (12, "reference")] {
        let g = try load(String(format: "vae_encode_%02d", i))
        let t = Date()
        let z = vae.encode(g["input"]!)
        eval(z)
        record("encode \(what) (\(String(format: "%.1f", Date().timeIntervalSince(t))) s)", z, g["output"]!)
    }
    var worst: Float = 999
    for i in 0 ..< 6 {
        let g = try load(String(format: "vae_decode_%02d", i))
        let t = Date()
        let out = vae.decode(g["input"]!)
        eval(out)
        let p = psnr(toBytes(out), toBytes(g["output"]!))
        worst = min(worst, p)
        record("decode tile \(i), golden latent (\(String(format: "%.1f", Date().timeIntervalSince(t))) s, \(String(format: "%.1f", p)) dB)", out, g["output"]!)
    }
    rows.append(("decode, golden latents: worst PSNR", String(format: "%.1f dB", worst)))
    let latentsPath = arg("latents", "")
    if !latentsPath.isEmpty {
        let canvas = try MLX.loadArrays(url: URL(fileURLWithPath: latentsPath))["canvas"]!
        var pictures: [MLXArray] = [], worstOurs: Float = 999
        for (i, tile) in UltraTexSampler.tiles(canvas).enumerated() {
            let ours = vae.decodeTile(tile)
            eval(ours)
            let g = toBytes(try load(String(format: "vae_decode_%02d", i))["output"]!)
            let p = psnr(floor(ours), g)
            worstOurs = min(worstOurs, p)
            rows.append(("picture, tile \(i): Swift latents vs golden", String(format: "%.1f dB", p)))
            pictures.append(floor(ours))
        }
        rows.append(("picture: worst PSNR over the six views", String(format: "%.1f dB", worstOurs)))
        let png = arg("png", "")
        if !png.isEmpty {
            let top = concatenated(Array(pictures[0 ..< 3]), axis: 1), bottom = concatenated(Array(pictures[3 ..< 6]), axis: 1)
            writePNG(concatenated([top, bottom], axis: 0), to: png)
        }
    }
    report("golden \(name): autoencoder, decoder route \(exactRoute ? "exact (conv3d)" : "default")\n"
           + "stage                                     cosine    rel L2   left out")
    exit(0)
}

if mode == "step" {
    let step = Int(arg("step", "0"))!
    let (model, loadSeconds) = try loadModel()
    let s = try load(String(format: "step_%02d", step))
    let s0 = step == 0 ? s : try load("step_00")            // the text embedding is dumped with step 0 only
    let hidden = s["x"]!.expandedDimensions(axis: 0).asType(.bfloat16)
    let encoder = s0["ctx"]!.expandedDimensions(axis: 0).asType(.bfloat16)
    // Upstream multiplies the bf16 timestep by 1000 in bf16 before the sinusoid, so the model sees e.g. 644, not
    // 644.53. A value above 1 also switches off the DiT's own x1000.
    let timestep = (s["t"]!.asType(.bfloat16) * MLXArray(Float(1000)).asType(.bfloat16)).asType(.float32).reshaped(1)
    if step == 0 {
        let doubles = [0, 3, 7], singles = [0, 11, 23]
        model.blockTap = { name, a, b in
            let parts = name.split(separator: "_")
            guard let index = Int(parts[1]) else { return }
            do {
                if parts[0] == "double", doubles.contains(index), let b {
                    let g = try load(String(format: "block_double_%02d", index))
                    record("double \(index) image", a[0], g["img"]!)
                    record("double \(index) text", b[0], g["txt"]!)
                } else if parts[0] == "single", singles.contains(index) {
                    let g = try load(String(format: "block_single_%02d", index))
                    record("single \(index) text+image", a[0], g["seq"]!)
                }
            } catch {
                fatalError("\(name): \(error)")
            }
        }
    }
    let t1 = Date()
    let out = model(hidden, encoder: encoder, timestep: timestep, guidance: nil,
                    imgIds: s["x_ids"]!, txtIds: s0["ctx_ids"]!)
    eval(out)
    let passSeconds = Date().timeIntervalSince(t1)
    record("velocity", out[0], s["velocity"]!)
    report("golden \(name), step \(step), stream \(stream), tokens \(hidden.dim(1)), "
           + "load \(String(format: "%.1f", loadSeconds)) s, pass \(String(format: "%.1f", passSeconds)) s\n"
           + "stage                                     cosine    rel L2   left out")
    exit(0)
}

// select / sample
let resolution = Int(arg("resolution", "1024"))!
let inputs = try load("inputs")
let cond = try load("conditions")
let s0 = try load("step_00")
let sel = UltraTexSampler.select(targetMask: inputs["mask_target"]!, normalsMask: inputs["mask_normals"]!,
                                 referenceMask: inputs["mask_reference"]!)
let n = sel.targetCount
let canvasTokens = sel.canvasRows * sel.canvasCols
let ids = UltraTexSampler.sequenceIds(sel)
exact("position ids, \(sel.keptCount) kept tokens", ids, s0["x_ids"]!)
let refTokens = cond["ref_tokens"]!
let conditions = UltraTexSampler.conditionTokens(
    sel, normals: refTokens[0 ..< canvasTokens, 0...], reference: refTokens[canvasTokens..., 0...])
// The golden holds the encoder's fp32 latents; what enters the DiT is their bf16 rounding. Comparing after the
// same cast also checks that MLX rounds to bf16 the way torch does.
exact("kept normals + reference tokens (bf16)", conditions.asType(.bfloat16), s0["x"]![n..., 0...])
exact("text ids", UltraTexSampler.textIds(count: s0["ctx"]!.dim(0)), s0["ctx_ids"]!)
let ts = UltraTexSampler.timesteps(steps: 25, resolution: resolution)
for i in [0, 12, 24] {
    let g = try load(String(format: "step_%02d", i))
    exact("timestep \(i) as bf16", MLXArray([UltraTexSampler.bf16(ts[i])]), g["t"]!.reshaped(1))
}
if mode == "select" {
    report("golden \(name): token selection and schedule (target \(n), normals \(sel.normals.count), "
           + "reference \(sel.reference.count))")
    exit(0)
}

let (model, loadSeconds) = try loadModel()
let t1 = Date()
MLX.Memory.peakMemory = 0
let checkpoints = [12: try load("step_12")["x"]!, 24: try load("step_24")["x"]!]
let result = UltraTexSampler.sample(
    model: model, noise: s0["x"]![0 ..< n, 0...], conditions: conditions, sequenceIds: ids, text: s0["ctx"]!,
    steps: 25, resolution: resolution
) { i, x in
    if let g = checkpoints[i] { record("target tokens entering step \(i)", x, g[0 ..< n, 0...]) }
    FileHandle.standardError.write("step \(i) at \(String(format: "%.0f", Date().timeIntervalSince(t1))) s\n".data(using: .utf8)!)
}
let sampleSeconds = Date().timeIntervalSince(t1)
let peak = MLX.Memory.peakMemory
// [1, 128, rows, cols] -> tokens [rows*cols, 128]
let background = inputs["background_latent"]![0].transposed(1, 2, 0).reshaped(canvasTokens, 128)
let canvas = UltraTexSampler.composite(sel, target: result, background: background)
let savePath = arg("save-latents", "")
if !savePath.isEmpty { try MLX.save(arrays: ["canvas": canvas], url: URL(fileURLWithPath: savePath)) }
var all: [MLXArray] = [], allGolden: [MLXArray] = []
for (i, tile) in UltraTexSampler.tiles(canvas).enumerated() {
    let g = try load(String(format: "vae_decode_%02d", i))["input"]![0].transposed(1, 2, 0)
    record("composite latent, tile \(i)", tile, g)
    all.append(tile.reshaped(-1)); allGolden.append(g.reshaped(-1))
}
record("composite latent, all six tiles", concatenated(all), concatenated(allGolden))
report("golden \(name): 25 steps, stream \(stream), \(sel.keptCount) kept tokens, load \(String(format: "%.1f", loadSeconds)) s, "
       + "sampling \(String(format: "%.0f", sampleSeconds)) s (\(String(format: "%.1f", sampleSeconds / 25)) s per step), "
       + "peak MLX memory \(peak / (1 << 20)) MB")
