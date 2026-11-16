// ultratex-texture — texture a mesh from one picture of it.
//
//   swift run -c release ultratex-texture --glb <mesh.glb> --reference <picture.png> --out <textured.glb> --models <dir>
//        [--resolution 1024] [--seed 42] [--steps 25] [--atlas 2048]
//        [--views <six-views.png>] [--turn <turnaround.png>] [--report <report.json>] [--max-gb N]
//
// The picture must have the object cut out on a transparent background. The mesh must have UVs and face -Z in
// glTF terms after TRELLIS.2 / Pixal3D's export convention (front toward the viewer of the picture).
//
// --models is a folder with:
//   transformer/                         FLUX.2-klein-base-9B's diffusers transformer (gated, BFL non-commercial)
//   dit_lora.safetensors                 the UltraTex albedo LoRA (ModelScope Yibo-Zhang/UltraTex, flux2/lora)
//   vae/                                 the FLUX.2 autoencoder with UltraTex's decoder (convert_decoder.py)
//   empty_prompt_qwen3_8b.safetensors    the cached empty-prompt embedding
//
// The run is sized before anything is loaded: the six normal maps are rendered on the CPU, the kept tokens are
// counted, and the process gets a hard MLX memory cap from that (see UltraTexMemory). Weights are under BFL's
// FLUX Non-Commercial License v2.1.

import Foundation
import MLX
import UltraTex

func arg(_ name: String, _ fallback: String) -> String {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: "--" + name), i + 1 < a.count else { return fallback }
    return a[i + 1]
}
func say(_ text: String) { FileHandle.standardError.write((text + "\n").data(using: .utf8)!) }
func fail(_ text: String) -> Never { say("ultratex: " + text); exit(64) }

let glb = arg("glb", ""), referencePath = arg("reference", ""), outPath = arg("out", ""), models = arg("models", "")
if glb.isEmpty || referencePath.isEmpty || outPath.isEmpty || models.isEmpty {
    fail("usage: ultratex-texture --glb <mesh.glb> --reference <picture.png> --out <textured.glb> --models <dir> [--resolution 1024] [--seed 42] [--atlas 2048] [--views png] [--turn png] [--report json]")
}
let resolution = Int(arg("resolution", "1024"))!
guard resolution % 16 == 0, (512 ... 2048).contains(resolution) else { fail("--resolution must be a multiple of 16 between 512 and 2048") }
let seed = UInt64(arg("seed", "42"))!, steps = Int(arg("steps", "25"))!, atlasSize = Int(arg("atlas", "2048"))!

do {
    let started = Date()
    // 1. CPU: read the mesh and the picture, render, prepare, and size the run.
    let mesh = try UltraTexMesh.load(glb: URL(fileURLWithPath: glb))
    guard !mesh.uvs.isEmpty else { fail("the mesh has no UVs to bake onto") }
    let reference = try UltraTexInputs.squareReference(try UltraTexImage.loadStraight(referencePath))
    let prepared = UltraTexPipeline.prepare(mesh: mesh, reference: reference, resolution: resolution)
    let tokens = UltraTexPipeline.keptTokens(prepared)
    let prepareSeconds = Date().timeIntervalSince(started)
    // Measured: MLX's peak is about 17 GB plus 0.26 GB per thousand tokens with dense attention (32.3 GB at 57,237
    // tokens). The cap counts MLX's buffer cache too, which may hold 8 GB, so allow that and 3 GB of room.
    let capGB = Double(arg("max-gb", String(format: "%.0f", 28 + 0.26 * Double(tokens + 512) / 1000)))!
    let available = UltraTexMemory.guardProcess(capGB: capGB, label: "ultratex")
    say("ultratex: \(mesh.faces.count) faces, \(tokens) tokens at \(resolution)²; memory cap \(Int(capGB)) GB, \(Int(available)) GB available")

    // 2. Models.
    let root = URL(fileURLWithPath: models)
    AttentionUtils.backend = .denseChunked(maxScoreBytes: 4 << 30)
    let loadStart = Date()
    let dit = try UltraTexWeights.loadTransformer(directory: root.appendingPathComponent("transformer"),
                                                  lora: root.appendingPathComponent("dit_lora.safetensors")).model
    dit.streamDType = .bfloat16
    let vae = try UltraTexVAE.load(directory: root.appendingPathComponent("vae"))
    guard let text = try MLX.loadArrays(url: root.appendingPathComponent("empty_prompt_qwen3_8b.safetensors"))["ctx"] else {
        fail("empty_prompt_qwen3_8b.safetensors has no `ctx`")
    }
    let loadSeconds = Date().timeIntervalSince(loadStart)

    // 3. Generate the six views.
    let pipeline = UltraTexPipeline(model: dit, vae: vae, text: text)
    let runStart = Date()
    let out = pipeline.run(prepared: prepared, steps: steps, seed: seed) { i in
        if i % 5 == 0 { say("ultratex: step \(i) of \(steps), \(Int(Date().timeIntervalSince(runStart))) s") }
    }
    let peakMLX = MLX.Memory.peakMemory
    if !arg("views", "").isEmpty { try out.canvas.writePNG(arg("views", "")) }

    // 4. Bake and write.
    let bakeStart = Date()
    let baked = UltraTexBake.bake(mesh: mesh, views: out.views, atlasSize: atlasSize)
    try UltraTexGLB.replaceBaseColour(in: URL(fileURLWithPath: glb), with: try baked.atlas.pngData(), to: URL(fileURLWithPath: outPath))
    let bakeSeconds = Date().timeIntervalSince(bakeStart)

    let turn = arg("turn", "")
    if !turn.isEmpty, let old = mesh.baseColourImage {
        let cell = 384
        var sheet = [UInt8](repeating: 255, count: cell * 8 * cell * 2 * 4)
        for (row, texture) in [try UltraTexImage.decode(old), baked.atlas].enumerated() {
            for col in 0 ..< 8 {
                let view = UltraTexRenderer.textured(mesh, texture: texture,
                                                     camera: UltraTexRenderer.orbitCamera(azimuth: Double(col) * 45, elevation: 15), size: cell)
                for p in 0 ..< cell * cell where view.rgba[p * 4 + 3] != 0 {
                    let to = ((row * cell + p / cell) * cell * 8 + col * cell + p % cell) * 4
                    sheet[to] = view.rgba[p * 4]; sheet[to + 1] = view.rgba[p * 4 + 1]; sheet[to + 2] = view.rgba[p * 4 + 2]
                }
            }
        }
        try UltraTexImage(width: cell * 8, height: cell * 2, rgba: sheet).writePNG(turn)
    }

    func gb(_ bytes: Int) -> Double { (Double(bytes) / 1_073_741_824 * 100).rounded() / 100 }
    func r1(_ x: Double) -> Double { (x * 10).rounded() / 10 }
    let report: [String: Any] = [
        "glb": glb, "reference": referencePath, "out": outPath, "resolution": resolution, "seed": seed, "steps": steps,
        "atlas": atlasSize, "faces": mesh.faces.count, "kept_tokens": tokens,
        "seconds": ["prepare": r1(prepareSeconds), "load": r1(loadSeconds), "encode": r1(out.seconds["encode"] ?? 0),
                    "sample": r1(out.seconds["sample"] ?? 0), "decode": r1(out.seconds["decode"] ?? 0), "bake": r1(bakeSeconds),
                    "total": r1(Date().timeIntervalSince(started))],
        "memory_gb": ["cap": capGB, "peak_mlx": gb(peakMLX), "peak_footprint": gb(UltraTexMemory.peakFootprintBytes()),
                      "available_at_start": r1(available)],
        "bake": ["painted": (baked.paintedFraction * 1000).rounded() / 1000, "filled": (baked.filledFraction * 1000).rounded() / 1000,
                 "texels_on_mesh": baked.texelsOnMesh],
        "attention": "dense, chunked over query rows", "stream": "bf16", "passes_per_step": 1,
    ]
    let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    if !arg("report", "").isEmpty { try json.write(to: URL(fileURLWithPath: arg("report", ""))) }
    print(String(data: json, encoding: .utf8)!)
} catch {
    fail("\(error)")
}
