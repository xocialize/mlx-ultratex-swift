# mlx-ultratex-swift (WIP)

Swift/MLX port of [UltraTex](https://github.com/yiboz2001/UltraTex): texture an existing mesh from six
normal-map views and one reference picture, on FLUX.2 klein base 9B. Task AB-T-0489, phase P2.

**Status (2026-11-15): every stage from a mesh and a picture to a textured GLB exists and has run on a Mac, at
parity with the CUDA oracle where there is something to compare with. It is still driven through the gate tool;
there is no package or plugin yet.**
- The real klein 9B weights load with the UltraTex LoRA merged in Swift, and one DiT pass matches the CUDA goldens
  at 1024² (velocity cosine 0.99996, 0.99995 and 0.99978 at steps 0, 12 and 24; AB-R-0892).
- Token selection, position ids and the schedule are identical to the goldens at 1024² and 2048², and a
  free-running 25-step run from the golden's noise ends at cosine 0.99989 on the latents upstream decodes
  (286 s, 22 GB peak; AB-R-0893).
- The autoencoder is gated with UltraTex's fine-tuned decoder: decode ≥ 59.6 dB at 1024² and ≥ 55.4 dB at 2048²
  against the goldens, encode cosine ≥ 0.99999. End to end in Swift (25 steps, then decode) the dress at 1024²
  matches the CUDA picture to 46.5 dB (AB-R-0894).
- The renderer and the input preparation are gated: the Swift normal maps match the oracle's on all five corpus
  meshes (coverage identical, normals byte-identical on over 99.9 % of pixels) and the canvas, reference tensor
  and token masks reproduce upstream's dataset exactly (AB-R-0895).
- A whole run works: the dress at 1024² from its GLB and photo in 301 s with 29 GB of process memory, matching
  the CUDA picture to 44.0 dB when started from the same noise (AB-R-0896).
- The bake puts the six views on the mesh's atlas in 1–3 s on the CPU, fills what no view reaches from the
  nearest painted surface, pads the charts and writes a GLB with only the base-colour image changed (AB-R-0897).
- Not started: a command-line tool for a whole job, measured footprints, the engine package and the plugin.

**Licence posture (AB-D-0231):** a Gaea plugin only. The 9B base and the UltraTex LoRAs are under BFL's FLUX
Non-Commercial License v2.1; the weight licence is declared in this package and never allowlisted. That is why
the 9B model lives here and not in the Apache `flux2-klein-swift`.

## What is here

- `Sources/UltraTex/DiT.swift`: `Transformer.swift` from `xocialize/flux2-klein-swift` @ v0.11.0 (MIT,
  `LICENSE-flux2-klein-swift`), with four changes marked `UltraTex:` — a size config (`DiTConfig.klein9B`), a
  pluggable `AttentionBackend` (`dense`, `denseChunked`), an optional bf16 activation stream (`streamDType`),
  and the `to_out` row-chunk workaround for mlx#3797 off by default. Ask AB-A-0296 proposes moving the
  attention hook upstream so this copy can go.
- `Sources/UltraTex/Weights.swift`: loads BFL's diffusers `transformer/` folder strictly and merges the UltraTex
  LoRA (PEFT keys on BFL names; fused q/k/v updates cut into thirds), with upstream's rounding points.
- `Sources/UltraTex/Sampler.swift`: token selection from the masks, position ids, the schedule, the
  foreground-only Euler loop with upstream's bf16 rounding points, and the latent canvas upstream decodes.
- `Sources/UltraTex/VAE.swift`: the FLUX.2 autoencoder through the fleet's `Flux2VAE` package, always on its
  row-band path, with UltraTex's fine-tuned decoder (converted by `DEV/UltraTexOracle/tools/convert_decoder.py`
  into `/Volumes/DEV_VOL1/Models/ultratex-mlx/vae`).
- `Sources/UltraTex/Mesh.swift`, `Renderer.swift`: a GLB reader (frame, scale, smooth normals) and a CPU
  rasterizer for the six normal maps. No GPU or MLX memory.
- `Sources/UltraTex/Inputs.swift`, `Images.swift`: resize, paste on white, the 2x3 canvas and the token masks,
  with Pillow's and OpenCV's arithmetic.
- `Sources/UltraTex/Pipeline.swift`: the whole run, mesh and picture to six views.
- `Sources/UltraTex/Bake.swift`, `GLBTexture.swift`: the six views onto the atlas with a fill and padding, and
  the GLB written back.
- `Sources/UltraTex/MemoryGuard.swift`: a hard cap on this process's MLX memory. See "Memory" below.
- `Sources/ultratex-bench`: seconds and memory for one DiT pass at a given token count.
- `Sources/ultratex-gate`: the port against the goldens. `--mode step` runs one DiT pass on a golden step's
  inputs; `--mode select` checks token selection and the schedule without a model; `--mode sample` runs all 25
  steps from the golden's noise; `--mode vae` gates the encoder and decoder and can decode the Swift run's latents;
  `--mode render` and `--mode inputs` gate the renderer and the input preparation; `--mode pipeline` runs a whole
  job from a GLB and a picture; `--mode bake` puts a six-view canvas on a mesh and writes the GLB.

```bash
swift run -c release ultratex-gate --mode sample --resolution 1024 \
    --transformer <FLUX.2-klein-base-9B>/transformer --lora <dit_lora.safetensors> \
    --golden /Volumes/DEV_VOL1/Evidence/ultratex-oracle-2026-11-15/golden_alb_dress_r1024_dense_s25_cam_photo/st
```

A whole run (about five minutes at 1024²):

```bash
swift run -c release ultratex-gate --mode pipeline --resolution 1024 --seed 42 \
    --glb <mesh.glb> --reference <picture.png> --png <six-views.png> \
    --transformer <FLUX.2-klein-base-9B>/transformer --lora <dit_lora.safetensors> \
    --vae /Volumes/DEV_VOL1/Models/ultratex-mlx/vae \
    --text /Volumes/DEV_VOL1/Models/ultratex-mlx/empty_prompt_qwen3_8b.safetensors
```

Then onto the mesh (a second or two, no model):

```bash
swift run -c release ultratex-gate --mode bake --glb <mesh.glb> --views <six-views.png> \
    --out-glb <textured.glb> --turn <turnaround.png>
```

Weights on this machine: `/Volumes/DEV_VOL1/Models/models--black-forest-labs--FLUX.2-klein-base-9B` (gated) and
`/Volumes/DEV_VOL1/Models/ultratex-upstream` (the LoRA and decoder from ModelScope, sha256-checked).

```bash
swift run -c release ultratex-bench --tokens 19174
```

Requires mlx-swift ≥ 0.32.3 (the half-precision GEMM fix); built against 0.33.0.

## Memory

Both tools call `UltraTexMemory.guardProcess(capGB:)` first. It refuses to start unless the machine has the cap
plus 16 GB available, limits MLX's buffer cache to 8 GB, and runs a watchdog that ends the process when active
plus cached MLX memory passes the cap (`--max-gb`; defaults 16 GB for `select` and `vae`, 32 GB for sampling,
and a token-count estimate for the bench). MLX's own memory limit is a guideline and does not stop paging.

This exists because of a hang on 2026-11-15 (lesson AB-L-0342): the autoencoder gate ran untiled at 2048², where
one encode needs a 17 GB attention score matrix and one decode about 42 GB, in a process that never cleared
MLX's cache, on a machine other sessions were using. Row-banded, the same gate peaks at 10.1 GB.

## Timing (AB-R-0891)

M5 Max, 128 GB, release build, random 9B-shaped bf16 weights, dense attention, one pass per sampling step.

| Kept tokens | Stands for | Seconds per pass | 25 steps | Peak MLX memory |
|---|---|---|---|---|
| 19,174 | dress at 1024² | 11.6 | 4.8 min | 21.9 GB |
| 57,237 | shoes at 2048² | 65.6 | 27.3 min | 31.5 GB |
| 69,520 | dress at 2048² | 95.3 | 39.7 min | 34.6 GB |

A bf16 activation stream is 1.83× faster than the fp32 one `flux2-klein-swift` runs today (21.3 s at 19,174
tokens). Chunking attention over query rows costs nothing.

## Plan

Dense attention only in the first version: 1024² as the default tier, 2048² as a slow best tier. Then, each
step gated against the CUDA goldens (`DEV/UltraTexOracle`, `--plan golden`): weights and the LoRA merge, DiT
parity, token drop and the sampler, the VAE with the fine-tuned decoder, the G-buffer renderer, the bake with
a fill. Block-sparse attention is deferred until a kernel is shown to pay.
