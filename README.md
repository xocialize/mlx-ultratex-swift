# mlx-ultratex-swift (WIP)

Swift/MLX port of [UltraTex](https://github.com/yiboz2001/UltraTex): texture an existing mesh from six
normal-map views and one reference picture, on FLUX.2 klein base 9B. Task AB-T-0489, phase P2.

**Status (2026-11-15): a command-line tool textures a mesh from one picture on a Mac, with every stage gated
against a CUDA oracle. There is no engine package or plugin yet.**
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
- `ultratex-texture` runs the whole job: mesh and picture in, textured GLB out. Measured on four more corpus
  items at 1024² (4 to 7 minutes, 27 to 31 GB) and on the shoes at 2048² (31 minutes, 40 GB); see "Runs" below.
- Known limit: at 1024² the result depends on the seed for light-coloured objects. See "Seeds" below.
- Not started: the engine package and the plugin.

**Licence posture (AB-D-0231):** a Gaea plugin only. The 9B base and the UltraTex LoRAs are under BFL's FLUX
Non-Commercial License v2.1; the weight licence is declared in this package and never allowlisted. That is why
the 9B model lives here and not in the Apache `flux2-klein-swift`.

## Use

```bash
swift run -c release ultratex-texture --glb mesh.glb --reference picture.png --out textured.glb \
    --models <models folder> [--resolution 1024] [--seed 42] [--views six-views.png] [--turn turnaround.png] \
    [--report report.json]
```

- The picture must have the object cut out on a transparent background. It is cropped to the object and
  squared for you.
- The mesh must have UVs, and its front must face the way TRELLIS.2 / Pixal3D exports face.
- `--resolution` is the size of each of the six generated views. 1024 takes four to seven minutes on an M5 Max
  and 27 to 31 GB of memory; 2048 takes half an hour to an hour and 40 GB or more (dense attention).
- `--seed` picks the starting noise. It is MLX's noise, so a seed does not reproduce upstream's picture for the
  same number. If a light-coloured object comes out flat or grey at 1024², try another seed or 2048².
- Only the base-colour texture is replaced. Roughness/metallic is left as the mesh had it.

The models folder holds four things, none of them in this repository:

| Entry | What | From |
|---|---|---|
| `transformer/` | FLUX.2 klein base 9B, diffusers layout | `black-forest-labs/FLUX.2-klein-base-9B` (gated, non-commercial) |
| `dit_lora.safetensors` | the UltraTex albedo LoRA | ModelScope `Yibo-Zhang/UltraTex`, `flux2/lora/` |
| `vae/` | the FLUX.2 autoencoder with UltraTex's fine-tuned decoder | built by `Tools/convert_decoder.py` (Python, torch + safetensors) from a diffusers FLUX.2 `vae/` folder, UltraTex's `flux2/decoder.pt` and BFL's `ae.safetensors` |
| `empty_prompt_qwen3_8b.safetensors` | the text embedding of the empty prompt, key `ctx`, [512, 12288] | Qwen3-8B hidden states 9, 18 and 27 of the chat-templated empty prompt, as upstream's `Qwen3Embedder` computes them. The copy used here was dumped by the CUDA oracle run; a standalone script for it is not written yet |

Symlinks into a download cache are fine. There is no downloader yet.

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
  row-band path, with UltraTex's fine-tuned decoder (converted by `Tools/convert_decoder.py`).
- `Sources/UltraTex/Mesh.swift`, `Renderer.swift`: a GLB reader (frame, scale, smooth normals) and a CPU
  rasterizer for the six normal maps. No GPU or MLX memory.
- `Sources/UltraTex/Inputs.swift`, `Images.swift`: resize, paste on white, the 2x3 canvas and the token masks,
  with Pillow's and OpenCV's arithmetic.
- `Sources/UltraTex/Pipeline.swift`: the whole run, mesh and picture to six views.
- `Sources/UltraTex/Bake.swift`, `GLBTexture.swift`: the six views onto the atlas with a fill and padding, and
  the GLB written back.
- `Sources/UltraTex/MemoryGuard.swift`: a hard cap on this process's MLX memory. See "Memory" below.
- `Sources/ultratex-texture`: the command-line tool above. It renders and counts tokens before loading any
  model, so the memory cap is set from the job's real size.
- `Sources/ultratex-bench`: seconds and memory for one DiT pass at a given token count.
- `Sources/ultratex-gate`: the port against the goldens. `--mode step` runs one DiT pass on a golden step's
  inputs; `--mode select` checks token selection and the schedule without a model; `--mode sample` runs all 25
  steps from the golden's noise; `--mode vae` gates the encoder and decoder and can decode the Swift run's latents;
  `--mode render` and `--mode inputs` gate the renderer and the input preparation; `--mode pipeline` runs a whole
  job from a GLB and a picture; `--mode bake` puts a six-view canvas on a mesh and writes the GLB.

```bash
swift run -c release ultratex-gate --mode sample --resolution 1024 \
    --transformer <FLUX.2-klein-base-9B>/transformer --lora <dit_lora.safetensors> \
    --golden <goldens>/golden_alb_dress_r1024_dense_s25_cam_photo/st
```

A whole run (about five minutes at 1024²):

```bash
swift run -c release ultratex-gate --mode pipeline --resolution 1024 --seed 42 \
    --glb <mesh.glb> --reference <picture.png> --png <six-views.png> \
    --transformer <FLUX.2-klein-base-9B>/transformer --lora <dit_lora.safetensors> \
    --vae <models>/vae --text <models>/empty_prompt_qwen3_8b.safetensors
```

Then onto the mesh (a second or two, no model):

```bash
swift run -c release ultratex-gate --mode bake --glb <mesh.glb> --views <six-views.png> \
    --out-glb <textured.glb> --turn <turnaround.png>
```

The gate needs the CUDA goldens, which are not public; its commands above show the shape of a run.

```bash
swift run -c release ultratex-bench --tokens 19174
```

Requires mlx-swift ≥ 0.32.3 (the half-precision GEMM fix); built against 0.33.0.

## Memory

Both tools call `UltraTexMemory.guardProcess(capGB:)` first. It refuses to start unless the machine has the cap
plus 16 GB available, limits MLX's buffer cache to 8 GB, and runs a watchdog that ends the process when active
plus cached MLX memory passes the cap (`--max-gb`; defaults 16 GB for `select` and `vae`, 32 GB for sampling,
and a token-count estimate for the bench). MLX's own memory limit is a guideline and does not stop paging.
`ultratex-texture` sets its cap from the job: 28 GB plus 0.26 GB per thousand tokens (29 to 35 GB at 1024²,
43 GB for the shoes at 2048²). The process footprint is 6 to 7 GB above MLX's peak.

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

## Runs

`ultratex-texture` on an M5 Max (128 GB), release build, dense attention, bf16 stream, 25 steps, one pass per
step, 2048² atlas. Meshes are Pixal3D outputs of about 950,000 faces; the picture is the one the mesh was made
from. Other sessions were using the machine during some runs, so times are upper bounds.

| Item | Views | Kept tokens | Sampling | Whole run | Peak MLX | Peak footprint | Atlas painted by a view |
|---|---|---|---|---|---|---|---|
| shoes | 1024² | 15,882 | 222 s | 246 s | 21.5 GB | 27.2 GB | 21 % |
| hair | 1024² | 17,010 | 278 s | 309 s | 22.1 GB | 27.8 GB | 33 % |
| dress (gate run, AB-R-0896) | 1024² | 19,174 | 288 s | 310 s | 22.4 GB | 29 GB | 64 % |
| skirt | 1024² | 22,904 | 385 s | 407 s | 23.3 GB | 30.3 GB | 76 % |
| top | 1024² | 24,738 | 415 s | 437 s | 23.8 GB | 30.9 GB | 71 % |
| shoes | 2048² | 57,237 | 1,804 s | 1,866 s | 32.3 GB | 39.6 GB | 21 % |

The rest of the atlas is filled from the nearest painted surface. Loading takes 8 to 10 s, encoding 6 s at
1024² and 26 s at 2048², decoding 6 to 24 s, the bake 1 to 3 s.

## Seeds

At 1024² the result on light-coloured objects depends on the starting noise. With seed 42 the white shoes came
out flat grey with stray orange strokes and the light-blue top came out flat and too dark; with seeds 7 and 123
the shoes are white and with seed 7 the top has its colours. The dark items (dress, hair, skirt) were right
with seed 42. At 2048² the shoes are white and clean with seed 42. Upstream on CUDA at 1024² with its own
seed-42 noise also gave white shoes, so this is the draw and not the port: one DiT pass on the shoes' 2048²
golden inputs matches CUDA's block-sparse velocity at cosine 0.993 and 0.991 (steps 12 and 24), where CUDA's
own dense and sparse passes differ by 0.9975 at step 0 on the dress.

Mean lightness L of the sides and back, against the reference picture's:

| Item | Reference | Seed 42 | Seed 7 | Seed 123 | CUDA |
|---|---|---|---|---|---|
| shoes 1024² | 80.2 | 72.4 | 85.6 | 85.4 | 80.2 (1024² dense) |
| shoes 2048² | 80.2 | 87.6 | | | 81.9 (2048² dense) |
| top 1024² | 71.9 | 64.9 | 73.7 | | 74.4 (2048² sparse) |

Seed 7 has not been run on the dark items, so the default stays 42 for now.

## Plan

Dense attention only in the first version: 1024² as the default tier, 2048² as a slow best tier. Then, each
step gated against the CUDA goldens (`DEV/UltraTexOracle`, `--plan golden`): weights and the LoRA merge, DiT
parity, token drop and the sampler, the VAE with the fine-tuned decoder, the G-buffer renderer, the bake with
a fill. Block-sparse attention is deferred until a kernel is shown to pay.
