# mlx-ultratex-swift (WIP)

Swift/MLX port of [UltraTex](https://github.com/yiboz2001/UltraTex): texture an existing mesh from six
normal-map views and one reference picture, on FLUX.2 klein base 9B. Task AB-T-0489, phase P2.

**Status (2026-11-15): the DiT is at parity; nothing here produces a texture yet.** The real klein 9B weights
load with the UltraTex LoRA merged in Swift, and one DiT pass matches the CUDA goldens at 1024² (velocity cosine
0.99996, 0.99995 and 0.99978 at steps 0, 12 and 24; AB-R-0892). Not started: the sampler with token selection,
the VAE with the fine-tuned decoder, the renderer and the bake.

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
- `Sources/ultratex-bench`: seconds and memory for one DiT pass at a given token count.
- `Sources/ultratex-gate`: one DiT pass on a golden step's inputs, compared with the golden velocity and block
  outputs.

```bash
swift run -c release ultratex-gate --transformer <FLUX.2-klein-base-9B>/transformer --lora <dit_lora.safetensors> \
    --golden /Volumes/DEV_VOL1/Evidence/ultratex-oracle-2026-11-15/golden_alb_dress_r1024_dense_s25_cam_photo/st
```

Weights on this machine: `/Volumes/DEV_VOL1/Models/models--black-forest-labs--FLUX.2-klein-base-9B` (gated) and
`/Volumes/DEV_VOL1/Models/ultratex-upstream` (the LoRA and decoder from ModelScope, sha256-checked).

```bash
swift run -c release ultratex-bench --tokens 19174
```

Requires mlx-swift ≥ 0.32.3 (the half-precision GEMM fix); built against 0.33.0.

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
