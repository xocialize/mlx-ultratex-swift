# Third-party licences

This package's own code is MIT (`LICENSE`). What it builds on and what it runs:

| What | Licence | Where it enters |
|---|---|---|
| `Sources/UltraTex/DiT.swift` | MIT, xocialize/flux2-klein-swift @ v0.11.0 (`LICENSE-flux2-klein-swift`), itself a port of mflux's FLUX.2 transformer (MIT) | vendored source |
| flux2-vae-mlx-swift, mlx-exact-conv-swift | MIT (Xocialize) | package dependencies |
| mlx-swift | MIT (Apple / ml-explore) | package dependency |
| UltraTex (Yibo Zhang et al., SIGGRAPH Asia 2026) | code MIT | the method; nothing is copied verbatim, the sampler and data path follow `train_flux2.py` and `ultratex/data/dataset.py` @ a726678 |
| Pillow (HPND), OpenCV (Apache-2.0) | — | `Inputs.swift` re-implements their 8-bit resize, paste and dilation arithmetic so inputs match upstream's bytes; no code is copied |

## Weights: not in this repository, and not permissive

Nothing here ships weights. A run needs, from their publishers:

- **FLUX.2 klein base 9B** (Black Forest Labs): **FLUX Non-Commercial License v2.1**. Gated on Hugging Face.
- **The UltraTex LoRA and decoder** (ModelScope `Yibo-Zhang/UltraTex`, labelled MIT there): fine-tuned from the
  above, so under BFL's licence they are Derivatives and its terms apply to running them.
- **The FLUX.2 autoencoder** (BFL): Apache-2.0.
- **The empty-prompt embedding** comes from Qwen3-8B (Apache-2.0).

Because of the first two, this package is for the Gaea plugin channel only and is never a shipping-app asset
(decision AB-D-0231). A notice shipped with any build must carry BFL's Attribution Notice and say the model was
modified (LoRA merged, converted).
