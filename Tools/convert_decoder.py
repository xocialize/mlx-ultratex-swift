"""UltraTex's foreground-aware FLUX.2 decoder -> a diffusers `vae/` folder the fleet's Flux2VAE package loads.

    python -I convert_decoder.py <stock diffusers vae dir> <decoder.pt> <out vae dir> [<BFL ae.safetensors>]

`decoder.pt` is upstream's `ae.decoder.state_dict()` (BFL / LDM names, bf16). The output is the stock VAE with
its `decoder.*` and `post_quant_conv.*` tensors replaced, so the encoder, `quant_conv` and the `bn` statistics
stay BFL's. Name map (the usual LDM -> diffusers one): `up.i` is `up_blocks.(3-i)`, `mid.block_k` is
`mid_block.resnets.(k-1)`, `mid.attn_1.{q,k,v,proj_out}` are 1x1 convs that become the Linear
`attentions.0.{to_q,to_k,to_v,to_out.0}`, `nin_shortcut` is `conv_shortcut`, `norm_out` is `conv_norm_out`.

With BFL's own `ae.safetensors` as a fourth argument, the encoder, `quant_conv` and the `bn` statistics are taken
from it in float32 instead of the diffusers copy, which stores them rounded to bf16. Upstream encodes with the
float32 file, so this is the faithful encoder. Its names map the same way, except that `down.i` keeps its index.

Every tensor must map both ways with equal shapes. As a check on the map itself, each fine-tuned tensor is
compared with the stock one it replaces: a light fine-tune stays close, a mis-mapped block would not.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

stock_dir, decoder_pt, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
bfl_ae = Path(sys.argv[4]) if len(sys.argv) > 4 else None


def encoder_target(key: str):
    if key.startswith('bn.'):
        return key
    if not key.startswith('encoder.'):
        return None                                   # the stock decoder: replaced by the fine-tuned one
    k = key[len('encoder.'):]
    if k.startswith('quant_conv.'):
        return k
    k = re.sub(r'^mid\.block_(\d)\.', lambda m: f'mid_block.resnets.{int(m.group(1)) - 1}.', k)
    k = re.sub(r'^mid\.attn_1\.norm\.', 'mid_block.attentions.0.group_norm.', k)
    k = re.sub(r'^mid\.attn_1\.proj_out\.', 'mid_block.attentions.0.to_out.0.', k)
    k = re.sub(r'^mid\.attn_1\.([qkv])\.', r'mid_block.attentions.0.to_\1.', k)
    k = re.sub(r'^down\.(\d)\.block\.(\d)\.', r'down_blocks.\1.resnets.\2.', k)
    k = re.sub(r'^down\.(\d)\.downsample\.conv\.', r'down_blocks.\1.downsamplers.0.conv.', k)
    k = k.replace('.nin_shortcut.', '.conv_shortcut.')
    k = re.sub(r'^norm_out\.', 'conv_norm_out.', k)
    return 'encoder.' + k


def target(key: str) -> str:
    if key.startswith('post_quant_conv.'):
        return key
    k = key
    k = re.sub(r'^mid\.block_(\d)\.', lambda m: f'mid_block.resnets.{int(m.group(1)) - 1}.', k)
    k = re.sub(r'^mid\.attn_1\.norm\.', 'mid_block.attentions.0.group_norm.', k)
    k = re.sub(r'^mid\.attn_1\.proj_out\.', 'mid_block.attentions.0.to_out.0.', k)
    k = re.sub(r'^mid\.attn_1\.([qkv])\.', r'mid_block.attentions.0.to_\1.', k)
    k = re.sub(r'^up\.(\d)\.block\.(\d)\.', lambda m: f'up_blocks.{3 - int(m.group(1))}.resnets.{m.group(2)}.', k)
    k = re.sub(r'^up\.(\d)\.upsample\.conv\.', lambda m: f'up_blocks.{3 - int(m.group(1))}.upsamplers.0.conv.', k)
    k = k.replace('.nin_shortcut.', '.conv_shortcut.')
    k = re.sub(r'^norm_out\.', 'conv_norm_out.', k)
    return 'decoder.' + k


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


stock_file = stock_dir / 'diffusion_pytorch_model.safetensors'
stock = load_file(str(stock_file))
tuned = torch.load(decoder_pt, map_location='cpu', weights_only=True)
want = {k for k in stock if k.startswith('decoder.') or k.startswith('post_quant_conv.')}
out, cos, moved = dict(stock), [], []
for key, value in tuned.items():
    t = target(key)
    assert t in want, f'{key} -> {t} is not a decoder tensor of the stock VAE'
    old = stock[t]
    if value.ndim == 4 and old.ndim == 2:                 # attention 1x1 conv -> Linear
        value = value[:, :, 0, 0]
    assert value.shape == old.shape and value.dtype == old.dtype, f'{key}: {tuple(value.shape)} {value.dtype} vs {tuple(old.shape)} {old.dtype}'
    a, b = value.float().flatten(), old.float().flatten()
    cos.append((float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)), t))
    moved.append(float((a - b).norm() / (b.norm() + 1e-30)))
    out[t] = value.contiguous()
    want.discard(t)
assert not want, f'stock decoder tensors left unreplaced: {sorted(want)[:5]}'
encoder_report = {}
if bfl_ae:
    full = load_file(str(bfl_ae))
    rest = {k for k in stock if not (k.startswith('decoder.') or k.startswith('post_quant_conv.'))}
    worst = 0.0
    for key, value in full.items():
        t = encoder_target(key)
        if t is None:
            continue
        assert t in rest, f'{key} -> {t} is not an encoder tensor of the diffusers VAE'
        old = stock[t]
        if value.ndim == 4 and old.ndim == 2:
            value = value[:, :, 0, 0]
        assert value.shape == old.shape, f'{key}: {tuple(value.shape)} vs {tuple(old.shape)}'
        if value.is_floating_point():                    # the diffusers copy must be this tensor rounded to bf16
            worst = max(worst, float((value.to(torch.bfloat16).float() - old.float()).abs().max()))
        out[t] = value.contiguous()
        rest.discard(t)
    assert not rest, f'encoder tensors left on the bf16 copy: {sorted(rest)[:5]}'
    encoder_report = {'encoder_source': str(bfl_ae), 'encoder_source_sha256': sha256(bfl_ae), 'encoder_dtype': 'float32',
                      'max_difference_to_diffusers_copy_after_bf16_rounding': worst}
out_dir.mkdir(parents=True, exist_ok=True)
save_file(out, str(out_dir / 'diffusion_pytorch_model.safetensors'))
shutil.copy(stock_dir / 'config.json', out_dir / 'config.json')
cos.sort()
report = {'replaced_tensors': len(tuned), 'kept_stock_tensors': len(stock) - len(tuned),
          'cosine_to_stock_min': round(cos[0][0], 6), 'cosine_to_stock_min_tensor': cos[0][1],
          'cosine_to_stock_median': round(cos[len(cos) // 2][0], 6),
          'relative_change_max': round(max(moved), 5), 'relative_change_median': round(sorted(moved)[len(moved) // 2], 5),
          'source': {'stock_vae': str(stock_file), 'stock_vae_sha256': sha256(stock_file),
                     'decoder_pt': str(decoder_pt), 'decoder_pt_sha256': sha256(decoder_pt)},
          'output_sha256': sha256(out_dir / 'diffusion_pytorch_model.safetensors'), **encoder_report}
(out_dir / 'SOURCE.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
