"""Loader for the Ressl ModelOpt NVFP4 checkpoint (Ornith-1.5 35B uncensored).

Nothing here is shared with the TorchAO-affine (`qwen35/`) path: this
checkpoint keeps everything bf16-dense EXCEPT the 256 switch experts per
layer, which are NVIDIA ModelOpt NVFP4 (E2M1 values packed 2/byte, e4m3
block scales per 16 inputs, one fp32 global scale per matrix).

Dequant (inference with fp32 activations; the stored input_scale only
scales quantized activations, so it is correctly ignored here):

    w_hat = E2M1(q) * block_scale[b] * global_scale
"""
from __future__ import annotations

import os
import pathlib

import torch

MODEL_DIR = pathlib.Path(
    os.environ.get(
        "FTV_RESSL_DIR",
        "/home/cus/.Llama/Models Dir/hub/"
        "models--ressl--Ornith-1.5-35B-A3B-uncensored-NVFP4/"
        "snapshots/64ec91a170f0d42048db662bf4ad6a8d54b680a9",
    )
)
PREFIX = "model.language_model"

# OCP E2M1 (NVFP4) codebook, index = nibble bits S E E M.
E2M1 = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

GROUP = 16  # NVFP4 block size (ModelOpt group_size)


def _nibble_order() -> str:
    order = os.environ.get("FTV_NIBBLE", "low")
    if order not in ("low", "high"):
        raise ValueError(f"FTV_NIBBLE must be low|high, got {order!r}")
    return order


def unpack_e2m1(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """Unpack U8 payload to E2M1 nibble indices ``[..., cols]``.

    Default ``low``: value 2k comes from the low nibble (universal
    convention; validated by parity before any generation run).
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected U8 NVFP4 payload, got {packed.dtype}")
    words = packed.shape[-1]
    if words * 2 != cols:
        raise ValueError(f"packed width {words} != cols/2 for cols={cols}")
    lead = packed.shape[:-1]
    flat = packed.reshape(-1, words).to(torch.int64)
    lo = flat & 0xF
    hi = (flat >> 4) & 0xF
    if _nibble_order() == "low":
        out = torch.stack([lo, hi], dim=-1).reshape(-1, cols)
    else:
        out = torch.stack([hi, lo], dim=-1).reshape(-1, cols)
    return out.reshape(*lead, cols)


def pack_scales_u32(scales: torch.Tensor) -> torch.Tensor:
    """Pack e4m3 block scales (``[rows, cols/16]`` float8_e4m3fn) into
    U32 words (``[rows, cols/64]``) for the NVFP4 shader: 4 scale bytes
    per word, low byte first (byte g lives in word g>>2 at byte g&3).
    The float8 bytes are REINTERPRETED as raw bytes, never converted
    (a value conversion of the decimals would zero them)."""
    if scales.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn scales, got {scales.dtype}")
    raw = scales.contiguous().reshape(-1).view(torch.uint8)
    pad = (-raw.numel()) % 4
    if pad:
        raw = torch.cat([raw, torch.zeros(pad, dtype=torch.uint8)])
    b = raw.reshape(-1, 4)
    # Byte packing in int32, not int64 (task #17: halves scale-pack memory
    # traffic on the decode miss path).  The high byte's shift overflows
    # into the sign bit but keeps the two's-complement bit pattern, which
    # the shader reads back as u32 — bit-identical to the int64 version.
    i0 = b[:, 0].to(torch.int32)
    words = i0 | (b[:, 1].to(torch.int32) << 8) \
               | (b[:, 2].to(torch.int32) << 16) \
               | (b[:, 3].to(torch.int32) << 24)
    return words


def dequantize_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Full NVFP4 dequant to FP32.

    Shapes: ``packed [rows, cols/2]`` U8, ``scales [rows, cols/16]`` e4m3,
    ``global_scale`` scalar fp32. Returns ``[rows, cols]`` fp32.
    """
    rows, half = packed.shape[-2], packed.shape[-1]
    cols = half * 2
    groups = cols // GROUP
    if tuple(scales.shape[-2:]) != (rows, groups):
        raise ValueError(
            f"scales {tuple(scales.shape)} incompatible with [{rows}, {cols}]"
        )
    q = unpack_e2m1(packed, cols).float()
    if _nibble_order() == "low":
        vals = E2M1[q.to(torch.int64)]
    else:
        vals = E2M1[q.to(torch.int64)]
    scale = scales.float().repeat_interleave(GROUP, dim=-1)
    try:
        gs = float(global_scale.item()) if isinstance(global_scale, torch.Tensor) else float(global_scale)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ValueError(f"bad NVFP4 global scale {global_scale!r}") from exc
    if not gs > 0.0:
        raise ValueError(f"non-positive NVFP4 global scale {gs}")
    return (vals * scale * gs).contiguous()
