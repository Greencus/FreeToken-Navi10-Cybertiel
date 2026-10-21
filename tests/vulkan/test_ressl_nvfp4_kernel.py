"""Port of the fork's test_ressl_nvfp4_kernel.py: the native NVFP4 resident kernel.

Two cases:

- ``test_nvfp4_kernel_synthetic`` (always runs): fabricates U8-packed E2M1
  weights + e4m3 block scales, runs ``linear_nvfp4_resident_io`` and checks
  it against the CPU dequant reference (``dequantize_nvfp4``). Gates the
  shader <-> host-dequant agreement with only the ext + small tensors.
- ``test_nvfp4_kernel_real_weights`` (opt-in, FREETOKEN_TEST_MODEL): the
  fork's original case on real Ressl expert weights, gate [512,2048] and
  down [2048,512] from layer 2 expert 0, tolerance rtol=2e-3 atol=2e-3.

Scales upload as U32 words (4 e4m3 scales/word, low byte first) through the
package's ``pack_scales_u32``; bytes are reinterpreted, never value-converted.
"""
from __future__ import annotations

import torch

from freetoken.models.qwen3_5_moe.vulkan import (
    ShardedSafetensors,
    dequantize_nvfp4,
    pack_scales_u32,
)

# Fork tolerances for the real-weights case (float8 scale rounding).
REAL_RTOL, REAL_ATOL = 2e-3, 2e-3


def _run_nvfp4_linear(ext, packed, scales_u32, x, rows, cols, gscale):
    h_x = ext.upload_resident(x.contiguous())
    h_packed = ext.upload_resident(packed.contiguous())
    h_scales = ext.upload_resident(scales_u32.contiguous())
    h_y = ext.allocate_resident(rows * 4)
    try:
        ext.linear_nvfp4_resident_io(
            h_x, h_packed, h_scales, h_y, 1, rows, cols, float(gscale)
        )
        return ext.download_resident(h_y, [1, rows])
    finally:
        for handle in (h_x, h_packed, h_scales, h_y):
            ext.free_resident(handle)


def test_nvfp4_kernel_synthetic(vulkan_ext):
    ext = vulkan_ext
    torch.manual_seed(171)
    rows, cols = 128, 256  # cols divisible by 16 (NVFP4 block size)
    packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
    # e4m3 block scales in [0.01, 0.2]; fp32 global scale.
    scales = (torch.rand(rows, cols // 16) * 0.19 + 0.01).to(torch.float8_e4m3fn)
    gscale = 0.6
    ref_w = dequantize_nvfp4(packed, scales, gscale)
    x = torch.randn(1, cols) * 0.05
    reference = x @ ref_w.T

    actual = _run_nvfp4_linear(ext, packed, pack_scales_u32(scales), x, rows, cols, gscale)
    diff = (actual - reference).abs()
    maxdiff = diff.max().item()
    assert torch.isfinite(actual).all(), "NVFP4 kernel output non-finite"
    # The shader decodes the same E2M1 codebook + scales as the host path;
    # fp32 accumulation error is ~1e-6 relative, keep a generous bound.
    assert torch.allclose(actual, reference, rtol=1e-3, atol=1e-3), (
        f"synthetic NVFP4 maxdiff {maxdiff:.3e}"
    )


def test_nvfp4_kernel_real_weights(vulkan_ext, nvfp4_model_dir):
    ext = vulkan_ext
    tensors = ShardedSafetensors(nvfp4_model_dir)
    for layer, expert, suffix, seed in (
        (2, 0, "gate_proj", 171),
        (2, 0, "down_proj", 172),
    ):
        stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{suffix}"
        packed = tensors.get(stem + ".weight").contiguous()
        scales = tensors.get(stem + ".weight_scale").contiguous()
        gs = float(tensors.get(stem + ".weight_scale_2").reshape(-1)[0].item())
        rows, half = packed.shape[-2], packed.shape[-1]
        cols = half * 2
        ref = dequantize_nvfp4(packed, scales, gs)
        torch.manual_seed(seed)
        x = torch.randn(1, cols) * 0.02
        reference = x @ ref.T

        actual = _run_nvfp4_linear(ext, packed, pack_scales_u32(scales), x, rows, cols, gs)
        diff = (actual - reference).abs()
        line = (
            f"L{layer}E{expert}.{suffix} shape=[{rows},{cols}] "
            f"max={diff.max().item():.9g} mean={diff.mean().item():.9g}"
        )
        print(line, flush=True)
        assert torch.isfinite(actual).all(), line
        assert torch.allclose(actual, reference, rtol=REAL_RTOL, atol=REAL_ATOL), line
