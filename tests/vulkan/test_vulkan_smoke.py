"""GPU smoke test: runtime resolution + core ops on a real Vulkan device.

Gates:
1. _runtime resolves the extension (preferred freetoken.kernel.vulkan path,
   staged FREETOKEN_VULKAN_EXT_DIR fallback) and the .spv directory exists.
2. ext.rmsnorm matches torch's rms_norm (max abs diff <= 1e-5).
3. One resident BF16 linear via linear_bf16_resident_io, call signature
   exactly as the fork's test_resident_bf16_linear.py:
   (x_handle, w_handle, b_handle, y_handle, T, N, K, use_bias).
4. A batched upload/download round trip through batch_begin/batch_upload/
   batch_download/batch_submit.
5. free_resident returns the resident pool to its baseline.

Skips cleanly when no Vulkan device/extension is present.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.models.qwen3_5_moe.vulkan import get_ext, shaders_dir
from freetoken.models.qwen3_5_moe.vulkan._runtime import ext_or_none, runtime_info


def test_runtime_resolves():
    ext = ext_or_none()
    if ext is None:
        import pytest

        pytest.skip(f"no Vulkan extension/device available ({runtime_info()})")
    assert get_ext() is ext  # cached after first resolution
    spv = sorted(shaders_dir().glob("*.spv"))
    if not spv:
        import pytest

        pytest.skip(f"no compiled shaders in {shaders_dir()}")
    assert len(spv) >= 40, f"expected ~45 .spv kernels, found {len(spv)}"


def test_rmsnorm_matches_torch(vulkan_ext):
    ext = vulkan_ext
    torch.manual_seed(42)
    x = torch.randn(4, 128)
    w = torch.rand(128) + 0.5
    gpu = ext.rmsnorm(x, w, 1e-6)
    ref = F.rms_norm(x, (128,), w, eps=1e-6)
    maxdiff = (gpu - ref).abs().max().item()
    assert maxdiff <= 1e-5, f"rmsnorm maxdiff {maxdiff}"


def test_resident_linear_bf16(vulkan_ext):
    """Port of the fork's test_resident_bf16_linear.py on synthetic weights
    (the fork used a real Qwen3-4B q_proj; shapes/semantics are identical)."""
    ext = vulkan_ext
    torch.manual_seed(121)
    T, N, K = 1, 512, 256
    x = torch.randn(T, K) * 0.02
    w_bf16 = (torch.randn(N, K) * (1.0 / K**0.5)).to(torch.bfloat16)
    reference = x @ w_bf16.float().T

    h_x = ext.upload_resident(x.contiguous())
    h_w = ext.upload_resident(w_bf16.contiguous())
    h_y = ext.allocate_resident(T * N * 4)
    try:
        # Exact fork call shape: (x, w, bias=0, y, T, N, K, use_bias=False)
        ext.linear_bf16_resident_io(h_x, h_w, 0, h_y, T, N, K, False)
        actual = ext.download_resident(h_y, [T, N])
    finally:
        ext.free_resident(h_x)
        ext.free_resident(h_w)
        ext.free_resident(h_y)
    diff = (actual - reference).abs()
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, reference, rtol=1e-4, atol=5e-5), (
        f"bf16 linear maxdiff {diff.max().item()}"
    )


def test_batch_round_trip(vulkan_ext):
    """batch_upload -> batch_download -> batch_submit must move data intact."""
    ext = vulkan_ext
    torch.manual_seed(7)
    x = torch.randn(64, 32)
    h = ext.allocate_resident(x.numel() * 4)
    out = torch.empty_like(x)
    try:
        ext.batch_begin()
        ext.batch_upload(h, x.contiguous())
        ext.batch_download(h, out)
        ext.batch_submit()
    finally:
        ext.free_resident(h)
    # fp32 device->host copy of a freshly uploaded buffer is bit-exact.
    assert torch.equal(out, x), f"maxdiff {(out - x).abs().max().item()}"


def test_free_resident_restores_pool(vulkan_ext):
    ext = vulkan_ext
    baseline = ext.resident_bytes_total()
    h = ext.allocate_resident(1 << 20)
    assert ext.resident_bytes_total() >= baseline + (1 << 20)
    ext.free_resident(h)
    assert ext.resident_bytes_total() == baseline, (
        f"pool {ext.resident_bytes_total()} != baseline {baseline}"
    )
