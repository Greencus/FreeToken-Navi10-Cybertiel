"""GPU smoke tests for the ported Vulkan compute runtime.

Signatures mirror the fork's pybind block (freetoken-vulkan/python/
ext_module.cpp), python/test_resident_bf16_linear.py and python/bench_experts.py.

Every test SKIPS cleanly when no Vulkan device or extension is available —
CI machines without RADV/loader must not fail on this file.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

torch = pytest.importorskip("torch")

import freetoken.kernel.vulkan as vk  # noqa: E402


def _skip_unless_vulkan() -> None:
    # available() never raises; it probes libvulkan via ctypes and tries to
    # load (or JIT-build) the extension.
    if not vk.available():
        pytest.skip("no usable Vulkan device/extension on this machine")


@pytest.fixture(scope="module")
def ext():
    _skip_unless_vulkan()
    return vk.require_ext()


def test_device_info_reports_primary_gpu():
    _skip_unless_vulkan()
    info = vk.device_info()
    assert info["name"], info
    assert info["driver"], info
    assert info["api_version"], info
    assert info["vram_bytes"] and info["vram_bytes"] > 0, info
    assert info["shaders_dir"], info
    # Count the .comp sources instead of hardcoding a total, so adding a
    # shader cannot silently skip compiling it (45 at port time, 52 with the
    # K-quant unpackers).
    src = pathlib.Path(vk.__file__).parent / "shaders"
    n_comp = len(list(src.glob("*.comp")))
    assert n_comp >= 45, n_comp
    assert info["shaders_compiled"] == n_comp, info


def test_public_surface():
    _skip_unless_vulkan()
    for name in (
        "available",
        "load_ext",
        "require_ext",
        "shaders_dir",
        "compile_shaders",
        "device_info",
    ):
        assert callable(getattr(vk, name)), name
    # Raw module reachable under both stable names.
    assert vk.ext is vk.ext_module
    assert vk.ext is vk.require_ext()
    assert (
        importlib.import_module("freetoken.kernel.vulkan.ext")
        is vk.ext
    )
    assert (
        importlib.import_module("freetoken.kernel.vulkan.ext_module")
        is vk.ext
    )
    # The fork's full op set (72 functions at port time) plus the K-quant
    # GEMV bindings.
    ops = [n for n in dir(vk.ext) if not n.startswith("_")]
    assert len(ops) >= 72, sorted(ops)
    for name in (
        "linear_kquant_resident_io",
        "linear_q4k_resident_io",
        "linear_q5k_resident_io",
        "linear_q8_0_resident_io",
        "linear_q4k_gateup_silu_io",
        "linear_q5k_gateup_silu_io",
        "linear_q4k_acc_resident_io",
        "linear_q5k_acc_resident_io",
        "batch_kquant_io",
        "batch_moe_kquant_record",
    ):
        assert name in ops, name


def test_rmsnorm(ext):
    torch.manual_seed(42)
    x = torch.randn(4, 128)
    w = torch.rand(128) + 0.5
    gpu = ext.rmsnorm(x, w, 1e-6)
    ref = torch.nn.functional.rms_norm(x, (128,), w, eps=1e-6)
    diff = (gpu - ref).abs().max().item()
    assert diff <= 1e-5, diff  # known-good: ~1.2e-07 - 3.6e-07


def test_linear_bf16_resident_io(ext):
    # Signature per fork python/test_resident_bf16_linear.py.
    torch.manual_seed(7)
    T, N, K = 1, 512, 256
    x = torch.randn(T, K) * 0.1
    w = (torch.randn(N, K) / K**0.5).to(torch.bfloat16)
    ref = x.float() @ w.float().T

    h_x = ext.upload_resident(x.contiguous().float())
    h_w = ext.upload_resident(w)  # raw bytes; kernel reads bf16
    h_y = ext.allocate_resident(T * N * 4)  # fp32 out
    try:
        ext.linear_bf16_resident_io(h_x, h_w, 0, h_y, T, N, K, False)
        out = ext.download_resident(h_y, [T, N])
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
    diff = (out - ref).abs().max().item()
    assert diff < 5e-3, diff


def test_batch_bf16_round_trip(ext):
    # batch_begin / batch_upload / batch_bf16_io / batch_download / batch_submit
    # per fork python/bench_experts.py.
    torch.manual_seed(11)
    T, N, K = 1, 256, 128
    x = (torch.randn(T, K) * 0.1).contiguous().float()
    w = (torch.randn(N, K) / K**0.5).to(torch.bfloat16)
    ref = x.float() @ w.float().T

    h_x = ext.upload_resident(x)
    h_w = ext.upload_resident(w)
    h_y = ext.allocate_resident(T * N * 4)
    try:
        out = torch.empty(T, N)
        ext.batch_begin()
        ext.batch_upload(h_x, x)
        ext.batch_bf16_io(h_x, h_w, 0, h_y, T, N, K, False)
        ext.batch_download(h_y, out)
        ext.batch_submit()
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
    diff = (out - ref).abs().max().item()
    assert diff < 5e-3, diff


def test_desc_cache_stats_and_resident_cleanup(ext):
    before = ext.desc_cache_stats()
    assert len(before) == 6
    assert all(isinstance(v, int) for v in before)

    total = ext.resident_bytes_total()
    h = ext.upload_resident(torch.zeros(1024))
    assert ext.resident_bytes_total() == total + 1024 * 4
    ext.free_resident(h)
    assert ext.resident_bytes_total() == total
    # free_resident is a no-op for unknown handles (fork contract).
    ext.free_resident(2**62)

    after = ext.desc_cache_stats()
    assert len(after) == 6
    assert after[3] >= before[3] and after[4] >= before[4]  # hits/misses monotone
