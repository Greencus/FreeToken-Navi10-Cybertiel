"""K-quant GEMV shader gates: in-shader unpack vs the numpy golden dequant.

The CyberTiel GGUF stores the MoE experts as Q5_K (blk.0-14) and Q4_K
(blk.15-40) with Q8_0 dense tensors. The Vulkan port used to dequantize every
routed expert on the host to BF16 (seconds per token, GB/s of upload); the
shaders in python/freetoken/kernel/vulkan/shaders/linear_q{k,8_0}_*_f32.comp
unpack the GGUF bytes in-shader instead, so the weights stay packed in VRAM
and on the wire. These tests pin that contract:

    packed bytes -> shader output
    vs packed bytes -> numpy dequant -> torch matmul

That numpy dequant (freetoken.third_party.gguf.quants) is the golden path
llama.cpp's own C dequantizers match bit-for-bit.

Every GPU test SKIPS when no Vulkan device/extension is available.
Set FREETOKEN_TEST_MODEL_GGUF to add the real-checkpoint expert test.

Contract exercised here (see docs/vulkan/kquant-kernels.md):
  gateup  slot0 = first GATE row of the bundle in the packed arena; the up
          rows follow at slot0 + N. Output is Y[T, N].
  acc     slot0 = first row of the down matrix in the packed arena; every
          dispatch does Y[T, N] += ..., so sequential expert calls sum into
          one accumulator (cleared once per layer with batch_fill).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import freetoken.kernel.vulkan as vk  # noqa: E402
from freetoken.layers.quantization.ggml_types import row_bytes  # noqa: E402
from freetoken.third_party.gguf import GGMLQuantizationType  # noqa: E402
from freetoken.third_party.gguf.quants import dequantize as np_dequant  # noqa: E402

QENUM = {
    "q4_k": GGMLQuantizationType.Q4_K,
    "q5_k": GGMLQuantizationType.Q5_K,
    "q8_0": GGMLQuantizationType.Q8_0,
}
RESIDENT_OP = {
    "q4_k": "linear_q4k_resident_io",
    "q5_k": "linear_q5k_resident_io",
    "q8_0": "linear_q8_0_resident_io",
}
GATEUP_OP = {"q4_k": "linear_q4k_gateup_silu_io", "q5_k": "linear_q5k_gateup_silu_io"}

GGUF = os.environ.get("FREETOKEN_TEST_MODEL_GGUF")
requires_gguf = pytest.mark.skipif(
    not GGUF or not os.path.isfile(GGUF),
    reason="FREETOKEN_TEST_MODEL_GGUF not set",
)


def _skip_unless_vulkan() -> None:
    if not vk.available():
        pytest.skip("no usable Vulkan device/extension on this machine")


@pytest.fixture(scope="module")
def ext():
    _skip_unless_vulkan()
    return vk.require_ext()


def random_packed(qtype: str, rows: int, K: int, seed: int = 1234) -> np.ndarray:
    """Random-but-valid packed K-quant bytes, shape (rows, row_bytes).

    Random bits everywhere except d/dmin: arbitrary bits decode to inf/NaN
    and every comparison would be meaningless. The quants/scales themselves
    are uniform noise, which exercises the full unpack path (all 6-bit scale
    positions, both nibbles, the qh bit matrix).
    """
    rng = np.random.default_rng(seed)
    if qtype in ("q4_k", "q5_k"):
        ty = 144 if qtype == "q4_k" else 176
        blocks = K // 256
    else:
        ty = 34
        blocks = K // 32
    rb = row_bytes(int(QENUM[qtype]), K)
    out = np.empty((rows, rb), np.uint8)
    for r in range(rows):
        for b in range(blocks):
            base = b * ty
            out[r, base:base + 2] = rng.uniform(1e-3, 1e-1, 1).astype(
                np.float16).view(np.uint8)  # d
            off = base + 2
            if qtype != "q8_0":
                out[r, off:off + 2] = rng.uniform(1e-4, 1e-2, 1).astype(
                    np.float16).view(np.uint8)  # dmin
                off += 2
            out[r, off:] = rng.integers(0, 256, rb - off, np.uint8)
    return out


def _ref_gemv(packed: np.ndarray, qtype: str, K: int, x: np.ndarray) -> np.ndarray:
    """(1, rows) fp32 reference: golden dequant, then a torch-grade matmul."""
    w = np_dequant(packed, QENUM[qtype]).reshape(packed.shape[0], K)
    return x.astype(np.float32) @ w.astype(np.float32).T


def _assert_close(out: np.ndarray, ref: np.ndarray, *, label: str) -> None:
    """Bound scaled to the reference magnitude, not a fixed absolute epsilon.

    fp32 accumulation over K packed products carries ~K*eps of rounding, so
    the bound is relative to max|ref|. Measured on this 5700 XT: worst
    synthetic case 2.5e-4 abs at K=2048 (Q5_K), 1.2e-4 (Q4_K), 2.3e-5
    (Q8_0); the real CyberTiel expert lands at 1.9e-8 abs. 5e-4 * max|ref|
    plus a 1e-4 floor keeps ~2x headroom over the worst synthetic case while
    still failing any wrong unpack (wrong scale index alone shifts the output
    by percent-level amounts).
    """
    scale = float(np.abs(ref).max()) + 1e-9
    diff = float(np.abs(out - ref).max())
    tol = 5e-4 * scale + 1e-4
    assert out.shape == ref.shape, (label, out.shape, ref.shape)
    assert np.isfinite(out).all(), f"{label}: non-finite shader output"
    assert diff <= tol, (
        f"{label}: max abs diff {diff:.3e} > tol {tol:.3e} (max|ref| {scale:.3e})")


@pytest.mark.parametrize("qtype", ["q4_k", "q5_k", "q8_0"])
@pytest.mark.parametrize("rows,K", [(64, 256), (33, 512), (512, 2048)])
def test_resident_gemv_matches_golden_dequant(ext, qtype, rows, K):
    packed = random_packed(qtype, rows, K)
    x = (np.random.default_rng(99).standard_normal((1, K)) * 0.1).astype(np.float32)
    ref = _ref_gemv(packed, qtype, K, x)

    h_x = ext.upload_resident(torch.from_numpy(np.ascontiguousarray(x)))
    h_w = ext.upload_resident(torch.from_numpy(packed.copy()))
    h_y = ext.allocate_resident(rows * 4)
    try:
        getattr(ext, RESIDENT_OP[qtype])(h_x, h_w, h_y, 1, rows, K, 1.0, 0)
        out = ext.download_resident(h_y, [1, rows]).numpy()
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
    _assert_close(out, ref, label=f"{qtype} rows={rows} K={K}")


@pytest.mark.parametrize("qtype", ["q4_k", "q5_k", "q8_0"])
@pytest.mark.parametrize("rows,K", [(64, 256), (512, 2048)])
def test_resident_gemv_nonzero_slot0(ext, qtype, rows, K):
    """slot0 offsets the packed W rows (arena layout), never the output."""
    prefix = 5  # dummy rows parked before this expert's block
    packed = np.concatenate(
        [random_packed(qtype, prefix, K, seed=41), random_packed(qtype, rows, K, seed=42)])
    inner = packed[prefix:prefix + rows]

    x = (np.random.default_rng(43).standard_normal((1, K)) * 0.1).astype(np.float32)
    ref = _ref_gemv(inner, qtype, K, x)

    h_x = ext.upload_resident(torch.from_numpy(np.ascontiguousarray(x)))
    h_w = ext.upload_resident(torch.from_numpy(packed.copy()))
    h_y = ext.allocate_resident(rows * 4)
    try:
        getattr(ext, RESIDENT_OP[qtype])(h_x, h_w, h_y, 1, rows, K, 1.0, prefix)
        out = ext.download_resident(h_y, [1, rows]).numpy()
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
    _assert_close(out, ref, label=f"{qtype} slot0={prefix} rows={rows} K={K}")


@pytest.mark.parametrize("qtype", ["q4_k", "q5_k"])
@pytest.mark.parametrize("slot0", [0, 3])
def test_gateup_silu_matches_golden_dequant(ext, qtype, slot0):
    """Fused silu(gate) * up over a stacked gate+up packed bundle.

    The bundle layout is the one packed_gate_up_expert returns: rows
    slot0..slot0+N-1 are gate, slot0+N..slot0+2N-1 are up.
    """
    N, K = 256, 512
    gate = random_packed(qtype, N, K, seed=7)
    up = random_packed(qtype, N, K, seed=8)
    bundle = np.concatenate([gate, up], axis=0)
    packed = np.concatenate([random_packed(qtype, slot0, K, seed=9), bundle])

    x = (np.random.default_rng(11).standard_normal((1, K)) * 0.1).astype(np.float32)
    g = _ref_gemv(gate, qtype, K, x)
    u = _ref_gemv(up, qtype, K, x)
    ref = (g / (1.0 + np.exp(-g))) * u

    h_x = ext.upload_resident(torch.from_numpy(np.ascontiguousarray(x)))
    h_w = ext.upload_resident(torch.from_numpy(packed.copy()))
    h_y = ext.allocate_resident(N * 4)
    try:
        getattr(ext, GATEUP_OP[qtype])(h_x, h_w, h_y, 1, N, K, 1.0, slot0)
        out = ext.download_resident(h_y, [1, N]).numpy()
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
    _assert_close(out, ref, label=f"{qtype} gateup_silu N={N} K={K} slot0={slot0}")


@pytest.mark.parametrize("qtype", ["q4_k", "q5_k"])
def test_batch_acc_sums_experts(ext, qtype):
    """The batched down path: 3 experts, one accumulator, one batch.

    This is the MoE contract batch_moe_kquant_record drives: every dispatch
    adds its expert into the SAME output rows (Y +=), with slot0 selecting
    which packed down matrix to read. The engine folds each expert's router
    scale into gscale the same way.
    """
    rows, K = 512, 512
    experts = 3
    packed = np.concatenate([random_packed(qtype, rows, K, seed=21 + e)
                             for e in range(experts)])
    scales = [1.0, 0.37, -1.25]
    rng = np.random.default_rng(22)
    x = (rng.standard_normal((1, K)) * 0.1).astype(np.float32)
    ref = np.zeros((1, rows), np.float32)
    for e in range(experts):
        ref += scales[e] * _ref_gemv(packed[e * rows:(e + 1) * rows], qtype, K, x)

    h_x = ext.upload_resident(torch.from_numpy(np.ascontiguousarray(x)))
    h_w = ext.upload_resident(torch.from_numpy(packed.copy()))
    h_y = ext.allocate_resident(rows * 4)
    try:
        ext.batch_begin()
        ext.batch_fill(h_y, 0.0, rows)
        for e in range(experts):
            ext.batch_kquant_io(qtype, "acc", h_w, h_x, h_y, 1, rows, K,
                                scales[e], e * rows)
        out = torch.empty(1, rows)
        ext.batch_download(h_y, out)
        ext.batch_submit()
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
    _assert_close(out.numpy(), ref, label=f"{qtype} batch acc x{experts} K={K}")


@pytest.mark.parametrize("qtype", ["q4_k", "q5_k"])
def test_batch_moe_record_matches_batch_kquant_io(ext, qtype):
    """batch_moe_kquant_record must reproduce the explicit gateup+acc calls.

    N must be a multiple of 256: it is the K-quant block granularity of both
    the gateup bundle rows and the down matrix this test feeds them into.
    """
    N, K = 256, 512
    dn_rows = 512
    experts = 4
    # one 2N-row bundle per expert: N gate rows then N up rows
    gu_bundle = np.concatenate([
        np.concatenate([random_packed(qtype, N, K, seed=31 + 2 * e),
                        random_packed(qtype, N, K, seed=32 + 2 * e)])
        for e in range(experts)])
    # the down matrix consumes the N-wide silu vector: its K is N, not the
    # gateup K
    dn_pack = np.concatenate([random_packed(qtype, dn_rows, N, seed=41 + e)
                              for e in range(experts)])
    scales = [1.0, 0.5, 0.25, -0.75]

    rng = np.random.default_rng(44)
    x = (rng.standard_normal((1, K)) * 0.1).astype(np.float32)

    # references: fused gateup-silu per expert, then the scaled sum of downs
    silu_refs, acc_ref = [], np.zeros((1, dn_rows), np.float32)
    for e in range(experts):
        sl = e * 2 * N  # gu_slot0: expert e's bundle starts at row 2*N*e
        g = _ref_gemv(gu_bundle[sl:sl + N], qtype, K, x)
        u = _ref_gemv(gu_bundle[sl + N:sl + 2 * N], qtype, K, x)
        s = (g / (1.0 + np.exp(-g))) * u
        silu_refs.append(s)
        acc_ref += scales[e] * _ref_gemv(dn_pack[e * dn_rows:(e + 1) * dn_rows],
                                         qtype, N, s.astype(np.float32))

    h_x = ext.upload_resident(torch.from_numpy(np.ascontiguousarray(x)))
    h_gu = ext.upload_resident(torch.from_numpy(gu_bundle.copy()))
    h_dn = ext.upload_resident(torch.from_numpy(dn_pack.copy()))
    h_acc = ext.allocate_resident(dn_rows * 4)
    # one buffer per expert for the fused silu slots, as the engine holds them
    h_slots = [ext.allocate_resident(N * 4) for _ in range(experts)]
    try:
        acc_t = torch.empty(1, dn_rows)
        slot_t = [torch.empty(1, N) for _ in range(experts)]
        ext.batch_begin()
        ext.batch_fill(h_acc, 0.0, dn_rows)
        ext.batch_moe_kquant_record(
            h_x, h_acc, qtype,
            h_slots,
            [h_gu] * experts, [1.0] * experts, N, K,
            [h_dn] * experts, list(scales), dn_rows, N,
            [e * 2 * N for e in range(experts)],
            [e * dn_rows for e in range(experts)],
        )
        for h, t in zip(h_slots, slot_t):
            ext.batch_download(h, t)
        ext.batch_download(h_acc, acc_t)
        ext.batch_submit()
    finally:
        for h in [h_x, h_gu, h_dn, h_acc] + h_slots:
            ext.free_resident(h)

    for e in range(experts):
        _assert_close(slot_t[e].numpy(), silu_refs[e].astype(np.float32),
                      label=f"{qtype} fused gateup expert {e}")
    _assert_close(acc_t.numpy(), acc_ref, label=f"{qtype} fused acc x{experts}")


@requires_gguf
def test_real_cybertiel_expert(ext):
    """One real Q5_K expert out of the 22.8 GB file, mmap'd, never dequant-copied."""
    from freetoken.models.gguf.packed import packed_expert, packed_expert_info
    from freetoken.models.gguf.reader import gguf_tensor_index

    index = gguf_tensor_index(GGUF)
    name = "blk.0.ffn_gate_exps.weight"
    meta = packed_expert_info(index, name, 7)
    if meta.qtype != "q5_k":
        pytest.skip(f"{name} is {meta.qtype} here, expected the Q5_K bank")
    packed = packed_expert(index, name, 7)

    x = (np.random.default_rng(3).standard_normal((1, meta.cols))
         * 0.1).astype(np.float32)
    ref = _ref_gemv(packed, meta.qtype, meta.cols, x)

    h_x = ext.upload_resident(torch.from_numpy(np.ascontiguousarray(x)))
    h_w = ext.upload_resident(torch.from_numpy(packed.copy()))
    h_y = ext.allocate_resident(meta.rows * 4)
    try:
        getattr(ext, RESIDENT_OP[meta.qtype])(
            h_x, h_w, h_y, 1, meta.rows, meta.cols, 1.0, 0)
        out = ext.download_resident(h_y, [1, meta.rows]).numpy()
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
    # measured 1.9e-8 abs / 3.2e-7 relative on this file (real x, real scales)
    scale = float(np.abs(ref).max())
    diff = float(np.abs(out - ref).max())
    assert diff <= 1e-5 + 1e-5 * scale, (meta.as_dict(), diff, scale)
