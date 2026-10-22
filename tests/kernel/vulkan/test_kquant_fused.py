"""Fused 8-expert K-quant MoE gates: fused gateup8+acc8 vs the per-expert sequence.

The fused dispatches
(`shaders/linear_q{4,5}k_{gateup8,acc8}_f32.comp`, bindings
`linear_kquant_{gateup8,acc8}_io` / `batch_kquant_{gateup8,acc8}_io` /
`batch_moe_kquant_fused_record`) replace a MoE layer's 8 gateup-silu +
8 down-accumulate per-expert dispatches with 2, over a row-contiguous
packed arena.  The bar this file enforces is NOT "close enough": the fused
pair must be BIT-IDENTICAL (same bytes) to the sequence it replaces -
same lane/word split, same super-block indexing, same FMA chain, same
reduction tree, same canonical-order accumulation.

Arena contract (mirrors the fork's fixed-slot expert arena):
  gateup arena slot j = rows [slot*rows_per_slot, slot*rows_per_slot+2N)
      gate rows 0..N-1 then up rows N..2N-1 (one packed bundle per slot);
  down arena   slot j = rows [slot*rows_per_slot, +dn_rows);
  Ids[j]  = arena slot of canonical expert j (u32[8]);
  Gs[j]   = gateup gscale, Gs[8+j] = acc gscale (fp32[16]) - the same
            values the per-expert push constant carried;
  slot 0  = the engine's 0x7FFFFFFF no-content sentinel.  The shaders
            address it like any slot (Id 0x7FFFFFFF would land outside the
            buffer; the engine never routes an expert there), and the
            tests here fill unused slots with the 0x7FFFFFFF word pattern,
            whose fp16 halves are NaN - any accidental read of a sentinel
            slot poisons the output and fails the bit gate.

Every test SKIPS cleanly when no Vulkan device is present.  Packed slabs
are random-but-valid synthetic bytes (d/dmin in a finite range, noise
everywhere else), the same generator test_kquant_shaders.py uses; a
numpy-golden cross-check (freetoken.third_party.gguf.quants, bit-exact
against llama.cpp's C dequantizers) keeps them honest.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import freetoken.kernel.vulkan as vk  # noqa: E402
from freetoken.layers.quantization.ggml_types import row_bytes  # noqa: E402
from freetoken.third_party.gguf import GGMLQuantizationType  # noqa: E402
from freetoken.third_party.gguf.quants import dequantize as np_dequant  # noqa: E402

QENUM = {"q4_k": GGMLQuantizationType.Q4_K, "q5_k": GGMLQuantizationType.Q5_K}
GATEUP_OP = {"q4_k": "linear_q4k_gateup_silu_io", "q5_k": "linear_q5k_gateup_silu_io"}
ACC_OP = {"q4_k": "linear_q4k_acc_resident_io", "q5_k": "linear_q5k_acc_resident_io"}

# Layer shape for the correctness gates (N a multiple of 256: K-quant block
# granularity of the down K = silu vector; small enough to keep the GPU
# window short on this crash-prone box, real enough to span super-blocks).
N = 256            # gate rows per expert == down K == silu length
K_GU = 512         # gateup K (x width)
DN_ROWS = 512      # down rows
EXPERTS = 8
# Non-trivial, non-uniform scales on both sides: catches per-expert scale
# plumbing bugs a 1.0-only test would miss (engine uses 1.0 for gateup and
# the renormalized router value for acc).
GU_SCALES = (1.0, 0.75, 1.25, 0.5, 1.5, 0.25, 2.0, 0.125)
DN_SCALES = (1.0, 0.37, -1.25, 0.5, 0.25, -0.75, 0.125, 1.75)

# u32 0x7FFFFFFF little-endian - the engine's _ARENA_NO_SLOT word.  Its
# fp16 halves (0x7FFF) are NaN, so a sentinel slot read cannot pass silently.
_POISON = np.array([0xFF, 0xFF, 0xFF, 0x7F], np.uint8)


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
    and every comparison would be meaningless (same generator as
    test_kquant_shaders.py).
    """
    rng = np.random.default_rng(seed)
    ty = 144 if qtype == "q4_k" else 176
    blocks = K // 256
    rb = row_bytes(int(QENUM[qtype]), K)
    out = np.empty((rows, rb), np.uint8)
    for r in range(rows):
        for b in range(blocks):
            base = b * ty
            out[r, base:base + 2] = rng.uniform(1e-3, 1e-1, 1).astype(
                np.float16).view(np.uint8)  # d
            out[r, base + 2:base + 4] = rng.uniform(1e-4, 1e-2, 1).astype(
                np.float16).view(np.uint8)  # dmin
            out[r, base + 4:] = rng.integers(0, 256, rb - base - 4, np.uint8)
    return out


def _ref_gemv(packed: np.ndarray, qtype: str, K: int, x: np.ndarray) -> np.ndarray:
    """(1, rows) fp32: golden dequant, then a torch-grade matmul."""
    w = np_dequant(packed, QENUM[qtype]).reshape(packed.shape[0], K)
    return x.astype(np.float32) @ w.astype(np.float32).T


def _assert_bit_identical(a: np.ndarray, b: np.ndarray, label: str) -> None:
    """Strict: the raw bytes of both float32 results must match exactly."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert a.dtype == np.float32 and b.dtype == np.float32, (label, a.dtype, b.dtype)
    assert np.isfinite(a).all(), f"{label}: non-finite result in A"
    assert np.isfinite(b).all(), f"{label}: non-finite result in B"
    if a.tobytes() != b.tobytes():
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        bad = np.argwhere(a.view(np.uint32) != b.view(np.uint32))
        first = bad[0].tolist() if len(bad) else None
        raise AssertionError(
            f"{label}: NOT bit-identical - {len(bad)} differing elements, "
            f"max abs {diff.max():.3e} (rel {(diff.max() / (np.abs(b).max() + 1e-30)):.3e}), "
            f"first at {first}: {a.ravel()[first[0]] if first else '?'} vs "
            f"{b.ravel()[first[0]] if first else '?'}")


def _assert_close(a: np.ndarray, b: np.ndarray, label: str) -> None:
    """Loose numpy-golden cross-check (unpack sanity, not the fusion gate)."""
    scale = float(np.abs(b).max()) + 1e-9
    diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    tol = 5e-4 * scale + 1e-4
    assert np.isfinite(a).all(), f"{label}: non-finite"
    assert diff <= tol, (label, diff, tol, scale)


def _poison(rows: int, rowb: int) -> np.ndarray:
    assert rowb % 4 == 0, rowb
    return np.tile(_POISON, (rows, rowb // 4))


def _scenario(qtype: str):
    """8 gateup bundles (2N rows each) + 8 down matrices + one x row."""
    gu = [np.concatenate([random_packed(qtype, N, K_GU, seed=1000 + e),
                          random_packed(qtype, N, K_GU, seed=1100 + e)])
          for e in range(EXPERTS)]
    dn = [random_packed(qtype, DN_ROWS, N, seed=1200 + e)
          for e in range(EXPERTS)]
    x = (np.random.default_rng(7).standard_normal((1, K_GU)) * 0.1).astype(
        np.float32)
    return gu, dn, x


def _upload(ext, arr: np.ndarray) -> int:
    return ext.upload_resident(torch.from_numpy(np.ascontiguousarray(arr)))


def _ids(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.uint32)


def _gs() -> torch.Tensor:
    # The fp32 values the per-expert push constant carried ((float)gscale
    # in C++, IEEE single rounding - same bits numpy gets here).
    return torch.tensor(list(GU_SCALES) + list(DN_SCALES), dtype=torch.float32)


def _numpy_refs(qtype: str, gu, dn, x):
    """Golden per-expert silu rows + sequential-order acc reference."""
    silu_refs, acc_ref = [], np.zeros((1, DN_ROWS), np.float32)
    for e in range(EXPERTS):
        g = GU_SCALES[e] * _ref_gemv(gu[e][:N], qtype, K_GU, x)
        u = GU_SCALES[e] * _ref_gemv(gu[e][N:], qtype, K_GU, x)
        s = ((g / (1.0 + np.exp(-g))) * u).astype(np.float32)
        silu_refs.append(s)
        acc_ref += np.float32(DN_SCALES[e]) * _ref_gemv(
            dn[e], qtype, N, s.astype(np.float32))
    return silu_refs, acc_ref


# ---------------------------------------------------------------------------
# The reference arm: the exact per-expert sequence the fusion replaces
# (8x gateup_silu_io interleaved with 8x acc_resident_io, as
# batch_moe_kquant_record records it; acc starts at exact +0.0).
# ---------------------------------------------------------------------------
def _run_per_expert(ext, qtype, gu, dn, x, *, batched: bool):
    h_x = _upload(ext, x)
    h_gu = [_upload(ext, b) for b in gu]
    h_dn = [_upload(ext, b) for b in dn]
    h_silu = [ext.allocate_resident(N * 4) for _ in range(EXPERTS)]
    h_acc = ext.upload_resident(torch.zeros(1, DN_ROWS, dtype=torch.float32))
    try:
        if batched:
            ext.batch_begin()
            ext.batch_fill(h_acc, 0.0, DN_ROWS)
            for e in range(EXPERTS):
                ext.batch_kquant_io(qtype, "gateup", h_gu[e], h_x, h_silu[e],
                                    1, N, K_GU, GU_SCALES[e], 0)
                ext.batch_kquant_io(qtype, "acc", h_dn[e], h_silu[e], h_acc,
                                    1, DN_ROWS, N, DN_SCALES[e], 0)
            silu = [torch.empty(1, N) for _ in range(EXPERTS)]
            acc_t = torch.empty(1, DN_ROWS)
            for h, t in zip(h_silu, silu):
                ext.batch_download(h, t)
            ext.batch_download(h_acc, acc_t)
            ext.batch_submit()
            return [t.numpy() for t in silu], acc_t.numpy()
        for e in range(EXPERTS):
            getattr(ext, GATEUP_OP[qtype])(h_x, h_gu[e], h_silu[e], 1, N,
                                           K_GU, GU_SCALES[e], 0)
            getattr(ext, ACC_OP[qtype])(h_silu[e], h_dn[e], h_acc, 1, DN_ROWS,
                                        N, DN_SCALES[e], 0)
        silu = [ext.download_resident(h, [1, N]).numpy() for h in h_silu]
        acc = ext.download_resident(h_acc, [1, DN_ROWS]).numpy()
        return silu, acc
    finally:
        for h in [h_x, h_acc] + h_gu + h_dn + h_silu:
            ext.free_resident(h)


def _run_fused(ext, qtype, arena_gu: np.ndarray, arena_dn: np.ndarray,
               ids_gu: np.ndarray, ids_dn: np.ndarray, x: np.ndarray,
               rows_gu: int, rows_dn: int, *, batched: bool):
    h_x = _upload(ext, x)
    h_gu = _upload(ext, arena_gu)
    h_dn = _upload(ext, arena_dn)
    h_ids_gu = ext.upload_resident(_ids(ids_gu.tolist()))
    h_ids_dn = ext.upload_resident(_ids(ids_dn.tolist()))
    h_gs = ext.upload_resident(_gs())
    h_silu = ext.allocate_resident(EXPERTS * N * 4)
    h_acc = ext.upload_resident(torch.zeros(1, DN_ROWS, dtype=torch.float32))
    try:
        if batched:
            ext.batch_begin()
            ext.batch_fill(h_acc, 0.0, DN_ROWS)
            ext.batch_kquant_gateup8_io(qtype, h_x, h_gu, h_silu, h_ids_gu,
                                        h_gs, N, K_GU, rows_gu)
            ext.batch_kquant_acc8_io(qtype, h_silu, h_dn, h_acc, h_ids_dn,
                                     h_gs, DN_ROWS, N, rows_dn, N)
            silu_t = torch.empty(EXPERTS * N)
            acc_t = torch.empty(1, DN_ROWS)
            ext.batch_download(h_silu, silu_t)
            ext.batch_download(h_acc, acc_t)
            ext.batch_submit()
            return silu_t.numpy().reshape(EXPERTS, N)[None], acc_t.numpy()
        ext.linear_kquant_gateup8_io(qtype, h_x, h_gu, h_silu, h_ids_gu,
                                     h_gs, N, K_GU, rows_gu)
        ext.linear_kquant_acc8_io(qtype, h_silu, h_dn, h_acc, h_ids_dn,
                                  h_gs, DN_ROWS, N, rows_dn, N)
        silu = ext.download_resident(h_silu, [1, EXPERTS * N]).numpy()
        acc = ext.download_resident(h_acc, [1, DN_ROWS]).numpy()
        return silu.reshape(1, EXPERTS, N), acc
    finally:
        for h in [h_x, h_gu, h_dn, h_ids_gu, h_ids_dn, h_gs, h_silu, h_acc]:
            ext.free_resident(h)


def _compare(silu_fused, acc_fused, silu_ref, acc_ref, label: str) -> None:
    for e in range(EXPERTS):
        _assert_bit_identical(silu_fused[0, e], silu_ref[e][0],
                              f"{label}: silu expert {e}")
    _assert_bit_identical(acc_fused, acc_ref, f"{label}: acc")


# ---------------------------------------------------------------------------
# GATE A: fused pair vs per-expert sequence over the same 8 experts,
# contiguous arena, ids = [0..7] - bit-identical, plus numpy-golden sanity.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("qtype", ["q4_k", "q5_k"])
def test_fused_matches_per_expert_bit_exact(ext, qtype):
    gu, dn, x = _scenario(qtype)
    rowb_gu = row_bytes(int(QENUM[qtype]), K_GU)
    rowb_dn = row_bytes(int(QENUM[qtype]), N)
    rows_gu, rows_dn = 2 * N, DN_ROWS

    arena_gu = np.concatenate(gu)                  # 8 bundles, slot e at e*2N
    arena_dn = np.concatenate(dn)                  # 8 matrices, slot e at e*N...
    assert arena_gu.shape == (EXPERTS * rows_gu, rowb_gu)
    ids = np.arange(EXPERTS, dtype=np.uint32)

    silu_ref, acc_ref = _run_per_expert(ext, qtype, gu, dn, x, batched=False)
    silu_f, acc_f = _run_fused(ext, qtype, arena_gu, arena_dn, ids, ids, x,
                               rows_gu, rows_dn, batched=False)

    # The fusion gate itself: exact bytes.
    _compare(silu_f, acc_f, silu_ref, acc_ref,
             f"{qtype} fused-vs-per-expert (sync)")

    # Sanity of the synthetic slabs against the numpy golden dequant (the
    # per-expert arm is already gated vs numpy in test_kquant_shaders; this
    # re-proves the reference the fusion matches is itself correct).
    silu_np, acc_np = _numpy_refs(qtype, gu, dn, x)
    for e in range(EXPERTS):
        _assert_close(silu_ref[e], silu_np[e], f"{qtype} per-expert vs numpy silu {e}")
    _assert_close(acc_ref, acc_np, f"{qtype} per-expert vs numpy acc")


# ---------------------------------------------------------------------------
# GATE B: slot0/arena addressing.  Slot 0 + a trailing slot hold the
# 0x7FFFFFFF sentinel, experts live in slots 1..8 in SHUFFLED order; the
# fused dispatch with slot0 = S != 0 must equal the per-expert sequence.
# Poison slots bracket the experts, so any cross-slot bleed yields NaN
# (torch.isfinite) or a byte mismatch at the slab edges.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("qtype", ["q4_k", "q5_k"])
def test_fused_arena_slot0_bit_exact(ext, qtype):
    gu, dn, x = _scenario(qtype)
    rowb_gu = row_bytes(int(QENUM[qtype]), K_GU)
    rowb_dn = row_bytes(int(QENUM[qtype]), N)
    rows_gu, rows_dn = 2 * N, DN_ROWS
    rng = np.random.default_rng(51)

    slots = rng.permutation(np.arange(1, EXPERTS + 1))  # expert e -> slot
    ids = slots.astype(np.uint32)
    total_gu = (1 + EXPERTS + 1) * rows_gu   # sentinel + 8 slots + sentinel
    total_dn = (1 + EXPERTS + 1) * rows_dn
    arena_gu = _poison(total_gu, rowb_gu)
    arena_dn = _poison(total_dn, rowb_dn)
    for e in range(EXPERTS):
        s = int(slots[e])
        arena_gu[s * rows_gu:(s + 1) * rows_gu] = gu[e]
        arena_dn[s * rows_dn:(s + 1) * rows_dn] = dn[e]

    silu_ref, acc_ref = _run_per_expert(ext, qtype, gu, dn, x, batched=False)
    silu_f, acc_f = _run_fused(ext, qtype, arena_gu, arena_dn, ids, ids, x,
                               rows_gu, rows_dn, batched=False)

    # Slab edges: the expert stored in the FIRST slot sits right after the
    # sentinel row 0, the one in the LAST slot right before the trailing
    # sentinel - both must be byte-exact (no over/under-reach).
    first_e = int(np.argmin(slots))
    last_e = int(np.argmax(slots))
    _compare(silu_f, acc_f, silu_ref, acc_ref,
             f"{qtype} fused arena slot0=S (shuffled)")
    # Explicit edge call-outs (subset of the full compare, kept because the
    # brief asks for them by name): NaN/Inf-free + first/last slot exact.
    assert np.isfinite(silu_f).all() and np.isfinite(acc_f).all(), \
        f"{qtype}: sentinel slot leaked NaN/Inf into the fused outputs"
    _assert_bit_identical(silu_f[0, first_e], silu_ref[first_e][0],
                          f"{qtype}: first-slot expert {first_e}")
    _assert_bit_identical(silu_f[0, last_e], silu_ref[last_e][0],
                          f"{qtype}: last-slot expert {last_e}")


# ---------------------------------------------------------------------------
# GATE C: the BATCHED record path (what the engine actually runs) - 16
# batch_kquant_io dispatches over the arena (per-expert slot0 offsets)
# vs the 2 fused batch dispatches over the same bytes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("qtype", ["q4_k", "q5_k"])
def test_fused_batch_path_bit_exact(ext, qtype):
    gu, dn, x = _scenario(qtype)
    rowb_gu = row_bytes(int(QENUM[qtype]), K_GU)
    rowb_dn = row_bytes(int(QENUM[qtype]), N)
    rows_gu, rows_dn = 2 * N, DN_ROWS
    arena_gu = np.concatenate(gu)
    arena_dn = np.concatenate(dn)
    ids = np.arange(EXPERTS, dtype=np.uint32)

    silu_ref, acc_ref = _run_per_expert(ext, qtype, gu, dn, x, batched=True)
    silu_f, acc_f = _run_fused(ext, qtype, arena_gu, arena_dn, ids, ids, x,
                               rows_gu, rows_dn, batched=True)
    _compare(silu_f, acc_f, silu_ref, acc_ref,
             f"{qtype} fused-vs-per-expert (batch)")


# ---------------------------------------------------------------------------
# GATE D: batch_moe_kquant_fused_record (the engine-facing crossing) vs
# the manual 16 dispatches + the shared-expert tail it subsumes.  The
# shared tail is BF16 dense here (small random weights) and must come out
# byte-identical too - the record must not perturb it.
# ---------------------------------------------------------------------------
def _record_arm(ext, qtype, arena_gu, arena_dn, ids, x, weights,
                rows_gu, rows_dn, manual: bool):
    h_sg_w, h_su_w, h_gsc_w, h_sdp_w = weights  # host bf16 tensors
    sg_rows, gsc_rows = int(h_sg_w.shape[0]), int(h_gsc_w.shape[0])
    sdp_rows, sdp_cols = int(h_sdp_w.shape[0]), int(h_sdp_w.shape[1])

    h_x = _upload(ext, x)
    h_gu = _upload(ext, arena_gu)
    h_dn = _upload(ext, arena_dn)
    h_ids = ext.upload_resident(_ids(ids.tolist()))
    h_gs = ext.upload_resident(_gs())
    h_sgw = ext.upload_resident(h_sg_w)
    h_suw = ext.upload_resident(h_su_w)
    h_gscw = ext.upload_resident(h_gsc_w)
    h_sdpw = ext.upload_resident(h_sdp_w)
    h_acc = ext.upload_resident(torch.zeros(1, DN_ROWS, dtype=torch.float32))
    h_silu_all = ext.allocate_resident(EXPERTS * N * 4)
    h_sg = ext.allocate_resident(sg_rows * 4)
    h_su = ext.allocate_resident(sg_rows * 4)
    h_gsc = ext.allocate_resident(gsc_rows * 4)
    h_x3 = ext.allocate_resident(sg_rows * 4)
    h_sdown = ext.allocate_resident(sdp_rows * 4)
    slots = [ext.allocate_resident(N * 4) for _ in range(EXPERTS)]
    silu_all_t = torch.empty(EXPERTS * N)
    slot_ts = [torch.empty(1, N) for _ in range(EXPERTS)]
    acc_t = torch.empty(DN_ROWS)
    sd_t = torch.empty(sdp_rows)
    gsc_t = torch.empty(gsc_rows)
    try:
        ext.batch_begin()
        ext.batch_fill(h_acc, 0.0, DN_ROWS)
        if manual:
            # Exactly what gguf_batched._record_moe_in records today:
            # batch_moe_kquant_record's 16 dispatches (arena slot0 offsets
            # stand in for the engine's per-expert handles)...
            for e in range(EXPERTS):
                ext.batch_kquant_io(qtype, "gateup", h_gu, h_x, slots[e],
                                    1, N, K_GU, GU_SCALES[e], e * rows_gu)
                ext.batch_kquant_io(qtype, "acc", h_dn, slots[e], h_acc,
                                    1, DN_ROWS, N, DN_SCALES[e], e * rows_dn)
            # ...then the caller's shared-expert tail, in order.
            ext.batch_bf16_io(h_x, h_sgw, h_sgw, h_sg, 1, sg_rows, K_GU, False)
            ext.batch_bf16_io(h_x, h_suw, h_suw, h_su, 1, sg_rows, K_GU, False)
            ext.batch_bf16_io(h_x, h_gscw, h_gscw, h_gsc, 1, gsc_rows, K_GU, False)
            ext.batch_swiglu_io(h_sg, h_su, h_x3, sg_rows)
            ext.batch_bf16_io(h_x3, h_sdpw, h_sdpw, h_sdown, 1,
                              sdp_rows, sdp_cols, False)
            ext.batch_download(h_acc, acc_t)
            ext.batch_download(h_sdown, sd_t)
            ext.batch_download(h_gsc, gsc_t)
            for h, t in zip(slots, slot_ts):
                ext.batch_download(h, t)
            ext.batch_submit()
            silu = np.concatenate([t.numpy()[0] for t in slot_ts]).reshape(
                1, EXPERTS, N)
            return silu, acc_t.numpy(), sd_t.numpy(), gsc_t.numpy()
        ext.batch_moe_kquant_fused_record(
            h_x, h_acc, h_sg, h_su, h_gsc, h_x3, h_sdown,
            qtype, h_silu_all, h_gu, h_dn, h_ids, h_ids, h_gs,
            N, K_GU, rows_gu, DN_ROWS, N, rows_dn, N,
            h_sgw, sg_rows, K_GU,
            h_suw, sg_rows, K_GU,
            h_gscw, gsc_rows, K_GU,
            h_sdpw, sdp_rows, sdp_cols,
            sg_rows,
            acc_t, sd_t, gsc_t,
        )
        ext.batch_download(h_silu_all, silu_all_t)
        ext.batch_submit()
        silu = silu_all_t.numpy().reshape(1, EXPERTS, N)
        return silu, acc_t.numpy(), sd_t.numpy(), gsc_t.numpy()
    finally:
        for h in [h_x, h_gu, h_dn, h_ids, h_gs, h_sgw, h_suw, h_gscw,
                  h_sdpw, h_acc, h_silu_all, h_sg, h_su, h_gsc, h_x3,
                  h_sdown] + slots:
            ext.free_resident(h)


@pytest.mark.parametrize("qtype", ["q4_k", "q5_k"])
def test_batch_moe_kquant_fused_record_matches_manual(ext, qtype):
    gu, dn, x = _scenario(qtype)
    rowb_gu = row_bytes(int(QENUM[qtype]), K_GU)
    rowb_dn = row_bytes(int(QENUM[qtype]), N)
    rows_gu, rows_dn = 2 * N, DN_ROWS
    arena_gu = np.concatenate(gu)
    arena_dn = np.concatenate(dn)
    ids = np.arange(EXPERTS, dtype=np.uint32)

    # Shared-expert tail: gate/up/gsc project x -> 64 rows, swiglu folds to
    # 64, sdp maps 64 -> 256.  Weights drawn ONCE so both arms see the same
    # bytes (random-but-finite, converted through bf16, not bit-viewed).
    rng = np.random.default_rng(9)
    def bf(r, c):
        return torch.from_numpy(
            (rng.standard_normal((r, c)) * 0.05).astype(np.float32)
        ).to(torch.bfloat16)
    weights = (bf(64, K_GU), bf(64, K_GU), bf(64, K_GU), bf(256, 64))

    silu_m, acc_m, sd_m, gsc_m = _record_arm(
        ext, qtype, arena_gu, arena_dn, ids, x, weights, rows_gu, rows_dn,
        manual=True)
    silu_f, acc_f, sd_f, gsc_f = _record_arm(
        ext, qtype, arena_gu, arena_dn, ids, x, weights, rows_gu, rows_dn,
        manual=False)

    _assert_bit_identical(silu_f, silu_m, f"{qtype} record: silu_all")
    _assert_bit_identical(acc_f, acc_m, f"{qtype} record: acc")
    _assert_bit_identical(sd_f, sd_m, f"{qtype} record: shared down")
    _assert_bit_identical(gsc_f, gsc_m, f"{qtype} record: shared gate scalar")


# ---------------------------------------------------------------------------
# Measurement: what the fusion buys at the kernel level.  Realistic
# CyberTiel-ish layer dims, both arms back to back, median of >=3, one
# printed JSON line (run pytest -s to see it).  Steady-state decode: the
# packed arenas are pre-uploaded (expert cache hot), so the wall numbers
# are the batch record+submit+fence path the engine pays per MoE layer.
# ---------------------------------------------------------------------------
def _meminfo() -> dict:
    out: dict[str, object] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            if k in ("Cached", "MemAvailable", "MemFree"):
                out[k] = rest.strip()
    with open("/proc/self/statm") as f:
        rss_pages = int(f.read().split()[1])
    out["rss_mb"] = rss_pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    return out


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


@pytest.mark.parametrize("qtype", ["q4_k"])
def test_fused_dispatch_bench(ext, qtype):
    GU_ROWS, GU_COLS = 512, 2048     # gate rows (N), gateup K
    DN_R, DN_C = 2048, 512           # down rows, down K == silu length
    rows_gu, rows_dn = 2 * GU_ROWS, DN_R

    gu = [np.concatenate([random_packed(qtype, GU_ROWS, GU_COLS, seed=2100 + e),
                          random_packed(qtype, GU_ROWS, GU_COLS, seed=2200 + e)])
          for e in range(EXPERTS)]
    dn = [random_packed(qtype, DN_R, DN_C, seed=2300 + e)
          for e in range(EXPERTS)]
    arena_gu = np.concatenate(gu)
    arena_dn = np.concatenate(dn)
    ids = np.arange(EXPERTS, dtype=np.uint32)
    x = (np.random.default_rng(77).standard_normal((1, GU_COLS)) * 0.1
         ).astype(np.float32)
    gs = _gs()
    content_gu = int(sum(b.nbytes for b in gu))
    content_dn = int(sum(b.nbytes for b in dn))
    rowb_gu = row_bytes(int(QENUM[qtype]), GU_COLS)
    rowb_dn = row_bytes(int(QENUM[qtype]), DN_C)

    h_x = _upload(ext, x)
    h_gu = _upload(ext, arena_gu)
    h_dn = _upload(ext, arena_dn)
    h_ids = ext.upload_resident(_ids(ids.tolist()))
    h_gs = ext.upload_resident(gs)
    h_acc = ext.upload_resident(torch.zeros(1, DN_R, dtype=torch.float32))
    h_silu = ext.allocate_resident(EXPERTS * GU_ROWS * 4)
    slots = [ext.allocate_resident(GU_ROWS * 4) for _ in range(EXPERTS)]

    def arm_per_expert():
        ext.batch_begin()
        ext.batch_fill(h_acc, 0.0, DN_R)
        for e in range(EXPERTS):
            ext.batch_kquant_io(qtype, "gateup", h_gu, h_x, slots[e], 1,
                                GU_ROWS, GU_COLS, 1.0, e * rows_gu)
            ext.batch_kquant_io(qtype, "acc", h_dn, slots[e], h_acc, 1,
                                DN_R, DN_C, 1.0, e * rows_dn)
        ext.batch_submit()

    def arm_fused(with_fill: bool = True):
        ext.batch_begin()
        if with_fill:
            ext.batch_fill(h_acc, 0.0, DN_R)
        ext.batch_kquant_gateup8_io(qtype, h_x, h_gu, h_silu, h_ids, h_gs,
                                    GU_ROWS, GU_COLS, rows_gu)
        ext.batch_kquant_acc8_io(qtype, h_silu, h_dn, h_acc, h_ids, h_gs,
                                 DN_R, DN_C, rows_dn, GU_ROWS)
        ext.batch_submit()

    def time_back_to_back(fn_a, fn_b, reps: int = 3):
        fn_a(); fn_b()          # warmup: pipeline creation + cache fills
        ta, tb = [], []
        for _ in range(reps):
            t0 = time.perf_counter(); fn_a(); ta.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter(); fn_b(); tb.append((time.perf_counter() - t0) * 1e3)
        return _median(ta), _median(tb)

    try:
        wall_before, wall_after = time_back_to_back(arm_per_expert, arm_fused)
        # Isolated fused GEMV (no fill/download): quantized-weight GB/s.
        arm_fused(with_fill=True)  # pre-fill the accumulator once
        iso = []
        for _ in range(3):
            t0 = time.perf_counter(); arm_fused(with_fill=False)
            iso.append((time.perf_counter() - t0) * 1e3)
        wall_iso = _median(iso)
        gbs = (content_gu + content_dn) / (wall_iso * 1e-3) / 1e9
        report = {
            "qtype": qtype,
            "dispatches_per_8_experts_before": 16,
            "dispatches_per_8_experts_after": 2,
            "wall_ms_per_8_experts_before": round(wall_before, 4),
            "wall_ms_per_8_experts_after": round(wall_after, 4),
            "wall_ms_fused_iso": round(wall_iso, 4),
            "h2d_bytes_before": content_gu + content_dn,
            "h2d_bytes_after": content_gu + content_dn,
            "arena_sentinel_overhead_bytes": rowb_gu * rows_gu + rowb_dn * rows_dn,
            "gbs_of_quantized_weights": round(gbs, 2),
            "meminfo": _meminfo(),
        }
        print("KFUSED_BENCH " + json.dumps(report))
        assert wall_before > 0 and wall_after > 0 and gbs > 0
    finally:
        for h in [h_x, h_gu, h_dn, h_ids, h_gs, h_acc, h_silu] + slots:
            ext.free_resident(h)
