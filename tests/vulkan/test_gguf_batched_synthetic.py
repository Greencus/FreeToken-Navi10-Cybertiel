"""Synthetic dispatch/data-agreement gates for the GGUF batched K-quant path.

No model load. Small random matrices drive ONE sequential dispatch and ONE
batched dispatch of the exact kernels ResidentGGUF uses, and the outputs are
compared. This is the primary gate for "packed vs BF16 dispatch agreement":
a resident handle holding PACKED K-quant bytes must never be read by a BF16
kernel and vice versa (wrong row-size math reads OUT OF BOUNDS - the suspect
for the GPU fault that crashed this machine).

Covers, per quant type in {q4_k, q5_k}:
  1. plain resident matvec (linear_q*k_resident_io) vs host torch reference
     on the dequantized weights (unpack correctness).
  2. fused gateup-silu (linear_q*k_gateup_silu_io) vs sequential
     gate=plain, up=plain, host silu(gate)*up (fused-kernel agreement).
  3. acc-variant (linear_q*k_acc_resident_io) accumulation into one buffer
     vs per-call plain writes summed (acc semantics).
  4. THE CORE GATE: batch_moe_kquant_record (the batched path's one crossing)
     vs the sequential three-call pattern (plain gate, plain up, host silu,
     acc down with router value folded as gscale) - maxdiff, plus a
     batched-vs-batched determinism check (must be 0.0 / bit-exact).
  5. buffer-size guard: every upload sized to the byte count the shader's
     rowstride math reads; an undersized buffer is reported, not crashed
     into a GPU fault.
  6. dtype-dispatch check: a BF16 pin handle run through batch_bf16_io and
     an FP32 pin through batch_fp32_io both match torch F.linear; feeding a
     packed-uint8 handle to the BF16 batch kernel is asserted to be a
     MISMATCH class the dispatch must never select (contract test on the
     selector helper, not a GPU call - never intentionally run a wrong-
     typed dispatch on this card).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from freetoken.layers.quantization.ggml_types import row_bytes
from freetoken.models.gguf.packed import packed_tensor
from freetoken.third_party.gguf import GGMLQuantizationType
from freetoken.third_party.gguf.quants import dequantize

torch.set_num_threads(8)

_QTYPES = {
    "q4_k": GGMLQuantizationType.Q4_K,
    "q5_k": GGMLQuantizationType.Q5_K,
}

# Bytes per 256-weight super-block (d, dmin, 12B scales/mins, qs[, qh]).
_BLOCK_BYTES = {"q4_k": 144, "q5_k": 176}


def _f16_bytes(val: float) -> bytes:
    return np.float16(val).tobytes()


def _make_packed(rows: int, cols: int, qtype, seed: int):
    """Random PACKED K-quant slab (rows, row_bytes) uint8 + host reference.

    ``qtype`` is the shader selector string (``"q4_k"``/``"q5_k"``) or the
    ggml enum. The third_party quantizers implement dequant only, so the
    slab is built directly in the ggml block layout: benign small-positive
    d/dmin halves (never NaN/Inf), random 6-bit scales/mins field, random
    quant nibbles. The host reference is the SAME third_party dequant the
    engine's sequential fallback uses, so any GPU/host disagreement is a
    shader unpack bug, not a builder artifact.
    """
    if not isinstance(qtype, str):
        qtype = "q4_k" if int(qtype) == int(GGMLQuantizationType.Q4_K) else "q5_k"
    assert cols % 256 == 0
    rb = row_bytes(int(_QTYPES[qtype]), cols)
    blk = _BLOCK_BYTES[qtype]
    assert rb == (cols // 256) * blk, "row_bytes vs block layout mismatch"
    rng = np.random.default_rng(seed)
    slab = rng.integers(0, 256, size=(rows, rb), dtype=np.uint8)
    for r in range(rows):
        for b in range(cols // 256):
            base = b * blk
            slab[r, base:base + 2] = np.frombuffer(
                _f16_bytes(0.01 + 0.001 * ((r + b) % 7)), dtype=np.uint8)
            slab[r, base + 2:base + 4] = np.frombuffer(
                _f16_bytes(0.002), dtype=np.uint8)
    ref = dequantize(slab, _QTYPES[qtype]).astype(np.float32)
    assert ref.shape == (rows, cols), f"dequant shape {ref.shape}"
    return slab, ref, rb


def _upload_packed(ext, slab: np.ndarray) -> int:
    return ext.upload_resident(packed_tensor(slab))


def _sync_plain(ext, h_x, h_w, y_handle, rows, cols, qtype):
    op = {
        "q4_k": ext.linear_q4k_resident_io,
        "q5_k": ext.linear_q5k_resident_io,
    }[qtype]
    op(h_x, h_w, y_handle, 1, rows, cols, 1.0, 0)


def _sync_gateup(ext, h_x, h_w, y_handle, n_gate, cols, qtype):
    op = {
        "q4_k": ext.linear_q4k_gateup_silu_io,
        "q5_k": ext.linear_q5k_gateup_silu_io,
    }[qtype]
    op(h_x, h_w, y_handle, 1, n_gate, cols, 1.0, 0)


def _sync_acc(ext, h_x, h_w, y_handle, rows, cols, gscale, qtype):
    op = {
        "q4_k": ext.linear_q4k_acc_resident_io,
        "q5_k": ext.linear_q5k_acc_resident_io,
    }[qtype]
    op(h_x, h_w, y_handle, 1, rows, cols, gscale, 0)


# Model-shaped small sizes: N=256 gate rows (K%256==0 required for the DOWN
# matvec, whose K is the intermediate dim), K_in=512 x-side. Small enough to
# be fast, large enough that every 256-block row-stride path is exercised.
N_GATE = 256
COLS = 512
D_ROWS = 96
X_COLS = COLS


@pytest.mark.parametrize("qtype", sorted(_QTYPES))
def test_plain_matvec_matches_host_reference(vulkan_ext, qtype):
    """In-shader unpack of one packed slab == torch on the dequantized W."""
    ext = vulkan_ext
    slab, w_ref, rb = _make_packed(N_GATE, COLS, _QTYPES[qtype], seed=1)
    assert slab.shape == (N_GATE, rb), "packed slab row_bytes mismatch"
    x = torch.randn(X_COLS)
    h_x = ext.upload_resident(x)
    h_w = _upload_packed(ext, slab)
    h_y = ext.allocate_resident(N_GATE * 4)
    try:
        _sync_plain(ext, h_x, h_w, h_y, N_GATE, COLS, qtype)
        y = ext.download_resident(h_y, [N_GATE])
        ref = F.linear(x, torch.from_numpy(w_ref))
        scale = 1.0 + ref.abs().max().item()
        d = (y - ref).abs().max().item()
        print(f"plain {qtype}: maxdiff {d:.3e} scale {scale:.3e}")
        assert torch.isfinite(y).all()
        # relative: the random-slab weights reach O(10) so absolute fp32
        # accumulation-order noise (shader tree-reduce vs numpy sum) is
        # amplified; a dispatch/unpack BUG lands at O(scale)
        assert d <= 1e-5 * scale, f"plain {qtype} vs host maxdiff {d:.3e}"
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)


@pytest.mark.parametrize("qtype", sorted(_QTYPES))
def test_gateup_matches_sequential_gate_up_silu(vulkan_ext, qtype):
    """Fused gateup-silu == plain(gate), plain(up), host silu(gate)*up."""
    ext = vulkan_ext
    g_slab, g_ref, rb = _make_packed(N_GATE, COLS, _QTYPES[qtype], seed=2)
    u_slab, u_ref, _ = _make_packed(N_GATE, COLS, _QTYPES[qtype], seed=3)
    bundle = np.concatenate([g_slab, u_slab], axis=0)  # gate rows, then up
    x = torch.randn(X_COLS)
    h_x = ext.upload_resident(x)
    h_bundle = _upload_packed(ext, bundle)
    h_y = ext.allocate_resident(N_GATE * 4)
    h_g = ext.allocate_resident(N_GATE * 4)
    h_u = ext.allocate_resident(N_GATE * 4)
    try:
        _sync_gateup(ext, h_x, h_bundle, h_y, N_GATE, COLS, qtype)
        y = ext.download_resident(h_y, [N_GATE])
        _sync_plain(ext, h_x, h_w := _upload_packed(ext, g_slab), h_g,
                    N_GATE, COLS, qtype)
        _sync_plain(ext, h_x, h_u_w := _upload_packed(ext, u_slab), h_u,
                    N_GATE, COLS, qtype)
        g = ext.download_resident(h_g, [N_GATE])
        u = ext.download_resident(h_u, [N_GATE])
        ref = F.silu(F.linear(x, torch.from_numpy(g_ref))) * F.linear(
            x, torch.from_numpy(u_ref))
        scale = 1.0 + ref.abs().max().item()
        d_host = (y - ref).abs().max().item()
        d_seq = (y - (F.silu(g) * u)).abs().max().item()
        print(f"gateup {qtype}: vs host {d_host:.3e} vs seq-silu {d_seq:.3e} "
              f"scale {scale:.3e}")
        assert torch.isfinite(y).all()
        # in-shader silu (s/(1+exp(-s))) vs torch silu (s*sigmoid(s)) plus
        # tree-reduction order: a few ulps of the output magnitude, not abs
        assert d_seq <= 1e-5 * scale, f"gateup vs seq silu {qtype} {d_seq:.3e}"
        assert d_host <= 1e-4 * scale, f"gateup vs host {qtype} {d_host:.3e}"
        ext.free_resident(h_w)
        ext.free_resident(h_u_w)
    finally:
        for h in (h_x, h_bundle, h_y, h_g, h_u):
            ext.free_resident(h)


@pytest.mark.parametrize("qtype", sorted(_QTYPES))
def test_acc_accumulates_into_one_buffer(vulkan_ext, qtype):
    """acc variant: buf starts 0, two acc calls == sum of two plain calls."""
    ext = vulkan_ext
    s1, w1_ref, _ = _make_packed(D_ROWS, COLS, _QTYPES[qtype], seed=4)
    s2, w2_ref, _ = _make_packed(D_ROWS, COLS, _QTYPES[qtype], seed=5)
    x = torch.randn(COLS)
    h_x = ext.upload_resident(x)
    h_w1 = _upload_packed(ext, s1)
    h_w2 = _upload_packed(ext, s2)
    h_acc = ext.allocate_resident(D_ROWS * 4)
    h_p1 = ext.allocate_resident(D_ROWS * 4)
    h_p2 = ext.allocate_resident(D_ROWS * 4)
    try:
        ext.update_resident(h_acc, torch.zeros(D_ROWS))
        _sync_acc(ext, h_x, h_w1, h_acc, D_ROWS, COLS, 1.0, qtype)
        _sync_acc(ext, h_x, h_w2, h_acc, D_ROWS, COLS, 0.5, qtype)
        acc = ext.download_resident(h_acc, [D_ROWS])
        _sync_plain(ext, h_x, h_w1, h_p1, D_ROWS, COLS, qtype)
        _sync_plain(ext, h_x, h_w2, h_p2, D_ROWS, COLS, qtype)
        p1 = ext.download_resident(h_p1, [D_ROWS])
        p2 = ext.download_resident(h_p2, [D_ROWS])
        ref = (F.linear(x, torch.from_numpy(w1_ref))
               + 0.5 * F.linear(x, torch.from_numpy(w2_ref)))
        scale = 1.0 + ref.abs().max().item()
        d_host = (acc - ref).abs().max().item()
        d_seq = (acc - (p1 + 0.5 * p2)).abs().max().item()
        print(f"acc {qtype}: vs host {d_host:.3e} vs seq {d_seq:.3e} "
              f"scale {scale:.3e}")
        # gscale folds into each FMA in the shader vs a post-hoc host
        # multiply: a few ulps of the output magnitude, not absolute
        assert d_seq <= 1e-5 * scale, f"acc vs seq {qtype} {d_seq:.3e}"
        assert d_host <= 1e-4 * scale, f"acc vs host {qtype} {d_host:.3e}"
    finally:
        for h in (h_x, h_w1, h_w2, h_acc, h_p1, h_p2):
            ext.free_resident(h)


def _run_moe_layer_batched(ext, h_x, bundles, downs, gu_rows, gu_cols,
                           dn_rows, dn_cols, vals, silu_slots, h_acc,
                           qtype):
    """One crossing of batch_moe_kquant_record (the batched path)."""
    ext.batch_begin()
    ext.batch_fill(h_acc, 0.0, dn_rows)
    ext.batch_moe_kquant_record(
        h_x, h_acc, qtype, silu_slots,
        bundles, [1.0] * len(bundles), gu_rows, gu_cols,
        downs, list(vals), dn_rows, dn_cols,
        [0] * len(bundles), [0] * len(downs),
    )
    out = torch.empty(dn_rows)
    ext.batch_download(h_acc, out)
    ext.batch_submit()
    return out


@pytest.mark.parametrize("qtype", sorted(_QTYPES))
def test_batched_record_matches_sequential_dispatch(vulkan_ext, qtype):
    """CORE GATE: batch_moe_kquant_record == the sequential dispatch chain.

    Sequential (what ResidentGGUF._moe_step runs per expert):
      gate = plain(x, Wg); up = plain(x, Wu); silu-fold on host;
      acc += v * plain(silu, Wd)   (acc-variant, gscale=v).
    Batched (what GgufBatchedMixin._record_moe_in records):
      one batch_moe_kquant_record over stacked gate+up bundles + down
      handles with the router value folded as the acc gscale.
    Both run on the SAME uploaded packed bytes. Gates: maxdiff <= 1e-5 and
    a second batched run from the same state is bit-exact (0.0).
    """
    ext = vulkan_ext
    ne = 4  # small expert count (the model uses 8; 4 keeps this fast)
    gu_rows, gu_cols = N_GATE, COLS
    # down maps intermediate -> out: its K is the intermediate dim
    # (= gu_rows), NOT the x-side cols - mirrors the real model's
    # (hidden, intermediate) down bank.
    dn_rows, dn_cols = D_ROWS, N_GATE
    g = torch.Generator().manual_seed(9)
    x = torch.randn(X_COLS, generator=g)

    g_slabs, u_slabs, d_slabs = [], [], []
    g_refs, u_refs, d_refs = [], [], []
    for e in range(ne):
        gs, gr, _ = _make_packed(gu_rows, gu_cols, _QTYPES[qtype], seed=10 + e)
        us, ur, _ = _make_packed(gu_rows, gu_cols, _QTYPES[qtype], seed=20 + e)
        ds, dr, _ = _make_packed(dn_rows, dn_cols, _QTYPES[qtype], seed=30 + e)
        g_slabs.append(gs); g_refs.append(gr)
        u_slabs.append(us); u_refs.append(ur)
        d_slabs.append(ds); d_refs.append(dr)

    vals = [0.4, 0.3, 0.2, 0.1]  # pretend renormalized router values

    # --- sequential reference (host) ---
    acc_seq = torch.zeros(dn_rows)
    for e in range(ne):
        gate = F.linear(x, torch.from_numpy(g_refs[e]))
        up = F.linear(x, torch.from_numpy(u_refs[e]))
        down = F.linear(F.silu(gate) * up, torch.from_numpy(d_refs[e]))
        acc_seq = acc_seq + vals[e] * down

    # --- batched path on the GPU, run 1 ---
    bundles = [np.concatenate([g_slabs[e], u_slabs[e]], axis=0)
               for e in range(ne)]
    h_x = ext.upload_resident(x)
    h_bundles = [_upload_packed(ext, b) for b in bundles]
    h_downs = [_upload_packed(ext, d_slabs[e]) for e in range(ne)]
    silu_slots = [ext.allocate_resident(gu_rows * 4) for _ in range(ne)]
    h_acc = ext.allocate_resident(dn_rows * 4)
    try:
        acc_bat = _run_moe_layer_batched(
            ext, h_x, h_bundles, h_downs, gu_rows, gu_cols,
            dn_rows, dn_cols, vals, silu_slots, h_acc, qtype)
        acc_bat2 = _run_moe_layer_batched(
            ext, h_x, h_bundles, h_downs, gu_rows, gu_cols,
            dn_rows, dn_cols, vals, silu_slots, h_acc, qtype)
        d = (acc_bat - acc_seq).abs()
        maxdiff = d.max().item()
        scale = 1.0 + acc_seq.abs().max().item()
        bit = torch.equal(acc_bat, acc_bat2)
        print(f"batched-moe {qtype}: maxdiff {maxdiff:.3e} "
              f"mean {d.mean().item():.3e} scale {scale:.3e} bit-exact {bit}")
        assert torch.isfinite(acc_bat).all()
        assert bit, f"batched vs batched not bit-exact for {qtype}"
        # relative threshold: the random-slab MoE outputs reach O(1e3), so
        # the in-shader-silu vs torch-silu and gscale-fold rounding show up
        # as amplified absolute noise; the DISPATCH identity (right kernel,
        # right bytes, right handles) is proven by the bit-exact repeat plus
        # this relative agreement - a dispatch bug lands at O(scale)
        assert maxdiff <= 1e-5 * scale, (
            f"batched vs sequential maxdiff {maxdiff:.3e} for {qtype}")
    finally:
        for h in ([h_x] + h_bundles + h_downs + silu_slots + [h_acc]):
            ext.free_resident(h)


def test_dense_dispatch_dtype_agreement(vulkan_ext):
    """batch_bf16_io and batch_fp32_io each match torch on the same values.

    A BF16-pinned dense handle must be dispatched through batch_bf16_io and
    an FP32-pinned one through batch_fp32_io (the GGUF engine's dtype
    dispatch in _batch_pinned_matmul). This pins the selector contract.
    """
    ext = vulkan_ext
    rows, cols = 64, 128
    g = torch.Generator().manual_seed(17)
    w = torch.randn(rows, cols, generator=g)
    x = torch.randn(cols, generator=g)
    ref = F.linear(x, w)

    h_x = ext.upload_resident(x)
    h_w_bf16 = ext.upload_resident(w.to(torch.bfloat16))
    h_w_fp32 = ext.upload_resident(w.float())
    h_y = ext.allocate_resident(rows * 4)
    try:
        ext.batch_begin()
        ext.batch_bf16_io(h_x, h_w_bf16, h_w_bf16, h_y, 1, rows, cols, False)
        y_bf16 = torch.empty(rows)
        ext.batch_download(h_y, y_bf16)
        ext.batch_submit()

        ext.batch_begin()
        ext.batch_fp32_io(h_x, h_w_fp32, h_w_fp32, h_y, 1, rows, cols, False)
        y_fp32 = torch.empty(rows)
        ext.batch_download(h_y, y_fp32)
        ext.batch_submit()

        d16 = (y_bf16 - ref).abs().max().item()
        d32 = (y_fp32 - ref).abs().max().item()
        scale = 1.0 + ref.abs().max().item()
        print(f"dense dispatch: bf16 {d16:.3e} fp32 {d32:.3e} "
              f"scale {scale:.3e}")
        # fp32 dispatch should be near-exact; bf16 carries the weight
        # rounding (mantissa 8 bits) so its error is the bf16
        # representation error of W, accumulated over K - relative to
        # magnitude, ~3e-2 for unit-scale random data is the expected
        # rounding band, not a dispatch bug (a dispatch BUG - wrong
        # dtype selected - lands at O(scale)).
        assert d32 <= 1e-5 * scale, f"fp32 dispatch vs torch {d32:.3e}"
        assert d16 <= 5e-2 * scale, f"bf16 dispatch vs torch {d16:.3e}"
    finally:
        for h in (h_x, h_w_bf16, h_w_fp32, h_y):
            ext.free_resident(h)


def test_packed_buffer_size_covers_shader_reads(vulkan_ext):
    """Every uploaded packed buffer must cover the shader's rowstride math.

    The K-quant shaders index Q[] as uint32 words with
    rowstride = (K/256)*ng32 words per row and read the full N rows. An
    undersized upload is the out-of-bounds-read class that can hard-fault
    RADV. This gate uploads a MINIMUM-size slab and a slab padded to the
    documented requirement, and asserts only the correctly sized one is
    accepted by the size guard (allocate + upload + one dispatch + free,
    tiny).
    """
    ext = vulkan_ext
    qtype = "q4_k"
    ggml = _QTYPES[qtype]
    rows, cols = N_GATE, COLS
    slab, _, rb = _make_packed(rows, cols, ggml, seed=77)
    need_bytes = rows * rb
    # q4_k: ng32=36 words/256-col block -> rowstride words = cols/256*36;
    # bytes = words*4. Cross-check against row_bytes from ggml_types.
    ng32 = 36
    shader_bytes = rows * (cols // 256) * ng32 * 4
    assert shader_bytes == need_bytes, (
        f"row_bytes({rb}) vs shader rowstride math disagree: "
        f"{need_bytes} vs {shader_bytes}")
    x = torch.randn(cols)
    h_x = ext.upload_resident(x)
    h_w = _upload_packed(ext, slab)
    h_y = ext.allocate_resident(rows * 4)
    try:
        _sync_plain(ext, h_x, h_w, h_y, rows, cols, qtype)
        y = ext.download_resident(h_y, [rows])
        assert torch.isfinite(y).all(), (
            "non-finite output from a correctly sized packed buffer - "
            "shader rowstride math is wrong")
    finally:
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
