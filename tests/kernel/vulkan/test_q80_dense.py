"""FTV_Q8_0 dense-shadow gates: the ext-side Q8_0 twin of big BF16 pins.

`FTV_Q80=1` (nonzero numeric; "true" stays OFF per the atoi convention)
makes `batch_bf16_io` / `linear_bf16_resident_io` serve qualifying weights
(`>= FTV_Q80_MIN_KB`, default 16000, budget-capped by `FTV_Q80_MB`,
default 700) from a Q8_0 twin packed once by `quant_bf16_q80.comp` and
read by `linear_q80_resident_f32.comp` - half the DRAM bytes per GEMV.

Bar this file enforces:

  * correctness against the plain torch reference (int8 quantisation
    error bound, 2% of the global output scale),
  * SYNC (`linear_bf16_resident_io`) and BATCH (`batch_bf16_io`) are
    BIT-IDENTICAL - both arms must share the shadow or fork2's
    seq-vs-batched arm diff would blow its 1e-5 threshold,
  * run-to-run determinism (0.0),
  * a partial final block (K not a multiple of 32) stays exact - this is
    the bound-check regression that once zeroed every row > 0
    (`base + i >= p.K` compared a row-global index against the row
    length, so row n>=1 broke out at i=0 and packed d=0/q=0),
  * the env gates: FTV_Q80 unset -> pure bf16 no-op, MIN_KB and
    FTV_Q80_MB budget first-come skip, shape-mismatch on handle reuse,
  * free_resident(original) drops the shadow (stats bytes return).

Every test SKIPS cleanly when no Vulkan device is present.  Env knobs are
set through monkeypatch so the process-wide registry is the only shared
state; counters are asserted as deltas (they are cumulative).
"""
from __future__ import annotations

import pytest

import freetoken.kernel.vulkan as vk

torch = pytest.importorskip("torch")


def _skip_unless_vulkan() -> None:
    if not vk.available():
        pytest.skip("no usable Vulkan device/extension on this machine")


@pytest.fixture(scope="module")
def ext():
    _skip_unless_vulkan()
    return vk.require_ext()


def _q80_on(monkeypatch, min_kb: str = "0", mb: str | None = None) -> None:
    monkeypatch.setenv("FTV_Q80", "1")
    monkeypatch.setenv("FTV_Q80_MIN_KB", min_kb)
    if mb is not None:
        monkeypatch.setenv("FTV_Q80_MB", mb)


def _run_sync(ext, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """One sync-path GEMV; returns (1, N) fp32 output."""
    N, K = W.shape
    h_w = ext.upload_resident(W.contiguous())
    h_x = ext.upload_resident(X.contiguous())
    h_y = ext.allocate_resident(N * 4)
    ext.linear_bf16_resident_io(h_x, h_w, 0, h_y, 1, N, K, False)
    return ext.download_resident(h_y, (N,))


def _run_batch(ext, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """One batch-path GEMV; returns (N,) fp32 output."""
    N, K = W.shape
    h_w = ext.upload_resident(W.contiguous())
    h_x = ext.upload_resident(X.contiguous())
    h_y = ext.allocate_resident(N * 4)
    ext.batch_begin()
    ext.batch_bf16_io(h_x, h_w, 0, h_y, 1, N, K, False)
    ext.batch_submit()
    return ext.download_resident(h_y, (N,))


def _assert_q80_close(out, ref, *, label: str) -> None:
    out = [float(v) for v in (out.tolist() if torch.is_tensor(out) else out)]
    ref = [float(v) for v in (ref.flatten().tolist())]
    assert len(out) == len(ref), (label, len(out), len(ref))
    scale = max(abs(v) for v in ref) + 1e-9
    err = max(abs(a - b) for a, b in zip(out, ref))
    tol = 2e-2 * scale + 1e-5  # int8 half-LSB bound, well above measured ~4e-3
    assert err <= tol, f"{label}: maxdiff {err:.3e} > tol {tol:.3e} (scale {scale:.3e})"


@pytest.mark.parametrize("N,K", [(8, 64), (17, 100), (32, 2048)])
def test_q80_correctness_batch_sync_determinism(ext, monkeypatch, N, K):
    """Sync, batch, torch ref and a rerun must all agree."""
    _q80_on(monkeypatch)
    g = torch.Generator().manual_seed(1234)
    W = (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)
    X = torch.randn(1, K, generator=g)
    ref = W.float() @ X.T  # (N, 1)

    s0 = ext.q80_stats()
    y_sync = _run_sync(ext, W, X)
    y_batch = _run_batch(ext, W, X)
    s1 = ext.q80_stats()

    _assert_q80_close(y_sync, ref, label=f"sync {N}x{K}")
    _assert_q80_close(y_batch, ref, label=f"batch {N}x{K}")
    assert y_sync.tolist() == y_batch.tolist(), (
        f"batch != sync for {N}x{K}: fork2 arms would diverge"
    )
    # determinism: a second shadowed run through the same path
    y_again = _run_sync(ext, W, X)
    assert y_again.tolist() == y_sync.tolist(), f"nondeterministic {N}x{K}"
    # both paths created shadows (not merely fell back to bf16)
    assert s1["created"] - s0["created"] >= 1, "no Q8_0 shadow was created"
    assert s1["bytes"] > s0["bytes"], "shadow bytes not accounted"


def test_q80_partial_block_bound_check(ext, monkeypatch):
    """K=100: the 4th block holds only 4 values - the exact regression
    that zeroed every row > 0 when the bound test used the row-global
    index against the row length."""
    _q80_on(monkeypatch)
    N, K = 5, 100  # blocks = ceil(100/32) = 4, last block partial (4 vals)
    g = torch.Generator().manual_seed(7)
    W = (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)
    X = torch.randn(1, K, generator=g)
    ref = W.float() @ X.T
    y = _run_sync(ext, W, X)
    _assert_q80_close(y, ref, label="partial block 5x100")
    # every row must be nonzero: a zero row is the old bound bug
    assert all(abs(float(v)) > 1e-6 for v in y.tolist()), y.tolist()


def test_q80_default_off_is_noop(ext, monkeypatch):
    """FTV_Q80 unset: no shadow ever, output is the plain bf16 path."""
    monkeypatch.delenv("FTV_Q80", raising=False)
    s0 = ext.q80_stats()
    g = torch.Generator().manual_seed(3)
    N, K = 64, 512
    W = (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)
    X = torch.randn(1, K, generator=g)
    y = _run_batch(ext, W, X)
    s1 = ext.q80_stats()
    _assert_q80_close(y, W.float() @ X.T, label="bf16 fallback")
    assert not s1["enabled"], "q80_enabled() true without FTV_Q80"
    assert s1["created"] == s0["created"], "shadow created with env off"
    assert s1["bytes"] == s0["bytes"], "shadow bytes changed with env off"


def test_q80_min_kb_gate(ext, monkeypatch):
    """Below MIN_KB the weight stays on bf16: no shadow, no dispatch."""
    _q80_on(monkeypatch, min_kb="999999")
    g = torch.Generator().manual_seed(11)
    N, K = 64, 512  # 64 KiB - under the (huge) gate
    W = (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)
    X = torch.randn(1, K, generator=g)
    s0 = ext.q80_stats()
    y = _run_batch(ext, W, X)
    s1 = ext.q80_stats()
    _assert_q80_close(y, W.float() @ X.T, label="min_kb fallback")
    assert s1["created"] == s0["created"], "shadow created below MIN_KB"


def test_q80_budget_gate(ext, monkeypatch):
    """FTV_Q80_MB=0: first-come budget refuses the twin, counts a skip."""
    _q80_on(monkeypatch, min_kb="0", mb="0")
    g = torch.Generator().manual_seed(13)
    N, K = 64, 512
    W = (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)
    X = torch.randn(1, K, generator=g)
    s0 = ext.q80_stats()
    y = _run_batch(ext, W, X)
    s1 = ext.q80_stats()
    _assert_q80_close(y, W.float() @ X.T, label="budget skip fallback")
    assert s1["skipped_budget"] - s0["skipped_budget"] == 1, (
        "budget refusal not counted"
    )
    assert s1["created"] == s0["created"], "shadow created over budget"


def test_q80_shape_mismatch_raises(ext, monkeypatch):
    """A pinned handle must never be re-quantised at another shape."""
    _q80_on(monkeypatch)
    g = torch.Generator().manual_seed(17)
    W = (torch.randn(8, 64, generator=g) * 0.05).to(torch.bfloat16)
    X = torch.randn(1, 64, generator=g)
    h_w = ext.upload_resident(W.contiguous())
    h_x = ext.upload_resident(X.contiguous())
    h_y = ext.allocate_resident(8 * 4)
    ext.linear_bf16_resident_io(h_x, h_w, 0, h_y, 1, 8, 64, False)
    # Same handle, different claimed K: the registry hit must refuse.
    X2 = torch.randn(1, 128, generator=g)
    h_x2 = ext.upload_resident(X2.contiguous())
    h_y2 = ext.allocate_resident(8 * 4)
    with pytest.raises(RuntimeError, match="different shape"):
        ext.linear_bf16_resident_io(h_x2, h_w, 0, h_y2, 1, 8, 128, False)


def test_q80_free_original_drops_shadow(ext, monkeypatch):
    """free_resident(original) must free the twin and fix the accounting."""
    _q80_on(monkeypatch)
    W = torch.zeros(8, 64, dtype=torch.bfloat16) + 0.25
    X = torch.ones(1, 64)
    h_w = ext.upload_resident(W.contiguous())
    h_x = ext.upload_resident(X)
    h_y = ext.allocate_resident(8 * 4)
    ext.linear_bf16_resident_io(h_x, h_w, 0, h_y, 1, 8, 64, False)
    s1 = ext.q80_stats()
    assert s1["bytes"] >= (8 * 128), "shadow bytes missing before free"
    ext.free_resident(h_w)
    s2 = ext.q80_stats()
    assert s2["bytes"] <= s1["bytes"] - 8 * 128, (
        f"shadow bytes not dropped: {s1['bytes']} -> {s2['bytes']}"
    )


# ---------------------------------------------------------------------------
# FTV_DENSE_Q80: declared FILE slabs (ggml block_q8_0, 34 B blocks, rows
# stride cols/32*34 with no padding).  Declaration-driven: the dispatch
# keys on ext.declare_q80_packed, never on an env at record time, and the
# FTV_Q80 shadow env stays OFF for every test here.
# ---------------------------------------------------------------------------

def _pack_q80(W: torch.Tensor) -> torch.Tensor:
    """(N, K) fp values -> (N, K/32*34) uint8, ggml block_q8_0 layout.

    d = fp16(amax/127) rounded FIRST, then q = clamp(round(w/d)) +-127;
    each 34 B block is [fp16 d little-endian | 32 int8 codes].
    """
    assert W.shape[1] % 32 == 0, "GGUF Q8_0 needs K%32==0"
    N, K = W.shape
    blocks = K // 32
    w = W.float().reshape(N, blocks, 32)
    d = (w.abs().amax(dim=-1) / 127.0).to(torch.float16)
    q = torch.round(w / d.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    slab = torch.empty(N, blocks, 34, dtype=torch.uint8)
    slab[..., 0:2] = d.contiguous().view(torch.uint8).reshape(N, blocks, 2)
    slab[..., 2:34] = q.view(torch.uint8).reshape(N, blocks, 32)
    return slab.reshape(N, blocks * 34)


def _q80_ref(slab: torch.Tensor, K: int) -> torch.Tensor:
    """Dequant reference of a packed slab: (N, K) fp32.

    Production contract: each q*d weight rounds to bf16 (RNE, exactly like
    the default dense path's stored weights) before the f32 accumulation -
    the packed Q8_0 kernel (default linear_q80gguf_wv) does the same so its
    outputs are bit-identical to linear_bf16_resident (the layer-parity
    gates rely on that).
    """
    N = slab.shape[0]
    blocks = K // 32
    v = slab.view(N, blocks, 34)
    d = v[..., 0:2].contiguous().view(torch.float16).reshape(N, blocks).float()
    q = v[..., 2:34].contiguous().view(torch.int8).reshape(N, blocks, 32).float()
    return (q * d.unsqueeze(-1)).reshape(N, K).to(torch.bfloat16).float()


def _run_packed(ext, slab: torch.Tensor, X: torch.Tensor, *, batch: bool):
    """Upload + declare + one GEMV; returns ((N,) fp32, handle)."""
    N, K = slab.shape[0], slab.shape[1] // 34 * 32
    h_w = ext.upload_resident(slab.contiguous())
    ext.declare_q80_packed(h_w, N, K)
    h_x = ext.upload_resident(X.contiguous())
    h_y = ext.allocate_resident(N * 4)
    if batch:
        ext.batch_begin()
        ext.batch_bf16_io(h_x, h_w, 0, h_y, 1, N, K, False)
        ext.batch_submit()
    else:
        ext.linear_bf16_resident_io(h_x, h_w, 0, h_y, 1, N, K, False)
    return ext.download_resident(h_y, (N,)), h_w


def test_q80_packed_sync_batch_bitexact(ext, monkeypatch):
    """Declared file slab: sync == batch bit-for-bit (fork2 arms), both
    matching the dequant reference of the slab bytes within fp32 noise."""
    monkeypatch.delenv("FTV_Q80", raising=False)
    g = torch.Generator().manual_seed(23)
    W = torch.randn(64, 2048, generator=g) * 0.05
    X = torch.randn(1, 2048, generator=g)
    slab = _pack_q80(W)
    s0 = ext.q80_stats()
    y_sync, h_w = _run_packed(ext, slab, X, batch=False)
    y_batch, _ = _run_packed(ext, slab, X, batch=True)
    s1 = ext.q80_stats()
    ref = (_q80_ref(slab, 2048) @ X.T).squeeze(-1)  # (N,) like the downloads
    err = (y_sync - ref).abs().max().item()
    assert err <= 1e-4 * (ref.abs().max().item() + 1e-9) + 1e-5, err
    assert y_sync.tolist() == y_batch.tolist(), "packed sync != batch"
    assert not s1["enabled"], "FTV_Q80 must stay OFF on the packed path"
    assert s1["created"] == s0["created"], "shadow created despite packing"
    assert s1["packed_declared"] - s0["packed_declared"] == 2, (
        "both declarations counted", s0, s1)
    assert s1["packed_dispatches"] - s0["packed_dispatches"] == 2, (
        "one GEMV per run served from the file slab")
    ext.free_resident(h_w)
    s2 = ext.q80_stats()
    assert s2["packed_declared"] == s1["packed_declared"] - 1, (
        "free did not drop the declaration", s1, s2)
    assert s2["packed_bytes"] == s1["packed_bytes"] - 64 * 2048 // 32 * 34


def test_q80_packed_unaligned_rowstride(ext, monkeypatch):
    """K=512: GGUF rowstride is 544 B (not 64 B aligned), the exact shape
    the shadow's aligned layout could not express."""
    monkeypatch.delenv("FTV_Q80", raising=False)
    g = torch.Generator().manual_seed(29)
    W = torch.randn(17, 512, generator=g) * 0.05
    X = torch.randn(1, 512, generator=g)
    slab = _pack_q80(W)
    assert slab.shape == (17, 512 // 32 * 34)
    y, h_w = _run_packed(ext, slab, X, batch=True)
    ref = (_q80_ref(slab, 512) @ X.T).squeeze(-1)
    err = (y - ref).abs().max().item()
    assert err <= 1e-4 * (ref.abs().max().item() + 1e-9) + 1e-5, err
    assert all(abs(float(v)) > 1e-7 for v in y.tolist()), y.tolist()
    ext.free_resident(h_w)


def test_q80_packed_shape_mismatch_raises(ext, monkeypatch):
    monkeypatch.delenv("FTV_Q80", raising=False)
    slab = _pack_q80(torch.randn(8, 64) * 0.05)
    h_w = ext.upload_resident(slab.contiguous())
    ext.declare_q80_packed(h_w, 8, 64)
    with pytest.raises(RuntimeError, match="different shape"):
        ext.declare_q80_packed(h_w, 16, 64)
    ext.free_resident(h_w)
    # freed handle: the declaration is gone, redeclare is a fresh entry
    h2 = ext.upload_resident(slab.contiguous())
    ext.declare_q80_packed(h2, 8, 64)
    ext.free_resident(h2)


def test_q80_packed_undelclared_is_plain_bf16(ext, monkeypatch):
    """No declaration + FTV_Q80 off: the bf16 path must run untouched
    (a packed declaration is the ONLY thing that can reroute a handle)."""
    monkeypatch.delenv("FTV_Q80", raising=False)
    g = torch.Generator().manual_seed(31)
    W = (torch.randn(64, 512, generator=g) * 0.05).to(torch.bfloat16)
    X = torch.randn(1, 512, generator=g)
    s0 = ext.q80_stats()
    y = _run_batch(ext, W, X)
    s1 = ext.q80_stats()
    _assert_q80_close(y, W.float() @ X.T, label="undeclared bf16")
    assert s1["packed_declared"] == s0["packed_declared"]
    assert s1["packed_dispatches"] == s0["packed_dispatches"]


def test_q80gguf_variants_bitidentical(ext, monkeypatch):
    """The vectorized Q8_0 GEMV variants (sg / lds / lds2 / rt / pt / tp / pa
    / pb / pv / w64 / wp / wpa / wb / wv, selected with set_q80gguf_variant
    or FTV_Q80_GGUF_VARIANT) must be BIT-IDENTICAL to the byte-gather
    reference on every path: batch and sync, aligned and unaligned row
    bases.  Shapes cover rowstride 2176
    (8-aligned, the production case), 544, 68 (4 mod 8 -> alternating row
    bases, the dword staging fallback) and 102 (odd block count -> 2 mod 8
    base).
    """
    monkeypatch.delenv("FTV_Q80", raising=False)
    prev = ext.q80gguf_variant()   # restore whatever the env preselected
    try:
        for N, K in [(64, 2048), (17, 512), (64, 64), (5, 96)]:
            g = torch.Generator().manual_seed(3)
            W = torch.randn(N, K, generator=g) * 0.05
            X = torch.randn(1, K, generator=g)
            slab = _pack_q80(W)
            ys: dict[str, tuple] = {}
            for var in ("", "ref", "sg", "lds", "lds2", "rt", "pt", "tp", "pa",
                        "pb", "pv", "w64", "wp", "wpa", "wb", "wv"):
                ext.set_q80gguf_variant(var)
                y_b, h_w = _run_packed(ext, slab, X, batch=True)
                y_s, _ = _run_packed(ext, slab, X, batch=False)
                ext.free_resident(h_w)
                ys[var] = (y_b, y_s)
            base_b, base_s = ys["ref"]
            for var, (y_b, y_s) in ys.items():
                assert y_b.tolist() == base_b.tolist(), (
                    f"variant {var!r} != ref (batch) at {N}x{K}")
                assert y_s.tolist() == base_s.tolist(), (
                    f"variant {var!r} != ref (sync) at {N}x{K}")
                assert y_b.tolist() == y_s.tolist(), (
                    f"variant {var!r}: sync != batch at {N}x{K} (fork2)")
    finally:
        ext.set_q80gguf_variant(prev)
