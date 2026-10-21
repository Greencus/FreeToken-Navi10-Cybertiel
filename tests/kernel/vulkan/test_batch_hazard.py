"""Batch hazard gates: a shared accumulator and mid-batch frees.

Regression for the multi-row MoE walk bug (see docs/vulkan and the walk
repro): rows 0..T-2 of a shared h_y_acc came back all-NaN when the whole
walk ran in ONE open batch while the expert LRU evicted (free_resident)
between rows.  Two engine-level defects combined:

  1. free_resident mid-batch could DESTROY the VkBuffer (pool full) while
     the still-open command buffer already recorded binds/dispatches on it;
  2. that destroy called desc_cache_clear, which reset the descriptor pool
     mid-batch - the set handles bound earlier were recycled for later
     dispatches, so the recorded commands executed with the WRONG buffers.

The fix defers both the destruction and the pool reset to batch drain
(after the submission fence).  These gates pin that contract:

  synthetic (always runs with a Vulkan device):
      dispatch D0 -> free the weight + churn buffers mid-batch -> dispatch
      D1 -> download both outputs; both must match the sequential
      references bit-for-bit (before the fix D0's output was garbage).

  model-gated (FREETOKEN_TEST_MODEL_GGUF):
      the faithful walk shape - T rows share one acc, per-layer MoE phase
      in one batch vs one batch per row - over every CyberTiel layer.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import freetoken.kernel.vulkan as vk  # noqa: E402

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


def _assert_close(out: np.ndarray, ref: np.ndarray, *, label: str) -> None:
    assert out.shape == ref.shape, (label, out.shape, ref.shape)
    assert np.isfinite(out).all(), f"{label}: non-finite output"
    diff = float(np.abs(out - ref).max())
    tol = 1e-4 * (float(np.abs(ref).max()) + 1e-9) + 1e-5
    assert diff <= tol, f"{label}: max abs diff {diff:.3e} > tol {tol:.3e}"


def test_batch_midbatch_free_readbacks(ext):
    """Freeing a dispatched weight mid-batch must not corrupt readbacks.

    D0 dispatches with W0 into Y0; W0 (and churn buffers that fill the
    device free list, forcing real destroys) are then freed while the batch
    is open; D1 dispatches with W1 into Y1; both outputs are downloaded in
    the same batch.  Before the fix the mid-batch destroys reset the
    descriptor pool and D0 executed with recycled (D1's) bindings - Y0 came
    back wrong/NaN.  After the fix destruction waits for the fence.
    """
    N, K = 256, 512
    w_bytes = N * K * 4
    g = torch.Generator().manual_seed(1234)
    W0 = (torch.randn(N, K, generator=g) * 0.05).contiguous()
    W1 = (torch.randn(N, K, generator=g) * 0.05).contiguous()
    X = (torch.randn(1, K, generator=g) * 0.1).contiguous()
    ref0 = (X.numpy() @ W0.numpy().T).astype(np.float32)
    ref1 = (X.numpy() @ W1.numpy().T).astype(np.float32)

    h_w0 = ext.upload_resident(W0)
    h_w1 = ext.upload_resident(W1)
    h_x = ext.upload_resident(X)
    h_y0 = ext.allocate_resident(N * 4)
    h_y1 = ext.allocate_resident(N * 4)
    # Same byte size as W0: their frees fill the per-size device free list
    # (kMaxPerSizeDev) so the trailing frees really destroy.
    churn = [ext.allocate_resident(w_bytes) for _ in range(20)]
    d0 = ext.batch_deferred_stats()
    try:
        ext.batch_begin()
        ext.batch_fill(h_y0, 0.0, N)
        ext.batch_fp32_io(h_x, h_w0, h_w0, h_y0, 1, N, K, False)
        # Mid-batch eviction churn: the first 16 frees pool, the rest (and
        # W0) destroy - the pre-fix path reset the desc pool right here.
        for h in churn:
            ext.free_resident(h)
        ext.free_resident(h_w0)
        h_w0 = -1
        ext.batch_fill(h_y1, 0.0, N)
        ext.batch_fp32_io(h_x, h_w1, h_w1, h_y1, 1, N, K, False)
        out0 = torch.empty(1, N)
        out1 = torch.empty(1, N)
        ext.batch_download(h_y0, out0)
        ext.batch_download(h_y1, out1)
        ext.batch_submit()
    finally:
        for h in (h_x, h_y0, h_y1, h_w1):
            if h >= 0:
                ext.free_resident(h)
        if h_w0 >= 0:
            ext.free_resident(h_w0)
        for h in churn:
            try:
                ext.free_resident(h)
            except Exception:
                pass  # already freed mid-batch
    d1 = ext.batch_deferred_stats()
    # The deferral path was exercised and fully drained at batch end.
    assert d1[0] > d0[0], f"no deferred frees recorded: {d0} -> {d1}"
    assert d1[1] == 0, f"deferred frees still pending after submit: {d1}"
    assert d1[3] == 0, f"cache-pool reset still pending after submit: {d1}"
    _assert_close(out0.numpy(), ref0, label="Y0 after mid-batch frees")
    _assert_close(out1.numpy(), ref1, label="Y1")


@requires_gguf
def test_multirow_walk_shared_acc(ext):
    """The faithful walk shape: T rows share one acc across every layer.

    Mirrors the pre-workaround decode_rows_batched MoE phase: one open
    batch carries all T rows (rows share h_y_acc, fill+dispatch+download
    per row) against the real CyberTiel decoder.  Every acc/sd/gsc readback
    must be finite and bit-identical to the one-batch-per-row reference
    (the current workaround shape).  Before the fix rows 0..T-2 went NaN
    from layer ~12 onward.
    """
    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

    model = ResidentGGUF(GGUF, ext)
    ext = model.ext
    model.reset()
    logits = model.prefill_chunked([760, 6511, 314, 9338, 369], chunk_size=64)
    assert torch.isfinite(logits).all(), "prefill produced non-finite logits"
    pos = model._seq_len
    cfg = model.config
    H = cfg.hidden_size
    T = 4

    def snap():
        return {
            "gdn": {i: t.clone() for i, t in model.gdn_state.items()},
            "conv": {i: t.clone() for i, t in model.conv_buf.items()},
            "k": {i: t.clone() for i, t in model.k_cache.items()},
            "v": {i: t.clone() for i, t in model.v_cache.items()},
            "seq": model._seq_len,
        }

    def restore(s):
        for i, t in s["gdn"].items():
            model.gdn_state[i].copy_(t)
        for i, t in s["conv"].items():
            model.conv_buf[i].copy_(t)
        for i, t in s["k"].items():
            model.k_cache[i].copy_(t)
        for i, t in s["v"].items():
            model.v_cache[i].copy_(t)
        model._seq_len = s["seq"]
        model._gdn_gpu_dirty = True

    s0 = snap()
    ids = [760, 6511, 314, 9338][:T]
    g = torch.Generator().manual_seed(1234)
    h2_rows = [0.05 * torch.randn(H, generator=g) for _ in range(T)]
    layer_list = list(enumerate(cfg.layer_types))

    def moe_phase(layer: int, mode: str):
        accs, sds, gscs = [], [], []
        if mode == "shared":
            ext.batch_begin()
            for t in range(T):
                ext.batch_upload(model.h_x_fused, h2_rows[t].contiguous().float())
                ext.batch_fill(model.h_y_acc, 0.0, H)
                a, s, c = model._record_moe_in(layer, h2_rows[t], ext)
                accs.append(a)
                sds.append(s)
                gscs.append(c)
            ext.batch_submit()
        else:
            for t in range(T):
                ext.batch_begin()
                ext.batch_upload(model.h_x_fused, h2_rows[t].contiguous().float())
                ext.batch_fill(model.h_y_acc, 0.0, H)
                a, s, c = model._record_moe_in(layer, h2_rows[t], ext)
                ext.batch_submit()
                accs.append(a)
                sds.append(s)
                gscs.append(c)
        return accs, sds, gscs

    def walk(mode: str):
        restore(s0)
        out = {}
        for i, lt in layer_list:
            out[i] = moe_phase(i, mode)
        return out

    shared = walk("shared")
    per = walk("per_row")
    for i, _lt in layer_list:
        for t in range(T):
            for kind, ref in (("acc", per[i][0][t]), ("sd", per[i][1][t]),
                              ("gsc", per[i][2][t])):
                got = shared[i][0][t] if kind == "acc" else (
                    shared[i][1][t] if kind == "sd" else shared[i][2][t])
                assert torch.isfinite(got).all(), (
                    f"layer {i} row {t} {kind}: shared batch produced NaN")
                diff = float((got - ref).abs().max())
                assert diff <= 1e-5, (
                    f"layer {i} row {t} {kind}: shared vs per-row maxdiff "
                    f"{diff:.3e}")
