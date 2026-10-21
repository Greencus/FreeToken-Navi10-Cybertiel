"""Port of the fork's test_gdn_kernel.py: GDN GPU kernels vs host reference.

Uploads fixed qkv/z/a/b + zero state/conv_buf, runs gdn_conv + gdn_recur in
one batch, and compares pre_out against the resident decoder's _gdn_host
(the fork's gate: maxdiff < 1e-5, i.e. fp32 libm ~1ulp).

Checkpoint gating: the shared ``resident_model`` fixture prefers
``FREETOKEN_TEST_MODEL_GGUF`` (the CyberTiel GGUF, the shipped checkpoint)
and falls back to ``FREETOKEN_TEST_MODEL`` (NVFP4 dir, if still present);
skips when neither is set. The gate is weight-source-neutral: it reads
conv/A_log/dt_bias/norm through ``_dense`` and runs the same host
reference either way, so a GGUF run exercises the same shaders.
"""
from __future__ import annotations

import math
import os

import torch

torch.set_num_threads(8)
os.environ.setdefault("FTV_NVFP4_GPU", "1")

PREFIX = "language_model.model.layers"


def test_gdn_kernel_matches_host(vulkan_ext, resident_model):
    ext = vulkan_ext
    model = resident_model
    model.reset()
    cfg = model.config

    li = next(
        i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"
    )
    base = f"{PREFIX}.{li}"
    nk, hk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    nv, hv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    key_dim = nk * hk
    c = 2 * key_dim + nv * hv

    torch.manual_seed(7)
    qkv = torch.randn(c)
    z = torch.randn(nv * hv)
    a = torch.randn(nv)
    b = torch.randn(nv)

    # Host reference: zero state/conv_buf after reset.
    ref = model._gdn_host(li, qkv, z, a, b).clone()

    conv_w = model._dense(f"{base}.linear_attn.conv1d.weight")
    taps = conv_w[:, 0, :].float().contiguous()
    alog = model._dense(f"{base}.linear_attn.A_log").float().contiguous()
    dtb = model._dense(f"{base}.linear_attn.dt_bias").float().contiguous()
    normw_full = model._dense(f"{base}.linear_attn.norm.weight").float()
    normw = (
        normw_full.reshape(nv, hv)[0].contiguous()
        if normw_full.numel() == nv * hv
        else normw_full.contiguous()
    )

    handles = [
        ext.allocate_resident(c * 4),                 # h_q
        ext.allocate_resident(c * 3 * 4),             # h_buf
        ext.allocate_resident(taps.numel() * 4),      # h_taps
        ext.allocate_resident(c * 4),                 # h_mixed
        ext.allocate_resident(z.numel() * 4),         # h_z
        ext.allocate_resident(a.numel() * 4),         # h_a
        ext.allocate_resident(b.numel() * 4),         # h_b
        ext.allocate_resident(alog.numel() * 4),      # h_al
        ext.allocate_resident(dtb.numel() * 4),       # h_dt
        ext.allocate_resident(normw.numel() * 4),     # h_nw
        ext.allocate_resident(nv * hk * hv * 4),      # h_state
        ext.allocate_resident(nv * hv * 4),           # h_preout
    ]
    (h_q, h_bu, h_t, h_m, h_z, h_a, h_bv, h_al, h_dt, h_nw, h_st, h_po) = handles
    try:
        ext.batch_begin()
        ext.batch_upload(h_q, qkv.contiguous().float())
        ext.batch_fill(h_bu, 0.0, c * 3)
        ext.batch_upload(h_t, taps)
        ext.batch_upload(h_z, z.contiguous().float())
        ext.batch_upload(h_a, a.contiguous().float())
        ext.batch_upload(h_bv, b.contiguous().float())
        ext.batch_upload(h_al, alog)
        ext.batch_upload(h_dt, dtb)
        ext.batch_upload(h_nw, normw)
        ext.batch_fill(h_st, 0.0, nv * hk * hv)
        ext.batch_gdn_conv_io(h_q, h_bu, h_t, h_m, c)
        scale = 1.0 / math.sqrt(hk)
        ext.batch_gdn_recur_io(
            h_m, h_z, h_a, h_bv, h_al, h_dt, h_nw, h_st, h_po,
            c, key_dim, nv, hk, hv, nv // nk, scale, 1e-6, cfg.rms_norm_eps,
        )
        out = torch.empty(nv * hv)
        ext.batch_download(h_po, out)
        ext.batch_submit()
    finally:
        for handle in handles:
            try:
                ext.free_resident(handle)
            except RuntimeError:
                pass

    diff = (out - ref).abs()
    maxdiff = diff.max().item()
    assert torch.isfinite(out).all(), "GDN GPU output contains non-finite values"
    assert maxdiff < 1e-5, f"gdn maxdiff {maxdiff:.3e} mean {diff.mean().item():.3e}"
