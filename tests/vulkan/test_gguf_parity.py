"""CyberTiel GGUF layer parity + greedy stability gates (Vulkan decode path).

Weight-source-neutral numerics gates for ResidentGGUF, opt-in via
FREETOKEN_TEST_MODEL_GGUF (skips otherwise):

1. GDN layer (0) and full-attention layer (3) sequential steps vs a host
   reference that uses the SAME BF16-rounded weights the decoder pins
   (``_dense`` reads) with fp32 math. The only differences are the GPU
   linear kernels (bf16-weight GEMVs); everything else - conv history,
   recurrence, rotary, softmax, gating - is the identical torch code, so
   maxdiff isolates kernel-vs-torch matmul error. Gate 1e-5 (fp32 libm /
   accumulation-order ulp over K=2048 GEMVs), the same bar
   test_gdn_kernel.py applies to the raw GDN shaders.
2. Greedy generate() twice from a fixed prompt -> identical token ids,
   all finite (decode_step already raises on non-finite logits).

For the GDN *shaders* (batch_gdn_conv/recur) vs host reference, see
test_gdn_kernel.py (also runs on the GGUF through resident_model).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

torch.set_num_threads(8)
os.environ.setdefault("FTV_NVFP4_GPU", "1")
os.environ.setdefault("FTV_EXP_CACHE_MB", "2048")

PREFIX = "language_model.model.layers"
PROMPT = [151643, 720, 37893, 2872, 48574]  # same ids as test_batched_fork2


def _snapshot(model, layers):
    snap = {
        "gdn_state": {i: model.gdn_state[i].clone() for i in layers
                      if i in model.gdn_state},
        "conv_buf": {i: model.conv_buf[i].clone() for i in layers
                     if i in model.conv_buf},
        "k_cache": {i: model.k_cache[i].clone() for i in layers
                    if i in model.k_cache},
        "v_cache": {i: model.v_cache[i].clone() for i in layers
                    if i in model.v_cache},
    }
    return snap


def _restore(model, snap):
    for i, s in snap["gdn_state"].items():
        model.gdn_state[i].copy_(s)
    for i, c in snap["conv_buf"].items():
        model.conv_buf[i].copy_(c)
    for i, k in snap["k_cache"].items():
        model.k_cache[i].copy_(k)
    for i, v in snap["v_cache"].items():
        model.v_cache[i].copy_(v)


def _gdn_host_ref(model, i, x):
    """``_gdn_step`` with the 5 GPU affines replaced by torch F.linear on
    the exact pinned (BF16-rounded) weights. Mutates the recurrent state
    the same way; call on a restored snapshot to compare against the real
    step run from the same state."""
    from freetoken.models.qwen3_5_moe.vulkan.linear_attn import l2norm

    cfg = model.config
    base = f"{PREFIX}.{i}"

    def lin(stem):
        w = model._dense(f"{base}.{stem}.weight")
        return F.linear(x, w)

    qkv = lin("linear_attn.in_proj_qkv")
    z = lin("linear_attn.in_proj_z")
    a = lin("linear_attn.in_proj_a")
    b = lin("linear_attn.in_proj_b")

    conv_w = model._dense(f"{base}.linear_attn.conv1d.weight")
    taps = conv_w[:, 0, :].float() if conv_w.dim() == 3 else conv_w.float()
    buf = model.conv_buf[i]
    mixed = (taps[:, :3] * buf).sum(-1) + taps[:, 3] * qkv
    buf[:, 0] = buf[:, 1]
    buf[:, 1] = buf[:, 2]
    buf[:, 2] = qkv
    mixed = F.silu(mixed)

    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    key_dim = nk * hk
    query, key, value = torch.split(mixed, [key_dim, key_dim, nv * hv], dim=-1)
    query = query.reshape(nk, hk).repeat_interleave(nv // nk, dim=0)
    key = key.reshape(nk, hk).repeat_interleave(nv // nk, dim=0)
    value = value.reshape(nv, hv)

    a_log = model._dense(f"{base}.linear_attn.A_log")
    dt_bias = model._dense(f"{base}.linear_attn.dt_bias")
    beta = torch.sigmoid(b)
    g = -torch.exp(a_log) * F.softplus(a + dt_bias)

    scale = 1.0 / math.sqrt(hk)
    qt = l2norm(query) * scale
    kt = l2norm(key)
    state = model.gdn_state[i]
    state.mul_(torch.exp(g)[:, None, None])
    kv_mem = (state * kt[:, :, None]).sum(dim=1)
    delta = (value - kv_mem) * beta[:, None]
    state.add_(kt[:, :, None] * delta[:, None, :])
    core = (state * qt[:, :, None]).sum(dim=1)

    norm_w = model._dense(f"{base}.linear_attn.norm.weight")
    var = core.pow(2).mean(-1, keepdim=True)
    normed = core * torch.rsqrt(var + cfg.rms_norm_eps)
    gated = (normed * norm_w * F.silu(z.reshape(nv, hv))).reshape(nv * hv)
    return F.linear(gated, model._dense(f"{base}.linear_attn.out_proj.weight"))


def _full_attn_host_ref(model, i, x, pos):
    """``_full_attn_step`` with the 4 GPU affines as torch F.linear."""
    from freetoken.models.qwen3_5_moe.vulkan.attention import (
        apply_partial_rotary,
        rms_norm_head,
    )

    cfg = model.config
    base = f"{PREFIX}.{i}"
    hq, hkv, hd = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    )

    def lin(stem):
        return F.linear(x, model._dense(f"{base}.{stem}.weight"))

    qg = lin("self_attn.q_proj")
    query, gate = qg.view(hq, 2 * hd).chunk(2, dim=-1)
    key = lin("self_attn.k_proj").view(hkv, hd)
    value = lin("self_attn.v_proj").view(hkv, hd)

    q_norm = model._dense(f"{base}.self_attn.q_norm.weight")
    k_norm = model._dense(f"{base}.self_attn.k_norm.weight")
    query = rms_norm_head(query, q_norm, cfg.rms_norm_eps).float()
    key = rms_norm_head(key, k_norm, cfg.rms_norm_eps).float()

    cos = model.rope_cos[pos].unsqueeze(0)
    sin = model.rope_sin[pos].unsqueeze(0)
    query = apply_partial_rotary(query, cos, sin, model.rotary_dim)
    key = apply_partial_rotary(key, cos, sin, model.rotary_dim)

    model.k_cache[i][pos].copy_(key)
    model.v_cache[i][pos].copy_(value)
    keys = model.k_cache[i][: pos + 1]
    values = model.v_cache[i][: pos + 1]
    repeat = hq // hkv
    if repeat > 1:
        keys = keys.repeat_interleave(repeat, dim=1)
        values = values.repeat_interleave(repeat, dim=1)
    q = query.unsqueeze(1)
    k = keys.transpose(0, 1)
    v = values.transpose(0, 1)
    scores = (q @ k.transpose(-1, -2)).squeeze(1) / math.sqrt(hd)
    probs = torch.softmax(scores.float(), dim=-1).unsqueeze(1)
    context = (probs @ v).transpose(0, 1).reshape(hq * hd)
    context = context * torch.sigmoid(gate.reshape(hq * hd))
    return F.linear(context, model._dense(f"{base}.self_attn.o_proj.weight"))


def test_gdn_layer_step_matches_torch_host(gguf_model):
    model = gguf_model
    model.reset()
    i = 0
    torch.manual_seed(11)
    x = torch.randn(model.config.hidden_size)
    snap = _snapshot(model, {i})
    out_step = model._gdn_step(i, x).detach().float().clone()
    _restore(model, snap)
    out_ref = _gdn_host_ref(model, i, x).detach().float().clone()
    d = (out_step - out_ref).abs()
    maxdiff = d.max().item()
    print(f"gdn layer {i}: maxdiff {maxdiff:.3e} mean {d.mean().item():.3e}")
    assert torch.isfinite(out_step).all()
    assert maxdiff <= 1e-5, f"gdn step vs host maxdiff {maxdiff:.3e}"


def test_full_attn_layer_step_matches_torch_host(gguf_model):
    model = gguf_model
    model.reset()
    i = 3  # first full-attention layer
    pos = 17  # non-trivial cache prefix
    torch.manual_seed(13)
    # Seed a short history so the attention mask/prefix path is exercised.
    x_hist = torch.randn(pos, model.config.hidden_size)
    for p in range(pos):
        model._full_attn_step(i, x_hist[p], p)
    snap = _snapshot(model, {i})
    x = torch.randn(model.config.hidden_size)
    out_step = model._full_attn_step(i, x, pos).detach().float().clone()
    _restore(model, snap)
    out_ref = _full_attn_host_ref(model, i, x, pos).detach().float().clone()
    d = (out_step - out_ref).abs()
    maxdiff = d.max().item()
    print(f"full-attn layer {i} @pos {pos}: maxdiff {maxdiff:.3e} "
          f"mean {d.mean().item():.3e}")
    assert torch.isfinite(out_step).all()
    assert maxdiff <= 1e-5, f"full-attn step vs host maxdiff {maxdiff:.3e}"


def test_greedy_generate_stable_and_finite(gguf_model):
    runs = []
    for _ in range(2):
        out = gguf_model.generate(
            PROMPT, max_new=12, temperature=0.0, prefill_chunk=64, seed=1234
        )
        assert out, "generated no tokens"
        runs.append(out)
    assert runs[0] == runs[1], (
        f"greedy runs differ:\n  run1 {runs[0]}\n  run2 {runs[1]}"
    )
    print(f"greedy 12tok ids: {runs[0]}")
