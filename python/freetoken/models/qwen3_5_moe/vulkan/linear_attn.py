"""CPU reference GatedDeltaNet forward for Qwen3.5-MoE linear_attention layers.

Mirrors ``transformers`` ``Qwen3_5MoeGatedDeltaNet`` prefill path exactly
(no cache): fused qkv projection, causal depthwise conv1d + SiLU, qkv/z/a/b
split, per-v-head beta/gate factors, per-step gated-delta recurrence with
QK L2-norm (fp32, scale 1/sqrt(head_k_dim)), RMSNorm-gated output, out_proj.
Batch size one; sequence-major ``[T, ...]`` throughout.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-aligned L2 normalization over the last dim."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gated_delta_forward(
    hidden: torch.Tensor,
    w_qkv: torch.Tensor,
    w_z: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    conv_w: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    w_out: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    rms_eps: float,
) -> torch.Tensor:
    """One GatedDeltaNet block forward, fp32, ``[T, H] -> [T, H]``.

    Args:
        hidden: ``[T, H]`` input.
        w_qkv: ``[2*key_dim + value_dim, H]`` fused projection.
        w_z: ``[value_dim, H]`` gate projection.
        w_a/w_b: ``[num_v_heads, H]`` time-step projections.
        conv_w: depthwise conv weight ``[conv_dim, 1, 4]``.
        a_log/dt_bias: ``[num_v_heads]`` discretization params.
        norm_w: ``[head_v_dim]`` RMSNorm-gated weight (zero-centered).
        w_out: ``[H, value_dim]`` output projection.
    """
    T = hidden.shape[0]
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    mixed = F.linear(hidden, w_qkv).transpose(0, 1).unsqueeze(0)  # [1, C, T]
    conv = F.conv1d(mixed, conv_w, bias=None, padding=3, groups=mixed.shape[1])
    mixed = F.silu(conv[:, :, :T]).squeeze(0).transpose(0, 1)  # [T, C]
    query, key, value = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(T, num_k_heads, head_k_dim)
    key = key.reshape(T, num_k_heads, head_k_dim)
    value = value.reshape(T, num_v_heads, head_v_dim)

    z = F.linear(hidden, w_z).reshape(T, num_v_heads, head_v_dim)
    beta = torch.sigmoid(F.linear(hidden, w_b))  # [T, V]
    a = F.linear(hidden, w_a)  # [T, V]
    g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias)  # [T, V]

    repeat = num_v_heads // num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=1)
        key = key.repeat_interleave(repeat, dim=1)

    scale = 1.0 / math.sqrt(head_k_dim)
    state = torch.zeros(num_v_heads, head_k_dim, head_v_dim)
    outs = []
    for t in range(T):
        qt = l2norm(query[t]) * scale
        kt = l2norm(key[t])
        vt = value[t]
        gt = torch.exp(g[t])[:, None, None]
        bt = beta[t][:, None]
        state = state * gt
        kv_mem = (state * kt[:, :, None]).sum(dim=1)
        delta = (vt - kv_mem) * bt
        state = state + kt[:, :, None] * delta[:, None, :]
        outs.append((state * qt[:, :, None]).sum(dim=1))
    core = torch.stack(outs, dim=0)  # [T, V, D]

    return _norm_gated_out(core, z, norm_w, w_out, rms_eps, T, value_dim)


def _norm_gated_out(core: torch.Tensor, z: torch.Tensor, norm_w: torch.Tensor,
                      w_out: torch.Tensor, rms_eps: float, T: int, value_dim: int) -> torch.Tensor:
    flat_out = core.reshape(-1, core.shape[-1])
    flat_z = z.reshape(-1, z.shape[-1])
    var = flat_out.pow(2).mean(-1, keepdim=True)
    normed = flat_out * torch.rsqrt(var + rms_eps)
    # NOTE: unlike Qwen3_5MoeRMSNorm (zero-centered, applied as 1+w),
    # RMSNormGated uses a plain ones-init weight: norm(x) * w * silu(z).
    gated = normed * norm_w.float() * F.silu(flat_z.float())
    return F.linear(gated.reshape(T, value_dim), w_out)


def gated_delta_forward_chunk(
    hidden: torch.Tensor,
    w_qkv: torch.Tensor,
    w_z: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    conv_w: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    w_out: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    rms_eps: float,
    state_in: torch.Tensor | None = None,
    conv_prefix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunk-continuation GatedDeltaNet for 120k streaming prefill.

    ``hidden`` is ``[Tc, H]`` for this chunk. ``state_in`` is the recurrent
    ``[V, K, D]`` state from previous chunks (zeros for chunk 0).
    ``conv_prefix`` holds the last <=3 pre-conv rows (None for chunk 0).
    Returns ``(out [Tc, H], state_out, conv_tail [<=3, C])``.
    """
    Tc = hidden.shape[0]
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    pre = F.linear(hidden, w_qkv)  # [Tc, C]
    if conv_prefix is not None and conv_prefix.numel() > 0:
        ext = torch.cat([conv_prefix.to(pre.dtype), pre], dim=0)
        mt = ext.transpose(0, 1).unsqueeze(0)
        conv = F.conv1d(mt, conv_w, bias=None, padding=3, groups=mt.shape[1])
        mixed = F.silu(conv[:, :, -Tc:]).squeeze(0).transpose(0, 1)
    else:
        mt = pre.transpose(0, 1).unsqueeze(0)
        conv = F.conv1d(mt, conv_w, bias=None, padding=3, groups=mt.shape[1])
        mixed = F.silu(conv[:, :, :Tc]).squeeze(0).transpose(0, 1)
    if Tc >= 3:
        conv_tail = pre[-3:].detach().clone()
    elif conv_prefix is not None:
        conv_tail = torch.cat([conv_prefix.to(pre.dtype), pre], dim=0)[-3:].detach().clone()
    else:
        conv_tail = pre.detach().clone()

    query, key, value = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(Tc, num_k_heads, head_k_dim)
    key = key.reshape(Tc, num_k_heads, head_k_dim)
    value = value.reshape(Tc, num_v_heads, head_v_dim)
    z = F.linear(hidden, w_z).reshape(Tc, num_v_heads, head_v_dim)
    beta = torch.sigmoid(F.linear(hidden, w_b))
    a = F.linear(hidden, w_a)
    g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias)
    repeat = num_v_heads // num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=1)
        key = key.repeat_interleave(repeat, dim=1)
    scale = 1.0 / math.sqrt(head_k_dim)
    state = state_in.clone() if state_in is not None else torch.zeros(num_v_heads, head_k_dim, head_v_dim)
    outs = []
    for t in range(Tc):
        qt = l2norm(query[t]) * scale
        kt = l2norm(key[t])
        vt = value[t]
        gt = torch.exp(g[t])[:, None, None]
        bt = beta[t][:, None]
        state = state * gt
        kv = (state * kt[:, :, None]).sum(dim=1)
        delta = (vt - kv) * bt
        state = state + kt[:, :, None] * delta[:, None, :]
        outs.append((state * qt[:, :, None]).sum(dim=1))
    core = torch.stack(outs, dim=0)
    return _norm_gated_out(core, z, norm_w, w_out, rms_eps, Tc, value_dim), state, conv_tail
