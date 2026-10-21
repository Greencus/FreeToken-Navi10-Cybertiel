"""CPU reference full-attention forward for Qwen3.5-MoE.

Mirrors ``transformers`` ``Qwen3_5MoeAttention`` for text-only input:
fused query+gate projection, per-head Q/K RMSNorm, partial rotary
application (first ``head_dim * partial_rotary_factor`` dims), causal SDPA
with GQA repetition, sigmoid output gate, output projection.

M-RoPE note: for text-only positions the three M-RoPE rows (T/H/W) are
identical, so the interleave is a value-preserving no-op and the table
reduces to standard partial RoPE. ``compute_text_rope`` builds exactly that
table; the parity test asserts it equals the reference module's table.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_text_rope(
    positions: torch.Tensor,
    rope_theta: float,
    head_dim: int,
    partial_rotary_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial RoPE cos/sin tables, ``[T, rotary_dim]`` each.

    Matches ``Qwen3_5MoeTextRotaryEmbedding`` for text positions:
    ``dim = head_dim * factor`` rotary dims, ``inv_freq[i] =
    theta^(-2i/dim)``, ``emb = [freqs | freqs]``.
    """
    positions = positions.to(torch.float32)
    try:
        dim = int(head_dim * partial_rotary_factor)
        half = dim // 2
        index = torch.arange(0, dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (index / float(dim)))
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid RoPE geometry: {exc}") from exc
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    assert emb.shape[-1] == dim == half * 2
    return torch.cos(emb).contiguous(), torch.sin(emb).contiguous()


def apply_partial_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Rotate first ``rotary_dim`` dims (GPT-NeoX style halves)."""
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    # rotate_half portion of the reference: cat((-x2, x1))
    rotated = torch.cat((-x2, x1), dim=-1)
    y_rot = x_rot * cos + rotated * sin
    return torch.cat([y_rot, x_pass], dim=-1)


def rms_norm_head(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Qwen3.5 stores the norm weight zero-centered: y = norm(x) * (1 + w).
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * (1.0 + weight.float())).to(x.dtype)


def full_attn_forward(
    hidden: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rms_eps: float,
) -> torch.Tensor:
    """Full causal self-attention for one layer, batch size one.

    Args:
        hidden: ``[T, H]`` fp32 input.
        q_proj: fused ``[2*num_heads*head_dim, H]`` (query ++ gate).
        k_proj/v_proj: ``[num_kv_heads*head_dim, H]``.
        o_proj: ``[H, num_heads*head_dim]``.
        cos/sin: ``[T, rotary_dim]`` text RoPE tables.
    """
    T = hidden.shape[0]
    repeat = num_heads // num_kv_heads
    scaling = 1.0 / math.sqrt(head_dim)

    qg = F.linear(hidden, q_proj)
    query, gate = qg.view(T, num_heads, 2 * head_dim).chunk(2, dim=-1)
    key = F.linear(hidden, k_proj).view(T, num_kv_heads, head_dim)
    value = F.linear(hidden, v_proj).view(T, num_kv_heads, head_dim)
    query = rms_norm_head(query, q_norm, rms_eps)
    key = rms_norm_head(key, k_norm, rms_eps)

    # cos/sin [T, rotary] broadcast over heads.
    cos_b = cos.reshape(T, 1, rotary_dim)
    sin_b = sin.reshape(T, 1, rotary_dim)
    query = apply_partial_rotary(query, cos_b, sin_b, rotary_dim)
    key = apply_partial_rotary(key, cos_b, sin_b, rotary_dim)

    if repeat > 1:
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    # [H, T, D] attention.
    q = query.transpose(0, 1)
    k = key.transpose(0, 1)
    v = value.transpose(0, 1)
    scores = (q @ k.transpose(-1, -2)) * scaling
    causal = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    context = probs @ v
    context = context.transpose(0, 1).reshape(T, num_heads * head_dim)
    context = context * torch.sigmoid(gate.reshape(T, num_heads * head_dim))
    return F.linear(context, o_proj)


def full_attn_forward_chunk(
    hidden: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_chunk: torch.Tensor,
    sin_chunk: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rms_eps: float,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunk-continuation full attention for 120k streaming prefill.

    ``hidden`` is ``[Tc, H]``; ``cos/sin_chunk`` are ``[Tc, rotary_dim]`` rows
    for these positions. ``k/v_cache`` are ``[P, Kv, D]`` prefix caches
    (None for chunk 0). Returns ``(out [Tc, H], k_full, v_full)`` where the
    caches now include this chunk. Q/K get per-head RMSNorm + partial RoPE;
    GQA repetition, causal-across-cache SDPA, sigmoid output gate, o_proj.
    """
    import math as _math
    Tc = hidden.shape[0]
    repeat = num_heads // num_kv_heads
    scaling = 1.0 / _math.sqrt(head_dim)
    qg = torch.nn.functional.linear(hidden, q_proj)
    query, gate = qg.view(Tc, num_heads, 2 * head_dim).chunk(2, dim=-1)
    key_new = torch.nn.functional.linear(hidden, k_proj).view(Tc, num_kv_heads, head_dim)
    val_new = torch.nn.functional.linear(hidden, v_proj).view(Tc, num_kv_heads, head_dim)
    query = rms_norm_head(query, q_norm, rms_eps)
    key_new = rms_norm_head(key_new, k_norm, rms_eps)
    cos_b = cos_chunk.reshape(Tc, 1, rotary_dim)
    sin_b = sin_chunk.reshape(Tc, 1, rotary_dim)
    query = apply_partial_rotary(query, cos_b, sin_b, rotary_dim)
    key_new = apply_partial_rotary(key_new, cos_b, sin_b, rotary_dim)
    if k_cache is not None and v_cache is not None and k_cache.numel() > 0:
        key = torch.cat([k_cache, key_new], dim=0)
        value = torch.cat([v_cache, val_new], dim=0)
    else:
        key, value = key_new, val_new
    P = key.shape[0]
    if repeat > 1:
        key_e = key.repeat_interleave(repeat, dim=1)
        val_e = value.repeat_interleave(repeat, dim=1)
    else:
        key_e, val_e = key, value
    # Queries attend to cache prefix: causal across [P-Tc prefix + Tc chunk].
    q = query.transpose(0, 1)  # [H, Tc, D]
    k = key_e.transpose(0, 1)  # [H, P, D]
    v = val_e.transpose(0, 1)
    scores = (q @ k.transpose(-1, -2)) * scaling  # [H, Tc, P]
    # Mask future positions within chunk: query i (global P-Tc+i) sees keys <= P-Tc+i.
    row = torch.arange(Tc) + (P - Tc)
    col = torch.arange(P)
    allowed = col.unsqueeze(0) <= row.unsqueeze(1)  # [Tc, P]
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    context = (probs @ v).transpose(0, 1).reshape(Tc, num_heads * head_dim)
    context = context * torch.sigmoid(gate.reshape(Tc, num_heads * head_dim))
    return torch.nn.functional.linear(context, o_proj), key.detach(), value.detach()
