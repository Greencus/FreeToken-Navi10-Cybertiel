"""CPU reference Qwen3.5-MoE decoder layer (linear_attention type).

Structure mirrors ``transformers`` ``Qwen3_5MoeDecoderLayer``: input RMSNorm,
token mixer, residual add, post-attention RMSNorm, MoE block, residual add.
RMSNorm weights are zero-centered (applied as ``1 + w``); see
``qwen35.attention.rms_norm_head``.
"""
from __future__ import annotations

import torch

from .attention import full_attn_forward, rms_norm_head
from .linear_attn import gated_delta_forward
from .loader import Qwen35MoeLayerWeights
from .moe import moe_forward


def _moe_tail(
    normed: torch.Tensor,
    weights: Qwen35MoeLayerWeights,
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Shared MoE block tail (post-attn norm input -> MoE output)."""
    gate = weights.gate_up.dequantized().float().reshape(num_experts, -1, hidden_size)  # type: ignore[union-attr]
    up = weights.up_proj.dequantized().float().reshape(num_experts, -1, hidden_size)  # type: ignore[union-attr]
    gate_up = torch.cat([gate, up], dim=1)
    down = weights.down.dequantized().float().reshape(num_experts, hidden_size, -1)  # type: ignore[union-attr]
    return moe_forward(
        normed,
        weights.router.float(),
        gate_up,
        down,
        weights.shared_gate.dequantized().float(),  # type: ignore[union-attr]
        weights.shared_up.dequantized().float(),  # type: ignore[union-attr]
        weights.shared_down.dequantized().float(),  # type: ignore[union-attr]
        weights.shared_expert_gate.dequantized().float(),  # type: ignore[union-attr]
        num_experts_per_tok,
    )


def decoder_layer_forward(
    hidden: torch.Tensor,
    weights: Qwen35MoeLayerWeights,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_experts: int,
    num_experts_per_tok: int,
    rms_eps: float,
    num_heads: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 256,
    rotary_dim: int = 64,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """One decoder-layer forward, fp32, ``[T, H] -> [T, H]``.

    Handles both ``linear_attention`` (GatedDeltaNet) and ``full_attention``
    layers; the norm/residual/MoE skeleton is shared. Full-attention layers
    need the text RoPE tables (``cos``/``sin`` ``[T, rotary_dim]``).
    """
    if weights.layer_type == "linear_attention":
        return _linear_layer_forward(
            hidden, weights, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            num_experts, num_experts_per_tok, rms_eps,
        )
    if weights.layer_type == "full_attention":
        if cos is None or sin is None:
            raise ValueError(f"layer {weights.layer_idx} full_attention needs cos/sin")
        return _full_layer_forward(
            hidden, weights, num_heads, num_kv_heads, head_dim, rotary_dim,
            cos, sin, num_experts, num_experts_per_tok, rms_eps,
        )
    raise ValueError(f"unsupported layer type {weights.layer_type}")


def _linear_layer_forward(
    hidden: torch.Tensor,
    weights: Qwen35MoeLayerWeights,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_experts: int,
    num_experts_per_tok: int,
    rms_eps: float,
) -> torch.Tensor:
    """GatedDeltaNet mixer + shared MoE tail."""
    for name in (
        "lin_qkv",
        "lin_z",
        "lin_a",
        "lin_b",
        "lin_out",
        "lin_conv",
        "lin_a_log",
        "lin_dt_bias",
        "lin_norm",
        "gate_up",
        "up_proj",
        "down",
        "shared_gate",
        "shared_up",
        "shared_down",
        "shared_expert_gate",
    ):
        if getattr(weights, name) is None:
            raise ValueError(f"layer {weights.layer_idx} missing {name}")
    assert weights.input_norm is not None and weights.post_attn_norm is not None
    assert weights.router is not None
    hidden_size = hidden.shape[-1]

    residual = hidden
    normed = rms_norm_head(hidden, weights.input_norm, rms_eps)
    mixed = gated_delta_forward(
        normed,
        weights.lin_qkv.dequantized().float(),  # type: ignore[union-attr]
        weights.lin_z.dequantized().float(),  # type: ignore[union-attr]
        weights.lin_a.dequantized().float(),  # type: ignore[union-attr]
        weights.lin_b.dequantized().float(),  # type: ignore[union-attr]
        weights.lin_conv.float(),
        weights.lin_a_log.float(),
        weights.lin_dt_bias.float(),
        weights.lin_norm.float(),
        weights.lin_out.dequantized().float(),  # type: ignore[union-attr]
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        rms_eps,
    )
    hidden = residual + mixed

    residual = hidden
    normed = rms_norm_head(hidden, weights.post_attn_norm, rms_eps)
    return residual + _moe_tail(normed, weights, hidden_size, num_experts, num_experts_per_tok)


def _full_layer_forward(
    hidden: torch.Tensor,
    weights: Qwen35MoeLayerWeights,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_experts: int,
    num_experts_per_tok: int,
    rms_eps: float,
) -> torch.Tensor:
    """Full-attention mixer + shared MoE tail."""
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        if getattr(weights, name) is None:
            raise ValueError(f"layer {weights.layer_idx} missing {name}")
    assert weights.input_norm is not None and weights.post_attn_norm is not None
    assert weights.router is not None
    hidden_size = hidden.shape[-1]

    residual = hidden
    normed = rms_norm_head(hidden, weights.input_norm, rms_eps)
    mixed = full_attn_forward(
        normed,
        weights.q_proj.dequantized().float(),  # type: ignore[union-attr]
        weights.k_proj.dequantized().float(),  # type: ignore[union-attr]
        weights.v_proj.dequantized().float(),  # type: ignore[union-attr]
        weights.o_proj.dequantized().float(),  # type: ignore[union-attr]
        weights.q_norm.float(),
        weights.k_norm.float(),
        cos, sin,
        num_heads, num_kv_heads, head_dim, rotary_dim, rms_eps,
    )
    hidden = residual + mixed

    residual = hidden
    normed = rms_norm_head(hidden, weights.post_attn_norm, rms_eps)
    return residual + _moe_tail(normed, weights, hidden_size, num_experts, num_experts_per_tok)
