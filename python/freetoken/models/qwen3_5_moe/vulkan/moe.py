"""CPU reference MoE forward for Qwen3.5-MoE.

Mirrors ``transformers`` ``Qwen3_5MoeSparseMoeBlock`` exactly (softmax router
in float32, top-k renormalize, fused gate/up SwiGLU experts, sigmoid-gated
shared expert) so weight loads and later Vulkan kernels can be checked for
parity token-by-token. Batch size one; sequence-major.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def moe_router(
    hidden: torch.Tensor, router_weight: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (top_k_weights [T, K], top_k_index [T, K])."""
    # Match the reference: logits are computed in the router's stored dtype
    # (BF16 in the checkpoint), with the softmax itself in float32.
    logits = F.linear(hidden.to(router_weight.dtype), router_weight).float()
    probs = torch.softmax(logits, dim=-1)
    values, index = torch.topk(probs, top_k, dim=-1)
    values = values / values.sum(dim=-1, keepdim=True)
    return values.to(hidden.dtype), index


def moe_forward(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    shared_expert_gate: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Full MoE block forward in the checkpoint dtype (torch.float32 here).

    Args:
        hidden: ``[T, H]`` input.
        router_weight: ``[E, H]`` BF16→FP32 router.
        gate_up: fused ``[E, 2*I, H]`` (concat dequantized gate/up halves).
        down: ``[E, H, I]``.
        shared_gate/shared_up: ``[I_s, H]``; shared_down: ``[H, I_s]``.
        shared_expert_gate: ``[1, H]``.
    """
    weights, index = moe_router(hidden, router_weight, top_k)
    out = torch.zeros_like(hidden)
    seen = torch.unique(index)
    for expert in seen.tolist():
        positions = (index == expert).nonzero(as_tuple=False)
        tokens = positions[:, 0]
        slots = positions[:, 1]
        gate, up = F.linear(hidden[tokens], gate_up[expert]).chunk(2, dim=-1)
        hidden_state = F.silu(gate) * up
        hidden_state = F.linear(hidden_state, down[expert])
        hidden_state = hidden_state * weights[tokens, slots, None]
        out.index_add_(0, tokens, hidden_state.to(out.dtype))
    shared = F.silu(F.linear(hidden, shared_gate)) * F.linear(hidden, shared_up)
    shared = F.linear(shared, shared_down)
    out = out + torch.sigmoid(F.linear(hidden, shared_expert_gate)) * shared
    return out
