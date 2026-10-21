"""Port of the fork's test_qwen35_moe_parity.py (CPU-only, no Vulkan).

Loads layer-0 MoE weights (TorchAO-affine) from a Cyber-Tiel checkpoint,
dequantizes, and checks the package's ``moe_forward`` against
``transformers`` ``Qwen3_5MoeSparseMoeBlock`` on a random input. Gates the
unpack/dequant math and the MoE wiring before any Vulkan work (fork gate:
maxdiff < 1e-4, all finite).

Opt-in via FREETOKEN_TEST_MODEL_QWEN35 (the TorchAO-affine checkpoint
directory); skips otherwise. This dequantizes ~256x512x2048 fp32 matrices,
so it also needs several GB of RAM.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.models.qwen3_5_moe.vulkan import (
    AffineQuantMatrix,
    ShardedSafetensors,
    load_qwen35_config,
    load_qwen35_moe_layer,
    moe_forward,
)
from freetoken.models.qwen3_5_moe.vulkan.loader import _load_quant_matrix

PREFIX = "language_model.model.layers.0.mlp.switch_mlp"


def test_moe_parity_vs_transformers(qwen35_model_dir):
    torch.set_num_threads(8)
    model_dir = qwen35_model_dir
    config = load_qwen35_config(model_dir)
    tensors = ShardedSafetensors(model_dir)
    layer = load_qwen35_moe_layer(tensors, config, 0)
    assert layer.gate_up is not None and layer.down is not None
    assert layer.shared_gate is not None and layer.shared_up is not None
    assert layer.shared_down is not None and layer.shared_expert_gate is not None

    e, inter, hidden = (
        config.num_experts,
        config.moe_intermediate_size,
        config.hidden_size,
    )

    def expert_half(name: str) -> torch.Tensor:
        stem = f"{PREFIX}.{name}"
        bits, group = config.quant_for(stem + ".weight")
        matrix: AffineQuantMatrix = _load_quant_matrix(
            tensors, stem, bits, group, (e,)
        )
        return matrix.dequantized().reshape(e, inter, hidden)

    # Checkpoint keeps gate/up split; reference fuses [E, 2*I, H].
    gate_fused = torch.cat(
        [layer.gate_up.dequantized().reshape(e, inter, hidden), expert_half("up_proj")],
        dim=1,
    )
    down = layer.down.dequantized().reshape(e, hidden, inter)
    shared_gate = layer.shared_gate.dequantized()
    shared_up = layer.shared_up.dequantized()
    shared_down = layer.shared_down.dequantized()
    shared_expert_gate = layer.shared_expert_gate.dequantized()

    namespace = SimpleNamespace(
        num_experts=e,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=config.shared_expert_intermediate_size,
        hidden_act=config.hidden_act,
        num_experts_per_tok=config.num_experts_per_tok,
        _experts_implementation="eager",
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )

    block: torch.nn.Module = Qwen3_5MoeSparseMoeBlock(namespace)
    state: dict = {
        "experts.gate_up_proj": gate_fused,
        "experts.down_proj": down,
        "shared_expert.gate_proj.weight": shared_gate,
        "shared_expert.up_proj.weight": shared_up,
        "shared_expert.down_proj.weight": shared_down,
        "shared_expert_gate.weight": shared_expert_gate,
        "gate.weight": layer.router,
    }
    # strict=True: the dict covers all 7 block params; a reference rename
    # must fail loudly, never silently fall back to random-init weights.
    block.load_state_dict(state, strict=True)
    generator = torch.Generator().manual_seed(7)
    hidden_t = torch.randn((4, hidden), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        reference = block(hidden_t.unsqueeze(0))
        if isinstance(reference, torch.Tensor):
            reference = reference[0]
    if isinstance(reference, tuple):
        reference = reference[0]
    with torch.no_grad():
        actual = moe_forward(
            hidden_t,
            layer.router.float(),
            gate_fused.float(),
            down.float(),
            shared_gate.float(),
            shared_up.float(),
            shared_down.float(),
            shared_expert_gate.float(),
            config.num_experts_per_tok,
        )
    ref_tensor = reference.float()

    difference = (actual - ref_tensor).abs()
    maxdiff = difference.max().item()
    print(
        f"moe parity maxdiff={maxdiff:.9g} "
        f"mean={difference.mean().item():.9g} "
        f"finite={torch.isfinite(actual).all().item()}"
    )
    assert torch.isfinite(actual).all()
    assert maxdiff < 1e-4, f"moe parity maxdiff {maxdiff:.9g}"
