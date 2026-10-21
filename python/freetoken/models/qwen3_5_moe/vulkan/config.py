"""Qwen3.5-MoE (Ornith 1.5 family) configuration.

Parses the text-level config of checkpoints like
``peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e``: a 40-layer hybrid
(30x GatedDeltaNet linear attention + 10x full attention) MoE with 256
experts, top-8 routing, a shared expert, and mixed-bit TorchAO-style affine
quantization (default int4/group-64 with per-module 5/6/8-bit overrides).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Qwen35MoeConfig:
    model_type: str = "qwen3_5_moe"
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    layer_types: List[str] = field(default_factory=list)
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    attn_output_gate: bool = True
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    # linear (GatedDeltaNet) attention geometry
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_conv_kernel_dim: int = 4
    # (bits, group_size) per quantized module stem; default when absent
    default_bits: int = 4
    default_group_size: int = 64
    quant_overrides: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def quant_for(self, module_stem: str) -> Tuple[int, int]:
        """Longest-prefix match of a weight stem against quantization_config.

        Stems look like ``language_model.model.layers.0.mlp.switch_mlp``;
        the config keys name quantized *modules*, so ``...gate_proj.weight``
        must resolve to its module entry.
        """
        stem = module_stem.removesuffix(".weight")
        best: Tuple[int, int] | None = None
        best_len = -1
        for key, value in self.quant_overrides.items():
            if stem == key or stem.startswith(key + "."):
                if len(key) > best_len:
                    best, best_len = value, len(key)
        if best is not None:
            return best
        return (self.default_bits, self.default_group_size)

    def validate(self) -> "Qwen35MoeConfig":
        assert self.hidden_size > 0 and self.num_hidden_layers > 0
        assert len(self.layer_types) == self.num_hidden_layers, (
            f"layer_types has {len(self.layer_types)} entries for "
            f"{self.num_hidden_layers} layers"
        )
        assert set(self.layer_types) <= {"linear_attention", "full_attention"}
        assert self.num_experts > 0 and self.num_experts_per_tok > 0
        return self


def load_qwen35_config(path: str | pathlib.Path) -> Qwen35MoeConfig:
    source = pathlib.Path(path)
    if source.is_dir():
        source = source / "config.json"
    try:
        data = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read qwen35 config at {source}: {exc}") from exc
    if data.get("model_type") != "qwen3_5_moe":
        raise ValueError(f"expected qwen3_5_moe model_type, got {data.get('model_type')!r}")
    try:
        text = data["text_config"]
        quant = data.get("quantization_config", data.get("quantization", {}))
        overrides: Dict[str, Tuple[int, int]] = {}
        for key, value in quant.items():
            if isinstance(value, dict) and "bits" in value:
                overrides[key] = (int(value["bits"]), int(value["group_size"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid qwen35 quantization config at {source}: {exc}") from exc
    rope = text.get("rope_parameters", {})
    try:
        return Qwen35MoeConfig(
        hidden_size=int(text["hidden_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        layer_types=list(text["layer_types"]),
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        head_dim=int(text.get("head_dim", 256)),
        partial_rotary_factor=float(rope.get("partial_rotary_factor", 0.25)),
        rope_theta=float(rope.get("rope_theta", 10_000_000.0)),
        mrope_section=list(rope.get("mrope_section", [11, 11, 10])),
        attn_output_gate=bool(text.get("attn_output_gate", True)),
        num_experts=int(text["num_experts"]),
        num_experts_per_tok=int(text["num_experts_per_tok"]),
        moe_intermediate_size=int(text["moe_intermediate_size"]),
        shared_expert_intermediate_size=int(text["shared_expert_intermediate_size"]),
        vocab_size=int(text["vocab_size"]),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        hidden_act=str(text.get("hidden_act", "silu")),
        linear_key_head_dim=int(text["linear_key_head_dim"]),
        linear_value_head_dim=int(text["linear_value_head_dim"]),
        linear_num_key_heads=int(text["linear_num_key_heads"]),
        linear_num_value_heads=int(text["linear_num_value_heads"]),
        linear_conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
        default_bits=int(quant.get("bits", 4)),
        default_group_size=int(quant.get("group_size", 64)),
            quant_overrides=overrides,
        ).validate()
    except (KeyError, ValueError, TypeError, AssertionError) as exc:
        raise ValueError(f"invalid qwen35 model config at {source}: {exc}") from exc
