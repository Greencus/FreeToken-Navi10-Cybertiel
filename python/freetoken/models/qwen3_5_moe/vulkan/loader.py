"""Safetensors loader for TorchAO-affine-quantized Qwen3.5-MoE checkpoints.

Weight layout per quantized module (TorchAO groupwise affine, as produced by
the ``oQ4e`` quantizer): three tensors ``<stem>.weight`` (U32 bit-packed
unsigned integers, little-endian flat bit-stream, ``ceil(cols*bits/32)``
words per row), ``<stem>.scales`` (BF16, one scale per ``group_size``
columns) and ``<stem>.biases`` (BF16, same shape as scales, holding
``zero_point * scale``). Dequantized value::

    w = q * scale + bias

with ``q`` the unpacked unsigned integer. Bits/group-size come from
``quantization_config`` per module (default int4/group-64; this checkpoint
mixes 4/5/6/8-bit with group 64/128).

Unquantized tensors (router gate, norms, conv1d, A_log/dt_bias) load as BF16.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open

from .config import Qwen35MoeConfig, load_qwen35_config

PREFIX = "language_model"


class ShardedSafetensors:
    """Lazy view over a sharded safetensors checkpoint (weight_map index).

    Ported verbatim from the fork's qwen3/loader.py so this package has no
    dependency on fork-local modules.
    """

    def __init__(self, model_dir: str | pathlib.Path):
        self.model_dir = pathlib.Path(model_dir).resolve()
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        self.weight_map: Dict[str, str] = index["weight_map"]
        self._handles: Dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                self.model_dir / shard, framework="pt", device="cpu"
            )
        return self._handles[shard]

    def contains(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        return self._handle(shard).get_tensor(name)


def unpack_affine_u32(packed: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    """Unpack a flat LE bit-stream of unsigned integers.

    Args:
        packed: U32 tensor ``[..., words]`` with ``words == ceil(cols*bits/32)``.
        bits: bits per value (4, 5, 6 or 8 in this checkpoint family).
        cols: logical values per row.

    Returns:
        int64 tensor ``[..., cols]`` of unsigned values in ``[0, 2**bits)``.
    """
    if packed.dtype != torch.int32 and packed.dtype != torch.uint32:
        raise TypeError(f"expected packed U32 weights, got {packed.dtype}")
    if not 1 <= bits <= 8:
        raise ValueError(f"unsupported affine bit-width {bits}")
    words = packed.shape[-1]
    expect = (cols * bits + 31) // 32
    if words != expect:
        raise ValueError(
            f"packed width {words} != ceil({cols}*{bits}/32)={expect}"
        )
    lead = packed.shape[:-1]
    flat = packed.reshape(-1, words).to(torch.int64)
    out = torch.empty(flat.shape[0], cols, dtype=torch.int64)
    mask = (1 << bits) - 1
    # Flat bit offset of value j is j*bits; word w=j*bits//32, shift=j*bits%32.
    # Values may straddle a word boundary (5- and 6-bit), so accumulate the
    # low part from word w and the spillover from word w+1 (zero-padded).
    padded = torch.zeros(
        flat.shape[0], words + 1, dtype=torch.int64, device=flat.device
    )
    padded[:, :words] = flat
    for j in range(cols):
        bit = j * bits
        word = bit // 32
        shift = bit % 32
        value = (padded[:, word] >> shift) & mask
        spill = shift + bits - 32
        if spill > 0:
            value = value | ((padded[:, word + 1] & ((1 << spill) - 1)) << (32 - shift))
        out[:, j] = value
    return out.reshape(*lead, cols)


def dequantize_affine(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Dequantize one affine-quantized matrix to FP32.

    Shapes: ``packed [rows, words]``, ``scales/biases [rows, groups]`` with
    ``groups == cols/group_size`` where ``cols = words*32//bits``.
    """
    if scales.shape != biases.shape:
        raise ValueError(f"scales {tuple(scales.shape)} != biases {tuple(biases.shape)}")
    rows, words = packed.shape[-2], packed.shape[-1]
    cols = words * 32 // bits
    groups = cols // group_size
    if (cols * bits) % 32 != 0 or cols % group_size != 0:
        raise ValueError(
            f"incompatible geometry words={words} bits={bits} group={group_size}"
        )
    if tuple(scales.shape[-2:]) != (rows, groups):
        raise ValueError(
            f"scales {tuple(scales.shape)} incompatible with [{rows}, {cols}] "
            f"at group {group_size}"
        )
    lead = packed.shape[:-2]
    q = unpack_affine_u32(packed.reshape(-1, words), bits, cols).float()
    q = q.reshape(*lead, rows, cols)
    scale = scales.float().repeat_interleave(group_size, dim=-1)
    bias = biases.float().repeat_interleave(group_size, dim=-1)
    # NOTE: the stored bias is pre-negated (bias = -zero_point * scale),
    # so dequantization ADDS it. Verified by parity: w = q*s - b gives
    # mean-shifted weights (+|b|) and 31.0 maxabs vs the reference MoE,
    # while w = q*s + b is zero-centered and drops to 0.67 (residual TBD).
    return (q * scale + bias).contiguous()


def _to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise TypeError(f"unexpected dense weight dtype {tensor.dtype}")
    return tensor.float().contiguous()


@dataclass
class AffineQuantMatrix:
    """One quantized matrix: packed payload plus BF16 scale/bias grids."""

    bits: int
    group_size: int
    packed: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    rows: int = 0
    cols: int = 0

    def dequantized(self) -> torch.Tensor:
        return dequantize_affine(
            self.packed, self.scales, self.biases, self.bits, self.group_size
        )


@dataclass
class Qwen35MoeLayerWeights:
    layer_idx: int
    layer_type: str
    input_norm: torch.Tensor = field(default=None)  # type: ignore[assignment]
    # linear_attention (GatedDeltaNet) projections; None on full_attention layers
    lin_qkv: AffineQuantMatrix | None = None
    lin_z: AffineQuantMatrix | None = None
    lin_a: AffineQuantMatrix | None = None
    lin_b: AffineQuantMatrix | None = None
    lin_out: AffineQuantMatrix | None = None
    lin_conv: torch.Tensor = field(default=None)  # type: ignore[assignment]
    lin_a_log: torch.Tensor = field(default=None)  # type: ignore[assignment]
    lin_dt_bias: torch.Tensor = field(default=None)  # type: ignore[assignment]
    lin_norm: torch.Tensor = field(default=None)  # type: ignore[assignment]
    # full_attention projections; None on linear_attention layers.
    # q_proj is fused query+gate: rows = 2 * num_heads * head_dim; the
    # gate half is the second chunk (there is no separate gate tensor).
    q_proj: AffineQuantMatrix | None = None
    k_proj: AffineQuantMatrix | None = None
    v_proj: AffineQuantMatrix | None = None
    o_proj: AffineQuantMatrix | None = None
    q_norm: torch.Tensor = field(default=None)  # type: ignore[assignment]
    k_norm: torch.Tensor = field(default=None)  # type: ignore[assignment]
    # MoE
    post_attn_norm: torch.Tensor = field(default=None)  # type: ignore[assignment]
    router: torch.Tensor = field(default=None)  # type: ignore[assignment]
    gate_up: AffineQuantMatrix | None = None  # gate half [E, inter, hidden]
    up_proj: AffineQuantMatrix | None = None  # up half [E, inter, hidden]
    down: AffineQuantMatrix | None = None
    shared_gate: AffineQuantMatrix | None = None
    shared_up: AffineQuantMatrix | None = None
    shared_down: AffineQuantMatrix | None = None
    shared_expert_gate: AffineQuantMatrix | None = None


@dataclass
class Qwen35MoeModelWeights:
    config: Qwen35MoeConfig
    embed_tokens: AffineQuantMatrix
    final_norm: torch.Tensor
    lm_head: AffineQuantMatrix
    layers: List[Qwen35MoeLayerWeights] = field(default_factory=list)


def _load_quant_matrix(
    tensors: ShardedSafetensors,
    stem: str,
    bits: int,
    group_size: int,
    extra_dims: Tuple[int, ...] = (),
) -> AffineQuantMatrix:
    """Load ``<stem>.weight/.scales/.biases``; supports leading expert dims.

    Expert tensors are 3-D (``[E, rows, packed]``); dense matrices are 2-D.
    The payload is reshaped to 2-D ``[E*rows, packed]`` with scales/biases
    flattened the same way; ``rows``/``cols`` record the logical 2-D shape.
    """
    try:
        packed = tensors.get(stem + ".weight")
        scales = tensors.get(stem + ".scales")
        biases = tensors.get(stem + ".biases")
    except (KeyError, RuntimeError) as exc:
        raise ValueError(f"missing affine triplet for {stem}: {exc}") from exc
    if packed.dtype not in {torch.int32, torch.uint32}:
        raise TypeError(f"{stem}.weight: expected packed U32, got {packed.dtype}")
    lead = packed.shape[:-2] if packed.dim() > 2 else ()
    if tuple(lead) != tuple(extra_dims):
        raise ValueError(f"{stem}.weight leading dims {tuple(lead)} != {tuple(extra_dims)}")
    rows = packed.shape[-2]
    words = packed.shape[-1]
    cols = words * 32 // bits
    flat_n = 1
    for dim in extra_dims:
        flat_n *= dim
    return AffineQuantMatrix(
        bits=bits,
        group_size=group_size,
        packed=packed.reshape(flat_n * rows, words).contiguous(),
        scales=scales.reshape(flat_n * rows, -1).contiguous(),
        biases=biases.reshape(flat_n * rows, -1).contiguous(),
        rows=rows,
        cols=cols,
    )


def load_qwen35_moe_layer(
    tensors: ShardedSafetensors,
    config: Qwen35MoeConfig,
    layer_idx: int,
) -> Qwen35MoeLayerWeights:
    prefix = f"{PREFIX}.model.layers.{layer_idx}"
    layer_type = config.layer_types[layer_idx]

    def dense(suffix: str) -> torch.Tensor:
        try:
            return _to_fp32(tensors.get(f"{prefix}.{suffix}"))
        except (KeyError, RuntimeError) as exc:
            raise ValueError(f"missing dense tensor {prefix}.{suffix}: {exc}") from exc

    def quant(suffix: str, extra: Tuple[int, ...] = ()) -> AffineQuantMatrix:
        stem = f"{prefix}.{suffix}"
        bits, group = config.quant_for(stem)
        return _load_quant_matrix(tensors, stem, bits, group, extra)

    weights = Qwen35MoeLayerWeights(
        layer_idx=layer_idx,
        layer_type=layer_type,
        input_norm=dense("input_layernorm.weight"),
        post_attn_norm=dense("post_attention_layernorm.weight"),
        router=dense("mlp.gate.weight"),
    )
    experts = (config.num_experts,)
    if layer_type == "linear_attention":
        weights.lin_qkv = quant("linear_attn.in_proj_qkv")
        weights.lin_z = quant("linear_attn.in_proj_z")
        weights.lin_a = quant("linear_attn.in_proj_a")
        weights.lin_b = quant("linear_attn.in_proj_b")
        weights.lin_out = quant("linear_attn.out_proj")
        # Checkpoint stores the depthwise conv as [C, K, 1]; nn.Conv1d (and
        # the FLA kernel's squeezed [C, K] view) needs [C, 1, K].
        weights.lin_conv = dense("linear_attn.conv1d.weight").permute(0, 2, 1).contiguous()
        weights.lin_a_log = dense("linear_attn.A_log")
        weights.lin_dt_bias = dense("linear_attn.dt_bias")
        weights.lin_norm = dense("linear_attn.norm.weight")
    else:
        weights.q_proj = quant("self_attn.q_proj")
        weights.k_proj = quant("self_attn.k_proj")
        weights.v_proj = quant("self_attn.v_proj")
        weights.o_proj = quant("self_attn.o_proj")
        weights.q_norm = dense("self_attn.q_norm.weight")
        weights.k_norm = dense("self_attn.k_norm.weight")
    weights.gate_up = quant("mlp.switch_mlp.gate_proj", experts)
    # Reference stores fused [E, 2*inter, hidden]; checkpoint keeps gate/up
    # split, so load both halves and concatenate on demand for parity.
    weights.up_proj = quant("mlp.switch_mlp.up_proj", experts)
    weights.down = quant("mlp.switch_mlp.down_proj", experts)
    weights.shared_gate = quant("mlp.shared_expert.gate_proj")
    weights.shared_up = quant("mlp.shared_expert.up_proj")
    weights.shared_down = quant("mlp.shared_expert.down_proj")
    weights.shared_expert_gate = quant("mlp.shared_expert_gate")
    return weights


def load_qwen35_model(
    model_dir: str | pathlib.Path,
    layers: List[int] | None = None,
) -> Qwen35MoeModelWeights:
    model_dir = pathlib.Path(model_dir)
    config = load_qwen35_config(model_dir)
    try:
        tensors = ShardedSafetensors(model_dir)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot open sharded safetensors at {model_dir}: {exc}") from exc

    def quant_root(suffix: str) -> AffineQuantMatrix:
        stem = f"{PREFIX}.{suffix}"
        bits, group = config.quant_for(stem)
        return _load_quant_matrix(tensors, stem, bits, group)

    def dense_root(suffix: str) -> torch.Tensor:
        return _to_fp32(tensors.get(f"{PREFIX}.{suffix}"))

    if layers is None:
        layers = list(range(config.num_hidden_layers))
    return Qwen35MoeModelWeights(
        config=config,
        embed_tokens=quant_root("model.embed_tokens"),
        final_norm=dense_root("model.norm.weight"),
        lm_head=quant_root("lm_head"),
        layers=[load_qwen35_moe_layer(tensors, config, index) for index in layers],
    )
