"""PACKED ggml expert bytes for the Vulkan K-quant shaders - no host dequant.

Why this exists
---------------
Before this module the Vulkan port served the CyberTiel GGUF by dequantizing
every routed K-quant expert on the HOST into BF16 and uploading BF16: ~3.6-5.5
s of numpy dequant per token plus ~1.9 GB/token of H2D traffic, i.e. 29.7
s/token. The Vulkan backend now has in-shader unpackers
(``freetoken/kernel/vulkan/shaders/linear_q*k_*_f32.comp``), so the hot path
must hand the GPU the GGUF bytes AS THEY ARE: one expert slab of
``(rows, row_bytes)`` uint8 straight out of the memory-mapped file, never a
dequantized copy. This module is the packed counterpart of ``dequant.py`` -
same mmap, same orientation contract, zero float math.

What the engine calls
---------------------
One routed expert of one MoE layer (see ``docs/vulkan/kquant-kernels.md``)::

    from freetoken.models.gguf.packed import (
        kquant_qtype, packed_expert, packed_expert_info, packed_gate_up_expert)

    info = packed_expert_info(index, f"blk.{layer}.ffn_down_exps.weight", e)
    slab = packed_expert(index, f"blk.{layer}.ffn_down_exps.weight", e)   # (rows, rowb) u8
    qtype = kquant_qtype(info.ggml_type)          # "q4_k" / "q5_k" / "q8_0"

    h_w = ext.upload_resident(packed_tensor(slab)) # packed bytes, NOT bf16
    ext.batch_kquant_io(qtype, "acc", h_w, h_x, h_acc,
                        1, info.rows, info.cols, 1.0, slot0)

For the fused gate+up the bundle is the two slabs stacked on the ROW axis
(gate rows 0..N-1, up rows N..2N-1) - exactly what
``linear_q*k_gateup_silu_f32.comp`` reads::

    gu = packed_gate_up_expert(index, layer, e)   # (2N, rowb) u8
    ext.batch_kquant_io(qtype, "gateup", h_gu, h_x, h_silu,
                        1, N, info.cols, 1.0, slot0)

``slot0`` is the first row of that matrix inside the resident arena, and the
arena must place rows at ``slot0 * row_bytes`` bytes - true for any
row-contiguous packing because ``row_bytes`` is a multiple of 4 for every
supported type.

Q8_0 caveat: the shader reads at most 4 bytes past a packed row while
rebuilding unaligned dwords, so a Q8_0 arena wants 4 bytes of slack after
its last row. Q4_K/Q5_K need none.
"""

from __future__ import annotations

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import is_quantized, row_bytes
from freetoken.third_party.gguf import GGMLQuantizationType

from .reader import GgufTensorInfo, gguf_tensor_index

# ggml type id -> the shader's qtype selector (ext.batch_kquant_io / the
# linear_q*_resident_io bindings). Anything not listed has no in-shader
# unpacker and must keep going through dequant.py on the host.
_KQUANT_BY_ID: dict[int, str] = {
    int(GGMLQuantizationType.Q4_K): "q4_k",
    int(GGMLQuantizationType.Q5_K): "q5_k",
    int(GGMLQuantizationType.Q8_0): "q8_0",
}

# bytes per packed row that are a multiple of 4 (the shaders index the
# buffer as uint32 words)
_ALIGN4_OK = {"q4_k": True, "q5_k": True, "q8_0": False}


def kquant_qtype(ggml_type: int) -> str | None:
    """The Vulkan shader selector for a ggml type id, or None if unsupported."""
    return _KQUANT_BY_ID.get(int(ggml_type))


def _resolve(index: dict[str, GgufTensorInfo] | None,
             tensor: str | GgufTensorInfo,
             model_path: str | None) -> GgufTensorInfo:
    if isinstance(tensor, GgufTensorInfo):
        return tensor
    if index is None:
        if model_path is None:
            raise ValueError("packed_expert needs an index or a model_path")
        index = gguf_tensor_index(model_path)
    return index[tensor]


class PackedExpert:
    """Shape/orientation facts about one expert slab of a -D GGUF bank."""

    __slots__ = ("name", "ggml_type", "qtype", "rows", "cols", "row_bytes",
                 "nbytes", "experts")

    def __init__(self, info: GgufTensorInfo, expert: int):
        if len(info.ne) != 3:
            raise ValueError(f"{info.name}: ne={info.ne} is not a 3-D expert bank")
        k, rows, experts = info.ne
        if not 0 <= expert < experts:
            raise IndexError(f"{info.name}: expert {expert} outside [0, {experts})")
        self.name = info.name
        self.ggml_type = info.ggml_type
        self.qtype = kquant_qtype(info.ggml_type)
        self.rows = int(rows)
        self.cols = int(k)          # the GEMV's K (torch: (rows, K))
        self.row_bytes = row_bytes(info.ggml_type, k)
        self.nbytes = int(rows) * self.row_bytes
        self.experts = int(experts)

    @property
    def aligned(self) -> bool:
        """True when row_bytes is a multiple of 4 (uint32-indexed shaders)."""
        if self.qtype is None:
            return False
        return self.row_bytes % 4 == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "qtype": self.qtype, "rows": self.rows,
            "cols": self.cols, "row_bytes": self.row_bytes,
            "nbytes": self.nbytes, "experts": self.experts,
            "aligned": self.aligned,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"PackedExpert({self.name!r}, qtype={self.qtype}, "
                f"rows={self.rows}, cols={self.cols}, "
                f"row_bytes={self.row_bytes}, experts={self.experts})")


def packed_expert_info(index: dict[str, GgufTensorInfo] | None,
                       tensor: str | GgufTensorInfo,
                       expert: int,
                       *,
                       model_path: str | None = None) -> PackedExpert:
    """Shape metadata for one expert slab; raises on unsupported types."""
    info = _resolve(index, tensor, model_path)
    if not is_quantized(info.ggml_type):
        raise ValueError(f"{info.name}: type {info.ggml_type} is not quantized; "
                         "use tensors.py for plain F16/BF16/F32 banks")
    meta = PackedExpert(info, expert)
    if meta.qtype is None:
        raise ValueError(
            f"{info.name}: ggml type {info.ggml_type} has no Vulkan in-shader "
            "unpacker (supported: Q4_K, Q5_K, Q8_0); dequant on the host via "
            "dequant.dequant_expert instead")
    return meta


def packed_expert(index: dict[str, GgufTensorInfo] | None,
                  tensor: str | GgufTensorInfo,
                  expert: int,
                  *,
                  model_path: str | None = None) -> np.ndarray:
    """One expert's PACKED bytes as ``(rows, row_bytes)`` uint8.

    Zero-copy view into the memory-mapped GGUF (same view ``dequant.py``
    calls a slab) - read it, upload it, drop it; never holds the model in
    RAM. The rows are in torch order ``(rows, K)``, i.e. ggml ``ne`` with
    the expert dimension removed, so ``y = x @ W.T`` with this W is the
    expert's matmul.
    """
    info = _resolve(index, tensor, model_path)
    k, rows, experts = info.ne
    if not 0 <= expert < experts:
        raise IndexError(f"{info.name}: expert {expert} outside [0, {experts})")
    per_expert = int(rows) * row_bytes(info.ggml_type, k)
    base = expert * per_expert
    rb = row_bytes(info.ggml_type, k)
    return np.frombuffer(info.data(), dtype=np.uint8, count=per_expert,
                         offset=base).reshape(int(rows), rb)


def packed_gate_up_expert(index: dict[str, GgufTensorInfo] | None,
                          layer: int,
                          expert: int,
                          *,
                          model_path: str | None = None,
                          gate: str = "ffn_gate_exps.weight",
                          up: str = "ffn_up_exps.weight",
                          block: str = "blk.{}") -> np.ndarray:
    """The stacked gate+up packed bundle ``(2N, row_bytes)`` uint8.

    Gate rows first, up rows after - the layout
    ``linear_q4k/q5k_gateup_silu_f32.comp`` expects (row n = gate, row n+N =
    up). One memcpy; no dequant. Falls back to the gate tensor alone when the
    model has no up tensor (M-shaped MoE).
    """
    if index is None:
        if model_path is None:
            raise ValueError("packed_gate_up_expert needs an index or a model_path")
        index = gguf_tensor_index(model_path)
    g = packed_expert(index, block.format(layer) + "." + gate, expert)
    u = index.get(block.format(layer) + "." + up)
    if u is None:
        return g
    return np.concatenate([g, packed_expert(index, u, expert)], axis=0)


def packed_tensor(slab: np.ndarray) -> torch.Tensor:
    """A contiguous, writable uint8 torch tensor of a packed slab (for upload).

    ``np.frombuffer`` views of an mmap are read-only; torch would warn. This
    is a memcpy of the packed bytes only (0.56 MiB per Q4_K expert), not a
    dequant - orders of magnitude cheaper than the BF16 path it replaces.
    """
    if slab.dtype != np.uint8:
        raise ValueError(f"packed slab must be uint8, got {slab.dtype}")
    return torch.from_numpy(np.ascontiguousarray(slab).copy())


def packed_expert_bytes(index: dict[str, GgufTensorInfo] | None,
                        tensor: str | GgufTensorInfo,
                        expert: int,
                        *,
                        model_path: str | None = None) -> int:
    """Packed bytes of one expert slab (alias of ``dequant.expert_bytes_packed``)."""
    info = _resolve(index, tensor, model_path)
    k, rows, _ = info.ne
    return int(rows) * row_bytes(info.ggml_type, k)


__all__ = [
    "PackedExpert",
    "kquant_qtype",
    "packed_expert",
    "packed_expert_bytes",
    "packed_expert_info",
    "packed_gate_up_expert",
    "packed_tensor",
]
