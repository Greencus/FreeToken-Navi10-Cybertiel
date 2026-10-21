"""GGUF tensors -> plain torch fp16/bf16, for backends with no ggml kernels (Vulkan).

The CUDA path runs packed ggml rows through its own kernels; the Vulkan backend has
none, so a tensor must be dequantized on the host into an ordinary torch tensor
before the Vulkan linear/GEMV shaders see it. This module wraps the numpy
dequantizer already vendored in ``freetoken.third_party.gguf.quants`` and adds the
three things the Vulkan loader needs:

- correct handling of the 3-D per-expert layout (``ffn_*_exps``): dequantize ONE
  expert slab at a time, never the whole bank as one blob (a 256-expert Q5_K slab
  is 184 MB packed / 840 MB as fp32);
- an explicit orientation contract: ggml dims are stored fastest-dim-first and are
  the REVERSE of torch's row-major order (``ne = (K, N)`` -> torch ``(N, K)``;
  ``ne = (K, rows, E)`` -> torch ``(E, rows, K)``);
- an optional byte-budgeted host LRU cache for dequantized expert slabs, because
  decode re-reads the same experts every token (budget: ``FTV_GGUF_DEQUANT_CACHE_MB``,
  default 256 MB; 0 disables).

Every function memory-maps through ``GgufTensorInfo.data()`` - nothing here copies
the 22.8 GB file.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import float_dtype, is_quantized, row_bytes, type_name
from freetoken.third_party.gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from freetoken.third_party.gguf.quants import dequantize as _np_dequantize

from .reader import GgufTensorInfo, gguf_tensor_index

# ggml F32 / F16 / BF16 -> torch (same table as tensors.py, spelled out for the CPU path)
_FLOAT_DTYPES = {
    int(GGMLQuantizationType.F32): torch.float32,
    int(GGMLQuantizationType.F16): torch.float16,
    int(GGMLQuantizationType.BF16): torch.bfloat16,
}

DEQUANT_CACHE_ENV = "FTV_GGUF_DEQUANT_CACHE_MB"
DEFAULT_CACHE_MB = 256


def torch_shape(info: GgufTensorInfo) -> tuple[int, ...]:
    """The torch shape of the fully dequantized tensor: ggml ``ne`` reversed."""
    return tuple(reversed(info.ne))


def _as_torch(floats: np.ndarray, out_dtype: torch.dtype) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(floats))
    return t if t.dtype == out_dtype else t.to(out_dtype)


def dequant_matrix(info: GgufTensorInfo, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The whole tensor dequantized to ``dtype``, in torch order (ggml dims reversed).

    A 2-D matrix ``ne=(K, N)`` comes out as ``(N, K)`` - the layout torch's
    ``linear(weight)`` wants. A 3-D expert bank ``ne=(K, rows, E)`` comes out as
    ``(E, rows, K)``; prefer :func:`dequant_expert` for those to bound peak memory.
    """
    if not is_quantized(info.ggml_type):
        raw = np.array(info.data())
        el = torch.from_numpy(raw).view(_FLOAT_DTYPES[info.ggml_type])
        if el.dtype != dtype:
            el = el.to(dtype)
        # reshape always: F16/BF16 at their native dtype used to return flat
        # (the `el if el.dtype == dtype` short-circuit skipped it).
        return el.reshape(torch_shape(info))
    floats = _np_dequantize(np.array(info.data()), GGMLQuantizationType(info.ggml_type))
    return _as_torch(floats.reshape(torch_shape(info)), dtype)


def _expert_slab(info: GgufTensorInfo, expert: int) -> np.ndarray:
    """One expert's packed bytes as ``[rows, row_bytes]`` uint8 (zero-copy mmap view)."""
    if len(info.ne) != 3:
        raise ValueError(f"{info.name}: ne={info.ne} is not a 3-D expert bank")
    k, rows, experts = info.ne
    if not 0 <= expert < experts:
        raise IndexError(f"{info.name}: expert {expert} outside [0, {experts})")
    per_expert = rows * row_bytes(info.ggml_type, k)
    base = expert * per_expert
    return np.frombuffer(info.data(), dtype=np.uint8, count=per_expert, offset=base).reshape(rows, row_bytes(info.ggml_type, k))


def dequant_expert(
    index: dict[str, GgufTensorInfo] | None,
    tensor: str | GgufTensorInfo,
    expert: int,
    *,
    model_path: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """One expert's slice of a 3-D bank, dequantized to ``dtype`` as ``(rows, K)``.

    ``tensor`` is either the :class:`GgufTensorInfo` itself or a GGUF tensor name
    resolved through ``index`` (build ``index`` once with
    ``gguf_tensor_index(model_path)`` and pass it; ``model_path`` alone also works).
    Dequantizes just that expert's slab - the other experts' bytes are never read
    into RAM.
    """
    if isinstance(tensor, str):
        if index is None:
            if model_path is None:
                raise ValueError("dequant_expert needs an index or a model_path")
            index = gguf_tensor_index(model_path)
        info = index[tensor]
    else:
        info = tensor
    if is_quantized(info.ggml_type):
        floats = _np_dequantize(_expert_slab(info, expert), GGMLQuantizationType(info.ggml_type))
    else:
        k, rows, _ = info.ne
        el = float_dtype(info.ggml_type)  # torch dtype
        np_dtype = torch.empty(0, dtype=el).numpy().dtype
        floats = np.frombuffer(info.data(), dtype=np_dtype, count=rows * k, offset=expert * rows * k * np_dtype.itemsize).reshape(rows, k)
    return _as_torch(floats, dtype)


def expert_count(info: GgufTensorInfo) -> int:
    """Experts in a 3-D bank (``ne`` must be ``(K, rows, E)``)."""
    if len(info.ne) != 3:
        raise ValueError(f"{info.name}: ne={info.ne} is not a 3-D expert bank")
    return int(info.ne[2])


def expert_bytes_packed(info: GgufTensorInfo) -> int:
    """Packed bytes of one expert slab of a 3-D bank."""
    k, rows, _ = info.ne
    return rows * row_bytes(info.ggml_type, k)


def dequant_expert_banks(
    model_path: str,
    layers: range | list[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
    cache: "DequantCache | None" = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """``{layer: {"gate_up": (2I, K) cat(gate, up), "down": (H, I)}}`` of dequantized experts.

    Matches the packed-bank layout ``freetoken.models.gguf.experts`` builds (gate rows
    then up rows on the output axis) so a Vulkan MoE dispatch can consume it directly.
    With ``cache``, repeated decodes reuse dequantized slabs up to the byte budget.
    """
    index = gguf_tensor_index(model_path)
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        blk = f"blk.{layer}"
        g = index.get(f"{blk}.ffn_gate_exps.weight")
        u = index.get(f"{blk}.ffn_up_exps.weight")
        d = index.get(f"{blk}.ffn_down_exps.weight")
        if g is None or d is None:
            raise KeyError(f"{model_path}: blk.{layer} lacks routed-expert tensors")
        row: dict[str, torch.Tensor] = {}
        row["gate_up"] = (
            torch.cat([_cached_expert(cache, index, g, e, dtype) for e in range(expert_count(g))], dim=0)
            if u is None
            else torch.cat(
                [
                    torch.cat(
                        [
                            _cached_expert(cache, index, g, e, dtype),
                            _cached_expert(cache, index, u, e, dtype),
                        ],
                        dim=0,
                    )
                    for e in range(expert_count(g))
                ],
                dim=0,
            )
        )
        row["down"] = torch.cat([_cached_expert(cache, index, d, e, dtype) for e in range(expert_count(d))], dim=0)
        out[layer] = row
    return out


def _cached_expert(
    cache: "DequantCache | None",
    index: dict[str, GgufTensorInfo],
    info: GgufTensorInfo,
    expert: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cache is None:
        return dequant_expert(index, info, expert, dtype=dtype)
    return cache.get_or_compute(
        (info.root or info.path, info.name, expert, str(dtype)),
        lambda: dequant_expert(index, info, expert, dtype=dtype),
    )


class DequantCache:
    """Byte-budgeted LRU of dequantized tensors, keyed by any hashable.

    The Vulkan decode path re-fetches the same routed experts every token; caching
    the dequantized (bf16) slab trades host RAM for the per-miss numpy dequant.
    Budget comes from ``FTV_GGUF_DEQUANT_CACHE_MB`` (default 256; 0 = disabled, in
    which case :meth:`get_or_compute` just computes). ``None`` cache instances are
    valid everywhere a cache is accepted, so callers can pass
    ``DequantCache.from_env() or None``.
    """

    def __init__(self, budget_mb: float):
        self.budget_bytes = int(budget_mb * 2**20)
        self._lock = threading.Lock()
        self._lru: OrderedDict[object, torch.Tensor] = OrderedDict()
        self._used = 0

    @classmethod
    def from_env(cls, env: str = DEQUANT_CACHE_ENV, default_mb: float = DEFAULT_CACHE_MB) -> "DequantCache | None":
        """A cache sized by ``env``; ``None`` when the env disables it (0 / non-numeric)."""
        raw = os.environ.get(env)
        mb = default_mb if raw in (None, "") else _parse_mb(raw)
        return None if mb <= 0 else cls(mb)

    def __len__(self) -> int:
        with self._lock:
            return len(self._lru)

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    def get_or_compute(self, key: object, compute):
        """Cached ``compute()``; the returned tensor may be shared - do not mutate it."""
        with self._lock:
            hit = self._lru.get(key)
            if hit is not None:
                self._lru.move_to_end(key)
                return hit
        value = compute()
        size = value.numel() * value.element_size()
        with self._lock:
            if size > self.budget_bytes:  # too big to ever fit: don't thrash the LRU
                return value
            self._lru[key] = value
            self._used += size
            while self._used > self.budget_bytes and self._lru:
                _, evicted = self._lru.popitem(last=False)
                self._used -= evicted.numel() * evicted.element_size()
        return value

    def clear(self) -> None:
        with self._lock:
            self._lru.clear()
            self._used = 0


def _parse_mb(raw: str) -> float:
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(f"{DEQUANT_CACHE_ENV} must be a number of MB, got {raw!r}") from None


__all__ = [
    "DEQUANT_CACHE_ENV", "DEFAULT_CACHE_MB", "DequantCache",
    "dequant_expert", "dequant_expert_banks", "dequant_matrix",
    "expert_bytes_packed", "expert_count", "torch_shape",
]
