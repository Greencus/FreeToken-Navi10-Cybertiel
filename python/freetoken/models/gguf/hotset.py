"""Static hot-set expert tables for qwen35moe GGUF MoE layers.

Why this exists
---------------
A router-locality study on the CyberTiel GGUF (218-token prompt, top-8 for
all 40 layers; see the module doc of the builder below) measured that only
~15 distinct experts per layer are ever selected (min 10, max 20 of 256),
so the routed-expert traffic looks pin-able: pinning the frequent experts
in VRAM would collapse the ~604 MB/token of expert H2D traffic (8/256
experts x 40 layers x 3 matrices) to the few misses outside the hot set.

**Measured outcome: identity-correct, but NOT a performance win, which is
why the pin ships opt-in (`FTV_GGUF_HOTSET=0` default).** The number this
module's first draft quoted - a per-layer LRU with 16 slots hitting 99.2%
of routed experts - came from ONE repeated 320-token prompt. On real text
the plain LRU already delivers an **85.8% request hit rate and ~83.7
MiB/token** of expert traffic at the default cache budget, and on the
merged real-text trace (repeated 320-tok + unique 218-tok, the trace the
shipped table was built from) the table's own replay is **66.3% by request
and 66.0% by packed bytes**, not 99.2%. Pinning the table's hot set (2925
MiB on this file) carves that out of the same `FTV_EXP_CACHE_MB` budget,
so the LRU region shrinks and its hit rate regresses to **59.5%** - the
pin trades LRU capacity for hit certainty and loses. Keep the table for
experiments; do not enable the pin for speed without re-measuring on the
workload you care about.

This module is the TABLE layer, GPU-agnostic: it builds, stores and replays
the per-layer ranked hot-expert lists. The runtime that consumes the table
and pins the experts lives in
``freetoken.models.qwen3_5_moe.vulkan.hotset_resident``. The table is a
small ``.npz`` (a few KiB) so it ships with the package and loads WITHOUT
the multi-GB GGUF or the original router trace.

Storage format (npz, format_version=1)
--------------------------------------
    expert_ids : int16  [L, K]  per layer, expert ids ranked by selections
    counts     : int32  [L, K]  selection counts in the source trace
    num_experts, trace_tokens, trace_topk, format_version : int scalars
    source     : bytes          trace provenance (file name / prompt tag)
    layer_expert_bytes : int64 [L] (optional) packed bytes of one expert's
                       gate+up+down slabs on that layer, used to weight
                       the byte hit rate (Q5_K layers are bigger than Q4_K)
    trace_hit_rate, trace_byte_hit_rate : float64 scalars (optional;
                       replay of THIS table on the source trace, computed
                       at build time when the model file is available)

Regenerating the table (reproducible)
-------------------------------------
Any router-id trace of shape ``[layers, tokens, topk]`` works; one prompt
is enough::

    PYTHONPATH=python python -m freetoken.models.gguf.hotset \
        --trace /tmp/opencode/perf_ceiling/router_ids.npy \
        --model /tmp/opencode/CyberTiel.gguf \
        --top-k 24

which rebuilds ``hotset_cybertiel_qwen35moe.npz`` next to this module and
prints the replayed per-layer / overall / byte-weighted hit rates. The
builder itself is ``build_hotset``; the trace is only needed at build
time, never at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Bump when the npz key set changes; load_hotset validates it.
TABLE_FORMAT_VERSION = 1

# The shipped CyberTiel table: top-24/layer covers the full distinct range
# seen in the source trace (max 20) with margin; the LRU study plateaus at
# 99.4% from 24 slots on, and 24/layer x 40 layers of packed K-quant bytes
# is ~2.2 GiB with both cache-key styles - inside the 8 GB card's budget.
DEFAULT_TOP_K = 24

_TABLE_NAME = "hotset_cybertiel_qwen35moe.npz"


def default_table_path() -> str:
    """Path of the table shipped next to this module (may not exist)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _TABLE_NAME)


@dataclass
class HotsetTable:
    """Per-layer ranked hot-expert list (see module doc for the format)."""

    expert_ids: np.ndarray          # int16 [L, K]
    counts: np.ndarray              # int32 [L, K]
    num_experts: int
    trace_tokens: int
    trace_topk: int
    source: str = ""
    layer_expert_bytes: np.ndarray | None = None   # int64 [L] or None
    trace_hit_rate: float | None = None
    trace_byte_hit_rate: float | None = None

    @property
    def num_layers(self) -> int:
        return int(self.expert_ids.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.expert_ids.shape[1])

    def experts_of(self, layer: int, k: int | None = None) -> np.ndarray:
        """Ranked expert ids of ``layer``, truncated to ``k`` (None = all)."""
        row = self.expert_ids[layer]
        return row if k is None else row[:k]

    def describe(self) -> str:
        hit = ("?" if self.trace_hit_rate is None
               else f"{self.trace_hit_rate * 100:.1f}%")
        bhit = ("?" if self.trace_byte_hit_rate is None
                else f"{self.trace_byte_hit_rate * 100:.1f}%")
        return (
            f"HotsetTable(layers={self.num_layers}, top_k={self.top_k}, "
            f"experts={self.num_experts}, trace_tokens={self.trace_tokens}, "
            f"trace_hit={hit}, trace_byte_hit={bhit}, source={self.source!r})"
        )


def build_hotset(
    router_ids: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    *,
    source: str = "",
    layer_expert_bytes: Sequence[int] | np.ndarray | None = None,
) -> HotsetTable:
    """Rank per-layer experts by selection frequency in a router trace.

    ``router_ids`` is ``[layers, tokens, topk]`` of routed expert ids (the
    shape ``/tmp/opencode/perf_ceiling/router_ids.npy`` has). Returns the
    top-``top_k`` ids per layer (ties broken by expert id, so the table is
    deterministic); layers with fewer distinct experts than ``top_k`` pad
    with the least-frequent ids (still real experts - a padded id can only
    cost VRAM, never correctness, because pinning never changes routing).
    """
    ids = np.asarray(router_ids)
    if ids.ndim != 3:
        raise ValueError(f"router_ids must be [L, T, k], got shape {ids.shape}")
    num_layers, tokens, trace_topk = ids.shape
    num_experts = int(ids.max()) + 1 if ids.size else 0
    top_k = max(1, min(int(top_k), num_experts))

    out_ids = np.zeros((num_layers, top_k), dtype=np.int16)
    out_counts = np.zeros((num_layers, top_k), dtype=np.int32)
    for layer in range(num_layers):
        flat = ids[layer].reshape(-1)
        counts = np.bincount(flat.astype(np.int64), minlength=num_experts)
        # -counts for frequency, expert id as deterministic tiebreak.
        order = np.lexsort((np.arange(num_experts), -counts))[:top_k]
        out_ids[layer] = order.astype(np.int16)
        out_counts[layer] = counts[order].astype(np.int32)

    table = HotsetTable(
        expert_ids=out_ids,
        counts=out_counts,
        num_experts=num_experts,
        trace_tokens=int(tokens),
        trace_topk=int(trace_topk),
        source=source,
    )
    if layer_expert_bytes is not None:
        table.layer_expert_bytes = np.asarray(layer_expert_bytes, dtype=np.int64)
    rates = replay_hit_rate(table, ids)
    table.trace_hit_rate = rates["overall"]
    if table.layer_expert_bytes is not None:
        table.trace_byte_hit_rate = rates["byte_weighted"]
    return table


def save_hotset(table: HotsetTable, path: str) -> None:
    """Write ``table`` to ``path`` (npz, compressed; a few KiB)."""
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray([TABLE_FORMAT_VERSION], dtype=np.int32),
        "expert_ids": table.expert_ids,
        "counts": table.counts,
        "num_experts": np.asarray([table.num_experts], dtype=np.int32),
        "trace_tokens": np.asarray([table.trace_tokens], dtype=np.int32),
        "trace_topk": np.asarray([table.trace_topk], dtype=np.int32),
        "source": np.asarray(table.source.encode("utf-8")),
    }
    if table.layer_expert_bytes is not None:
        arrays["layer_expert_bytes"] = np.asarray(
            table.layer_expert_bytes, dtype=np.int64)
    if table.trace_hit_rate is not None:
        arrays["trace_hit_rate"] = np.asarray([table.trace_hit_rate])
    if table.trace_byte_hit_rate is not None:
        arrays["trace_byte_hit_rate"] = np.asarray([table.trace_byte_hit_rate])
    np.savez_compressed(path, **arrays)


def load_hotset(path: str) -> HotsetTable:
    """Load a table written by :func:`save_hotset` (no model, no trace)."""
    with np.load(path) as z:
        version = int(z["format_version"][0])
        if version != TABLE_FORMAT_VERSION:
            raise ValueError(
                f"{path}: hotset format_version {version}, "
                f"this build understands {TABLE_FORMAT_VERSION}")
        table = HotsetTable(
            expert_ids=z["expert_ids"].astype(np.int16),
            counts=z["counts"].astype(np.int32),
            num_experts=int(z["num_experts"][0]),
            trace_tokens=int(z["trace_tokens"][0]),
            trace_topk=int(z["trace_topk"][0]),
            source=str(np.asarray(z["source"]).tobytes().decode("utf-8")),
            layer_expert_bytes=(
                z["layer_expert_bytes"].astype(np.int64)
                if "layer_expert_bytes" in z.files else None),
            trace_hit_rate=(
                float(z["trace_hit_rate"][0])
                if "trace_hit_rate" in z.files else None),
            trace_byte_hit_rate=(
                float(z["trace_byte_hit_rate"][0])
                if "trace_byte_hit_rate" in z.files else None),
        )
    return table


def layer_expert_bytes_from_gguf(model_path: str, num_layers: int) -> np.ndarray:
    """Packed bytes of one expert's gate+up+down slabs, per layer.

    Read from GGUF metadata only (never the tensor data), so Q5_K layers
    (blocks 0-14 on CyberTiel) weigh more than Q4_K ones and the byte hit
    rate is honest.
    """
    from .packed import packed_expert_info
    from .reader import gguf_tensor_index

    index = gguf_tensor_index(model_path)
    out = np.zeros(num_layers, dtype=np.int64)
    for layer in range(num_layers):
        total = 0
        for proj in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            info = index.get(name)
            if info is None:
                continue
            meta = packed_expert_info(index, name, 0)
            total += meta.nbytes
        out[layer] = total
    return out


def replay_hit_rate(
    table: HotsetTable,
    router_ids: np.ndarray,
    *,
    layer_expert_bytes: Sequence[int] | np.ndarray | None = None,
) -> dict:
    """Replay a router trace against a static table.

    Returns ``{"per_layer": [rate_l], "overall": r, "byte_weighted": r,
    "requests": n}`` where a request is ONE routed expert (top-8 per token
    per layer). ``byte_weighted`` weights each layer's hit rate by the
    packed bytes its experts occupy (defaults to the table's own
    ``layer_expert_bytes`` when present, else uniform).
    """
    ids = np.asarray(router_ids)
    if ids.shape[0] != table.num_layers:
        raise ValueError(
            f"trace has {ids.shape[0]} layers, table has {table.num_layers}")
    k = table.top_k
    per_layer = np.empty(table.num_layers, dtype=np.float64)
    req_layer = np.empty(table.num_layers, dtype=np.float64)
    for layer in range(table.num_layers):
        hot = table.expert_ids[layer, :k]
        req = ids[layer]                      # [T, topk]
        hits = np.isin(req, hot).sum()
        req_layer[layer] = req.size
        per_layer[layer] = hits / req.size
    overall = float((per_layer * req_layer).sum() / req_layer.sum())

    weights = layer_expert_bytes
    if weights is None:
        weights = table.layer_expert_bytes
    if weights is None:
        byte_weighted = overall
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != table.num_layers:
            raise ValueError("layer_expert_bytes length != num_layers")
        byte_weighted = float(
            (per_layer * req_layer * w).sum() / (req_layer * w).sum())
    return {
        "per_layer": per_layer.tolist(),
        "overall": overall,
        "byte_weighted": byte_weighted,
        "requests": int(req_layer.sum()),
    }


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m freetoken.models.gguf.hotset",
        description="Build/refresh a static hot-set expert table from a "
                    "router-id trace (.npy, [L, T, topk]).")
    ap.add_argument("--trace", required=True,
                    help="router trace .npy (e.g. /tmp/opencode/perf_ceiling/"
                         "router_ids.npy)")
    ap.add_argument("--out", default=None,
                    help="output .npz (default: the shipped CyberTiel table)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"experts per layer to keep (default {DEFAULT_TOP_K})")
    ap.add_argument("--model", default=None,
                    help="GGUF for per-layer packed byte sizes + byte hit "
                         "rate (optional; metadata-only reads)")
    ap.add_argument("--source", default=None,
                    help="provenance tag (default: trace file name)")
    args = ap.parse_args(argv)

    trace = np.load(args.trace)
    source = args.source or os.path.basename(args.trace)
    layer_bytes = None
    if args.model:
        layer_bytes = layer_expert_bytes_from_gguf(
            args.model, int(trace.shape[0]))
    table = build_hotset(trace, top_k=args.top_k, source=source,
                         layer_expert_bytes=layer_bytes)
    out = args.out or default_table_path()
    save_hotset(table, out)

    rates = replay_hit_rate(table, trace)
    print(f"wrote {out}")
    print(table.describe())
    per = rates["per_layer"]
    print(f"replay on source trace: overall={rates['overall'] * 100:.1f}% "
          f"byte_weighted={rates['byte_weighted'] * 100:.1f}% "
          f"(per-layer min={min(per) * 100:.1f}% max={max(per) * 100:.1f}%)")
    if layer_bytes is not None:
        total = int(np.mean(layer_bytes) * table.top_k * table.num_layers)
        print(f"pin cost estimate: ~{total / 2**30:.2f} GiB "
              f"(top-{table.top_k} x {table.num_layers} layers, "
              f"3 per-matrix slabs/layer-expert)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
