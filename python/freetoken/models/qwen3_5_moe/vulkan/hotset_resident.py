"""Static hot-set expert residency for the Vulkan GGUF decoder.

Why this exists
---------------
Serving the CyberTiel GGUF moves ~604 MB of packed expert bytes over PCIe
per generated token (8 of 256 experts x 40 layers x 3 matrices), so expert
traffic is the decode cost. A router-locality study measured that routing
concentrates on a small per-layer working set, which suggests PINNING the
frequent experts in VRAM once at init so the per-token path only streams
the experts outside the hot set.

**Measured: identity-correct, but not a speed win - hence the opt-in
default (`FTV_GGUF_HOTSET=0`).** The plain LRU already hits 85.8% of
routed requests at ~83.7 MiB/token (66.3% request / 66.0% byte replay of
the table on the merged real-text trace), and pinning the table's 2925 MiB
carves that out of the same `FTV_EXP_CACHE_MB` budget, regressing the LRU
hit rate to 59.5%. Details in ``freetoken.models.gguf.hotset``.

This module adds that pinning on top of ``ResidentGGUF`` WITHOUT touching
it: ``HotsetResidentGGUF`` is a subclass that

- loads a static per-layer hot-expert table (``freetoken.models.gguf.hotset``
  ``.npz`` - a few KiB, no GGUF/trace needed at runtime);
- uploads the hot experts of every layer through the engine's OWN miss
  paths (``_expert_handle`` for the sequential/prefill per-matrix keys and
  ``_gguf_gu_handle``/``_gguf_dn_handle`` for the batched-decode keys - the
  two paths use DIFFERENT cache keys, see below), then marks those keys
  non-evictable;
- overrides ``_cache_put`` so LRU/LFU eviction never picks a pinned key:
  the hot set is carved OUT of the existing ``FTV_EXP_CACHE_MB`` budget
  (total expert VRAM stays capped), and the LRU region handles everything
  else exactly as before.

Cache-key contract (why both styles are pinned): the sequential path
(prefill ``_moe_chunk`` and ``decode_step``) keys experts per matrix -
``(id(gate), e)``, ``(id(up), e)``, ``(id(down), e)`` - while the batched
decode path (``_record_moe_in``, on by default) keys the FUSED gate+up
bundle ``(id(gate), id(up), e)`` plus the down matrix ``(id(down), e)``
(the down key is shared). Pinning both styles costs 5 packed slabs per
hot (layer, expert) instead of 3; ``FTV_GGUF_HOTSET_KEYS`` selects
``both`` (default) / ``batched`` / ``sequential`` when VRAM is tight.

Env knobs (all read at construction):
    FTV_GGUF_HOTSET        0 (default; opt-in) pin on init; the pin is
                           identity-correct but NOT a speed win - the plain
                           LRU already hits 85.8% of routed requests at
                           ~83.7 MiB/token (66.3% table replay on the merged
                           real-text trace), and pinning the table's 2925
                           MiB steals that LRU budget and regresses the hit
                           rate to 59.5%. See models/gguf/hotset.py.
    FTV_GGUF_HOTSET_MB     pin budget in MiB (default 4096; clamped to
                           FTV_EXP_CACHE_MB so the total never grows)
    FTV_GGUF_HOTSET_TABLE  table .npz path (default: the shipped
                           hotset_cybertiel_qwen35moe.npz)
    FTV_GGUF_HOTSET_K      cap experts/layer from the table (0 = table's K)
    FTV_GGUF_HOTSET_KEYS   both | batched | sequential (default both)

Integration point
-----------------
``engine.load_engine(gguf=...)`` currently returns ``ResidentGGUF``; to
wire the hot set in (env-gated, no behavior change when FTV_GGUF_HOTSET=0
or the table file is missing)::

    if gguf is not None:
        from .hotset_resident import HotsetResidentGGUF
        return HotsetResidentGGUF(gguf, ext, max_len=max_len, kv_dtype=kv_dtype)

or swap the import in ``vulkan/__init__.py``. The subclass is a drop-in
replacement (same constructor signature, same decode paths).

Correctness contract
--------------------
Pinning never changes routing or numerics: a pinned expert is the same
upload the engine would have made on its first miss, just made earlier and
kept. Generation with the hot set on must be token-identical to the run
without it (argmax/greedy identity; see tests/vulkan/test_hotset_resident.py).
"""
from __future__ import annotations

import itertools
import os
import time
from typing import Any

import torch

from freetoken.models.gguf.hotset import HotsetTable, default_table_path, load_hotset
from freetoken.models.gguf.packed import packed_gate_up_expert, packed_tensor

from .gguf_resident import ResidentGGUF

_MODES = ("both", "batched", "sequential")


class HotsetResidentGGUF(ResidentGGUF):
    """``ResidentGGUF`` plus a non-evictable pinned expert hot set (module doc)."""

    def __init__(
        self,
        model_path: str,
        ext=None,
        max_len: int = 4096,
        kv_dtype: str = "fp32",
    ) -> None:
        # Before super().__init__: the _cache_put override below must see a
        # valid pinned-key set for every call, including any during init.
        self._hotset_keys: set[tuple] = set()
        self._hotset_bytes = 0
        self._hotset_per_layer: list[int] = []
        self._hotset_table: HotsetTable | None = None
        self._hotset_table_path = ""
        self._hotset_pin_stats: dict[str, int] = {}
        self._hotset_pin_seconds = 0.0
        self._hotset_enabled = os.environ.get("FTV_GGUF_HOTSET", "0").strip() != "0"
        try:
            self._hotset_budget = (
                int(os.environ.get("FTV_GGUF_HOTSET_MB", "4096")) * 1024 * 1024
            )
        except (TypeError, ValueError):
            self._hotset_budget = 4096 * 1024 * 1024
        mode = os.environ.get("FTV_GGUF_HOTSET_KEYS", "both").strip().lower()
        self._hotset_mode = mode if mode in _MODES else "both"
        try:
            self._hotset_k_cap = int(os.environ.get("FTV_GGUF_HOTSET_K", "0"))
        except (TypeError, ValueError):
            self._hotset_k_cap = 0
        super().__init__(model_path, ext, max_len=max_len, kv_dtype=kv_dtype)
        if self._hotset_enabled:
            if not self._packed_kq:
                print(
                    "FTV_GGUF_HOTSET: disabled (needs the packed K-quant lane; "
                    "FTV_GGUF_PACKED=0)",
                    flush=True,
                )
            elif self._exp_cache_budget <= 0:
                print(
                    "FTV_GGUF_HOTSET: disabled (FTV_EXP_CACHE_MB=0 - no cache "
                    "to pin into)",
                    flush=True,
                )
            else:
                self._hotset_load_and_pin()

    # -- pinning ----------------------------------------------------------
    def _hotset_load_and_pin(self) -> None:
        """Upload every hot expert of every layer and mark it non-evictable."""
        path = os.environ.get("FTV_GGUF_HOTSET_TABLE") or default_table_path()
        if not os.path.isfile(path):
            print(f"FTV_GGUF_HOTSET: no table at {path}; skipping pin", flush=True)
            return
        table = load_hotset(path)
        num_layers = len(self.config.layer_types)
        if (table.num_layers < num_layers
                or table.num_experts != self.config.num_experts):
            print(
                f"FTV_GGUF_HOTSET: table mismatch ({table.describe()} vs model "
                f"layers={num_layers} experts={self.config.num_experts}); "
                "skipping pin",
                flush=True,
            )
            return
        k = table.top_k
        if self._hotset_k_cap > 0:
            k = min(k, self._hotset_k_cap)
        # The pin is carved from the TOTAL expert budget so 8 GB VRAM stays
        # 8 GB: LRU region = FTV_EXP_CACHE_MB - pinned bytes.
        budget = min(self._hotset_budget, self._exp_cache_budget)
        mode = self._hotset_mode

        banks: list[tuple[Any, Any, Any] | None] = []
        for layer in range(num_layers):
            base = f"language_model.model.layers.{layer}"
            try:
                g = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
                u = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
                d = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
            except KeyError:
                banks.append(None)
                continue
            if any(getattr(m, "qtype", None) is None for m in (g, u, d)):
                banks.append(None)  # host-dequant bank: not pinnable cheaply
                continue
            banks.append((g, u, d))

        def pin_bytes(g: Any, u: Any, d: Any) -> int:
            b = 0
            if mode in ("both", "sequential"):
                b += g.packed_bytes + u.packed_bytes + d.packed_bytes
            else:
                b += d.packed_bytes  # down key is shared by both paths
            if mode in ("both", "batched"):
                b += g.packed_bytes + u.packed_bytes  # stacked gate+up bundle
            return b

        saved_stats = self.stat_calls
        self.stat_calls = {}
        per_layer = [0] * num_layers
        depth_done = 0
        t0 = time.perf_counter()
        try:
            # Round-robin by RANK depth across layers: budget truncation
            # degrades uniformly (every layer keeps its top experts) instead
            # of the early layers eating the whole budget.
            for depth in range(k):
                would_fit = True
                for layer in range(num_layers):
                    trio = banks[layer]
                    if trio is None:
                        continue
                    g, u, d = trio
                    e = int(table.expert_ids[layer, depth])
                    est = pin_bytes(g, u, d)
                    if self._hotset_bytes + est > budget:
                        would_fit = False
                        break
                    gid, uid, did = id(g), id(u), id(d)
                    # Mark BEFORE the upload so eviction pressure mid-pin
                    # can never pick an entry that just landed.
                    if mode in ("both", "sequential"):
                        self._hotset_keys.update({(gid, e), (uid, e), (did, e)})
                    else:
                        self._hotset_keys.add((did, e))
                    if mode in ("both", "batched"):
                        self._hotset_keys.add((gid, uid, e))
                    before = self._exp_cache_bytes
                    if mode in ("both", "sequential"):
                        self._expert_handle(g, e)
                        self._expert_handle(u, e)
                        self._expert_handle(d, e)
                    else:
                        self._expert_handle(d, e)
                    if mode in ("both", "batched"):
                        self._pin_fused_gateup(g, u, e)
                    self._hotset_bytes += self._exp_cache_bytes - before
                    per_layer[layer] += 1
                if not would_fit:
                    break
                depth_done += 1
        finally:
            self._hotset_pin_stats = dict(self.stat_calls)
            self.stat_calls = saved_stats
        self._hotset_pin_seconds = time.perf_counter() - t0
        self._hotset_per_layer = per_layer
        self._hotset_table = table
        self._hotset_table_path = path
        self.stats["hotset_pinned_entries"] = len(self._hotset_keys)
        self.stats["hotset_pinned_bytes"] = self._hotset_bytes
        print(self.hotset_report(), flush=True)

    def _pin_fused_gateup(self, g: Any, u: Any, expert: int) -> int:
        """Pin the stacked gate+up bundle for the batched-decode cache key.

        Data path mirrors ``GgufBatchedMixin._gguf_gu_handle``'s miss
        byte-for-byte (``packed_gate_up_expert`` -> ``packed_tensor`` ->
        Q8_0 slack), so the pinned buffer is indistinguishable from one
        the batched path would have uploaded. The UPLOAD is the explicit
        one-shot ``upload_resident_batch`` instead of
        ``_gguf_cache_resident``'s ``batch_upload``: the pre-pin runs at
        init, outside any open batch, and must not depend on batch
        context or its sync fallback.
        """
        key = (id(g), id(u), expert)
        hit = self._exp_cache.get(key)
        if hit is not None:
            self._cache_touch(key)
            return hit[0]
        ext = self.ext
        t0 = time.perf_counter()
        slab = packed_gate_up_expert(self.tensors.index, g.layer, expert)
        if slab.shape[0] != g.rows * 2:
            raise RuntimeError(
                f"layer {g.layer} expert {expert}: gate+up bundle has "
                f"{slab.shape[0]} rows, expected {g.rows * 2}")
        weight = packed_tensor(slab)
        if self._packed_qtype(g) == "q8_0":
            weight = torch.cat([weight, weight.new_zeros(4)])
        self._bump("exp_packed", time.perf_counter() - t0)
        w_bytes = weight.numel() * weight.element_size()
        t0 = time.perf_counter()
        pooled = self._pool_get(w_bytes, 0)
        h_w = pooled[0] if pooled is not None else ext.allocate_resident(w_bytes)
        ext.upload_resident_batch([h_w], [weight], [0])
        self._bump("exp_up", time.perf_counter() - t0)
        self._cache_put(key, (h_w, 0, w_bytes, w_bytes, 0))
        self._cache_freq(key)
        return h_w

    # -- non-evictable LRU ------------------------------------------------
    def _cache_put(self, key: tuple, entry: tuple) -> None:
        """Base LRU/LFU insert + eviction, minus the pinned hot set.

        Identical to ``ResidentNVFP4._cache_put`` except the victim scan
        skips ``_hotset_keys``; when only pinned entries remain the loop
        stops (the hot set was budget-clamped at pin time, so this only
        trips if the operator raises FTV_EXP_CACHE_MB below the pin after
        the fact - still safe, just over budget).
        """
        budget = self._exp_cache_budget
        if budget <= 0:
            return
        self._exp_cache[key] = entry
        self._exp_cache_bytes += entry[2]
        try:
            lfu = os.environ.get("FTV_CACHE_POLICY", "lru") == "lfu"
        except (TypeError, ValueError):
            lfu = False
        pinned = self._hotset_keys
        while self._exp_cache_bytes > budget and self._exp_cache:
            victim: tuple | None = None
            if lfu:
                # LFU-sampled: least-frequently used among the oldest 64
                # NON-pinned keys (same policy window as the base class).
                best_f = None
                for cand in itertools.islice(self._exp_cache, 64):
                    if cand in pinned:
                        continue
                    f = self._exp_freq.get(cand, 0)
                    if best_f is None or f < best_f:
                        best_f = f
                        victim = cand
                if victim is None:
                    for cand in self._exp_cache:
                        if cand not in pinned:
                            victim = cand
                            break
            else:
                for cand in self._exp_cache:
                    if cand not in pinned:
                        victim = cand
                        break
            if victim is None:
                break
            _hw, _hs, width, wb, sb = self._exp_cache.pop(victim)
            if lfu:
                self.stat_calls["cache_evict_lfu"] = (
                    self.stat_calls.get("cache_evict_lfu", 0) + 1)
            self.stat_calls["cache_evict"] = (
                self.stat_calls.get("cache_evict", 0) + 1)
            self._exp_cache_bytes -= width
            if victim in self._arena_keys:
                self._arena_drop(victim)
            else:
                key_sz = (wb, sb)
                self._exp_pool.setdefault(key_sz, []).append((_hw, _hs))
                self._exp_pool_bytes += wb + sb
                if self._exp_pool_bytes > budget // 4:
                    for _i in range(len(self._exp_pool.get(key_sz, []) ) - 1):
                        pass
                while self._exp_pool_bytes > budget // 4:
                    for _psz, _lst in list(self._exp_pool.items()):
                        if _lst:
                            _phw, _phs = _lst.pop()
                            self._exp_pool_bytes -= (_psz[0] + _psz[1])
                            try:
                                self.ext.free_resident(_phw)
                                self.ext.free_resident(_phs)
                            except RuntimeError:
                                pass

    # -- reporting ---------------------------------------------------------
    def hotset_info(self) -> dict:
        """Machine-readable hot-set state (for tests/benches)."""
        n_layers = len(self.config.layer_types)
        table_hit = None
        table_byte_hit = None
        if self._hotset_table is not None:
            table_hit = self._hotset_table.trace_hit_rate
            table_byte_hit = self._hotset_table.trace_byte_hit_rate
        return {
            "enabled": bool(self._hotset_keys),
            "table": self._hotset_table_path,
            "mode": self._hotset_mode,
            "top_k": max(self._hotset_per_layer, default=0),
            "per_layer": list(self._hotset_per_layer),
            "layers_pinned": sum(1 for n in self._hotset_per_layer if n),
            "pinned_experts": sum(self._hotset_per_layer),
            "pinned_entries": len(self._hotset_keys),
            "pinned_bytes": self._hotset_bytes,
            "pin_budget": min(self._hotset_budget, self._exp_cache_budget),
            "cache_budget": self._exp_cache_budget,
            "lru_region": self._exp_cache_budget - self._hotset_bytes,
            "pin_seconds": self._hotset_pin_seconds,
            "pin_stats": dict(self._hotset_pin_stats),
            "table_trace_hit_rate": table_hit,
            "table_trace_byte_hit_rate": table_byte_hit,
        }

    def hotset_report(self) -> str:
        """Human-readable ledger (printed once at pin time)."""
        info = self.hotset_info()
        per = info["per_layer"]
        if not per:
            return "hotset: disabled"
        mib = 2**20
        hit = info["table_trace_hit_rate"]
        hit_s = "?" if hit is None else f"{hit * 100:.1f}%"
        return (
            f"hotset: table={os.path.basename(info['table'])} "
            f"mode={info['mode']} "
            f"experts/layer={min(per)}..{max(per)} "
            f"(pinned {info['pinned_experts']} experts = "
            f"{info['pinned_entries']} cache entries, "
            f"{info['pinned_bytes'] / mib:.0f} MiB packed, "
            f"pin {info['pin_seconds']:.1f}s); "
            f"budget {info['cache_budget'] / mib:.0f} MiB "
            f"(pin {info['pin_budget'] / mib:.0f} + LRU region "
            f"{info['lru_region'] / mib:.0f} MiB); "
            f"table replay hit {hit_s}"
        )


__all__ = ["HotsetResidentGGUF"]
