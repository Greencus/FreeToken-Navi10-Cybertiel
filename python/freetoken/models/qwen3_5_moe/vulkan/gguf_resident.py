"""Streaming decoder for a ``qwen35moe`` GGUF on the Vulkan backend.

``ResidentGGUF`` derives from ``ResidentNVFP4`` to reuse its proven decode
machinery - the VRAM expert LRU (``_exp_cache``/``_pool_get``/``_cache_put``),
host pack cache, chunked prefill, and the base class's GDN/full-attention
steps - and replaces ONLY the weight-source layer:

- ``_open``: config + tensors come from ``GgufWeightProvider`` (one mmap'd
  GGUF) instead of a Ressl safetensors directory;
- ``_pin_all`` / ``_pinned_affine``: dense matrices pin as BF16 (or their
  native FP16/FP32 for k/v and the tiny GDN gate params) and dispatch to the
  matching resident kernel;
- ``_expert_affine[_batch]``: routed experts run through the PACKED K-quant
  lane by default - the GGUF slab (Q5_K/Q4_K/Q8_0) is uploaded verbatim and
  unpacked in-shader by ``linear_q4k/q5k/q8_0_resident_io``, so the host
  never dequantizes and VRAM holds ~1.4 MB of packed bytes per expert pair
  instead of ~6 MB of BF16. ``FTV_GGUF_PACKED=0`` forces the original
  host-dequant BF16 path (``_expert_handle``'s else-branch), which stays
  byte-for-byte the reference the golden tests compare against;
- ``_moe_step`` / ``_moe_chunk``: base-style routing with those experts.

Banks without an in-shader unpacker (``bank.qtype is None``) fall back to the
host path per bank, so an unusual quant mix still runs.

The fixed-slot arena and speculative packer are NVFP4-pack-only and forced
off during init.  The batched ``decode_step_batched`` IS available here via
``GgufBatchedMixin`` (K-quant record overrides; see gguf_batched.py), on by
default and disabled with ``FTV_GGUF_BATCHED=0`` (``generate(batched=True)``
then downgrades to the sequential path with a printed warning).
"""
from __future__ import annotations

import os
import pathlib
from typing import Any

import torch
import torch.nn.functional as F

from freetoken.models.gguf.packed import packed_tensor

from .gguf_batched import GgufBatchedMixin
from .gguf_provider import ExpertBank, GgufWeightProvider, gguf_config
from .nvfp4_resident import ResidentNVFP4, _translate

# Resident K-quant GEMV bindings per shader qtype (see
# freetoken/kernel/vulkan/ext_module.cpp, "KQuantPC" above them).
_KQUANT_OP = {
    "q4_k": "linear_q4k_resident_io",
    "q5_k": "linear_q5k_resident_io",
    "q8_0": "linear_q8_0_resident_io",
}

# Expert-LRU default for this decoder when FTV_EXP_CACHE_MB is unset; see
# the measurement note in __init__ (VRAM headroom on an 8 GB card).
_GGUF_CACHE_MB_DEFAULT = 2048


class _GgufPinned:
    """One resident dense matrix pinned in VRAM, with its source dtype."""

    __slots__ = ("handle", "rows", "cols", "dtype")

    def __init__(self, ext, tensor: torch.Tensor) -> None:
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"expected bf16/fp16/fp32 pin tensor, got {tensor.dtype}")
        contiguous = tensor.contiguous()
        self.handle = ext.upload_resident(contiguous)
        self.rows = contiguous.shape[-2]
        self.cols = contiguous.shape[-1]
        self.dtype = tensor.dtype


class ResidentGGUF(GgufBatchedMixin, ResidentNVFP4):
    """Vulkan decoder over one GGUF file (see module doc)."""

    _open_desc = "GGUF"

    def __init__(
        self,
        model_path: str | pathlib.Path,
        ext=None,
        max_len: int = 4096,
        kv_dtype: str = "fp32",
    ) -> None:
        # Packed K-quant lane on by default: the routed experts go to the GPU
        # as GGUF bytes and unpack in-shader. FTV_GGUF_PACKED=0 restores the
        # host-dequant BF16 path (the golden reference). Read before the base
        # init so _pin_gguf sees the same mode when it pins.
        self._packed_kq = os.environ.get("FTV_GGUF_PACKED", "1").strip() != "0"
        # Batched decode path (gguf_batched.py). ON by default: the
        # batch_kquant_io hazard fix (Y is WRITE for every kind; "plain" and
        # "gateup" used to declare it READ while the shaders wrote it) made
        # it reproduce the sequential greedy output, and it measures ~3.4x
        # faster (fork2 gate: seq-vs-batched maxdiff 4.530e-06 under the
        # 1e-5 bound, batched-vs-batched2 bit-exact 0.0). FTV_GGUF_BATCHED=0
        # restores the sequential decoder (and pins every fp16 weight at its
        # native dtype); generate(batched=True) then downgrades with a
        # printed warning.
        # Read before the base init so _pin_gguf's fp16->fp32 widening of
        # the k/v pins matches the mode the decode loop will run in.
        self._gguf_batched = (
            os.environ.get("FTV_GGUF_BATCHED", "1").strip() != "0"
        )
        # Arena/spec pack drive NVFP4-packed-only kernels; force them off for
        # this class even when the ambient env opts in (restored right after,
        # they are read once during the base init).
        saved = {k: os.environ.get(k) for k in ("FTV_EXP_ARENA", "FTV_SPEC_PACK")}

        os.environ["FTV_EXP_ARENA"] = "0"
        os.environ["FTV_SPEC_PACK"] = "0"
        # The code-wide expert-cache default (4096) overcommits an 8 GB card
        # here: 3781 MiB of pinned dense + ~1 GiB of per-layer scratch + 4096
        # + ~0.8 GiB of driver overhead saturates VRAM at 8167 of 8176 MiB
        # and RADV silently spills the overflow into GTT. 2048 saturates at
        # ~7010 MiB with ~1.1 GiB of headroom; pass FTV_EXP_CACHE_MB
        # explicitly to trade headroom for hit rate.
        saved_cache = os.environ.get("FTV_EXP_CACHE_MB")
        if not (saved_cache or "").strip():
            os.environ["FTV_EXP_CACHE_MB"] = str(_GGUF_CACHE_MB_DEFAULT)
        try:
            super().__init__(model_path, ext, max_len=max_len, kv_dtype=kv_dtype)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if saved_cache is None:
                os.environ.pop("FTV_EXP_CACHE_MB", None)
            else:
                os.environ["FTV_EXP_CACHE_MB"] = saved_cache

    # -- weight source ----------------------------------------------------
    def _open(self, model_dir):
        return gguf_config(model_dir), GgufWeightProvider(model_dir)

    # -- pinning ----------------------------------------------------------
    def _pin_gguf(self, stem: str) -> None:
        rstem = _translate(stem)
        tensor = self.tensors.get(rstem + ".weight")
        if self._gguf_batched and tensor.dtype == torch.float16:
            # The batch recorders have no fp16 op (batch_bf16_io /
            # batch_fp32_io only); widening here is lossless and the
            # fp16/fp32 resident shaders share the fma chain + reduction,
            # so batch numerics stay bit-identical to the sequential
            # fp16 kernel (the k/v projections are the only fp16 pins).
            tensor = tensor.float()
        self.pinned[rstem] = _GgufPinned(self.ext, tensor)

    def _pin_all(self) -> None:
        cfg = self.config
        for i, layer_type in enumerate(cfg.layer_types):
            base = f"language_model.model.layers.{i}"
            if layer_type == "linear_attention":
                stems = (
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.in_proj_a",
                    "linear_attn.in_proj_b",
                    "linear_attn.out_proj",
                )
            else:
                stems = (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                )
            for suffix in stems:
                self._pin_gguf(f"{base}.{suffix}")
            for suffix in (
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj",
                "mlp.shared_expert_gate",
            ):
                self._pin_gguf(f"{base}.{suffix}")
            # Routed experts: stay packed in the file; the packed K-quant
            # lane uploads the slab verbatim (FTV_GGUF_PACKED=0: BF16-on-miss).
            for suffix, proj in (
                ("mlp.switch_mlp.gate_proj", "gate"),
                ("mlp.switch_mlp.up_proj", "up"),
                ("mlp.switch_mlp.down_proj", "down"),
            ):
                bank = self.tensors.expert_bank(i, proj)
                stem = f"{base}.{suffix}"
                self.expert_mats[stem] = bank
                self._mat_stems[id(bank)] = stem
                self._mat_layer[id(bank)] = i
        self._pin_gguf("language_model.lm_head")

    # -- pinned kernels (dtype dispatch) ----------------------------------
    def _gguf_kernel(
        self, h_x: int, slot: _GgufPinned, h_y: int, t: int
    ) -> None:
        ext = self.ext
        if slot.dtype == torch.bfloat16:
            ext.linear_bf16_resident_io(
                h_x, slot.handle, 0, h_y, t, slot.rows, slot.cols, False
            )
        elif slot.dtype == torch.float16:
            ext.linear_fp16_resident_io(
                h_x, slot.handle, 0, h_y, t, slot.rows, slot.cols, False
            )
        else:
            ext.linear_resident_io(
                h_x, slot.handle, 0, h_y, t, slot.rows, slot.cols, False
            )

    def _pinned_affine(self, x: torch.Tensor, stem: str) -> torch.Tensor:
        import time as _time

        slot = self.pinned[_translate(stem)]
        ext = self.ext
        t0 = _time.perf_counter()
        ext.update_resident(self.h_x, x.contiguous().float())
        self._bump("pin_up", _time.perf_counter() - t0)
        t0 = _time.perf_counter()
        self._gguf_kernel(self.h_x, slot, self.h_y, 1)
        self._bump("pin_kernel", _time.perf_counter() - t0)
        t0 = _time.perf_counter()
        out = ext.download_resident(self.h_y, [slot.rows])
        self._bump("pin_down", _time.perf_counter() - t0)
        return out

    def _pinned_affine_batch(self, x: torch.Tensor, stem: str) -> torch.Tensor:
        import time as _time

        slot = self.pinned[_translate(stem)]
        tc = x.shape[0]
        ext = self.ext
        h_x = ext.allocate_resident(tc * slot.cols * 4)
        h_y = ext.allocate_resident(tc * slot.rows * 4)
        try:
            t0 = _time.perf_counter()
            ext.update_resident(h_x, x.contiguous().float())
            self._gguf_kernel(h_x, slot, h_y, tc)
            self._bump("pin_batch_kernel", _time.perf_counter() - t0)
            t0 = _time.perf_counter()
            out = ext.download_resident(h_y, [tc, slot.rows])
            self._bump("pin_batch_down", _time.perf_counter() - t0)
            return out
        finally:
            ext.free_resident(h_x)
            ext.free_resident(h_y)

    # -- routed experts: packed K-quant lane + the NVFP4 VRAM LRU ---------
    def _packed_qtype(self, matrix: Any) -> str | None:
        """Shader selector for this bank, or None when the host path runs.

        None for a disabled lane (``FTV_GGUF_PACKED=0``), a bank with no
        in-shader unpacker (``qtype is None``: non-K-quant or K not a
        multiple of 256), so those matrices keep dequantizing on the host.
        """
        if not self._packed_kq:
            return None
        return getattr(matrix, "qtype", None)

    def _expert_handle(self, matrix: Any, expert: int) -> tuple[int, int, int]:
        """VRAM handle of one expert weight via the resident LRU.

        Packed lane: the GGUF slab (uint8) uploads verbatim and the cache
        accounts PACKED bytes (~0.56-1.1 MiB per matrix, 3.5x smaller than
        the BF16 copy it replaces). Host lane: unchanged BF16 upload.

        Returns ``(h_w, rows, cols)``; the handle stays alive as a cache
        entry (or is freed by the caller when caching is disabled).
        """
        if not isinstance(matrix, ExpertBank):
            raise TypeError(
                "ResidentGGUF expects GGUF expert banks, got "
                f"{type(matrix).__name__}"
            )
        import time as _time

        ext = self.ext
        qtype = self._packed_qtype(matrix)
        key = (id(matrix), expert)
        entry = self._exp_cache.get(key)
        if entry is not None:
            self._cache_touch(key)
            self._cache_freq(key)
            self.stat_calls["cache_hit"] = self.stat_calls.get("cache_hit", 0) + 1
            return entry[0], matrix.rows, matrix.cols
        self.stat_calls["cache_miss"] = self.stat_calls.get("cache_miss", 0) + 1
        if qtype is not None:
            t0 = _time.perf_counter()
            slab = self.tensors.expert_packed(
                matrix.layer, matrix.proj, expert
            )
            weight = packed_tensor(slab)
            if qtype == "q8_0":
                # The Q8_0 shader rebuilds unaligned dwords and reads up to
                # 4 B past the last packed row; give the buffer that slack.
                weight = torch.cat([weight, weight.new_zeros(4)])
            self._bump("exp_packed", _time.perf_counter() - t0)
        else:
            t0 = _time.perf_counter()
            weight = self.tensors.expert_bf16(
                matrix.layer, matrix.proj, expert
            )
            self._bump("exp_dequant", _time.perf_counter() - t0)
        w_bytes = weight.numel() * weight.element_size()
        t0 = _time.perf_counter()
        pooled = self._pool_get(w_bytes, 0)
        if pooled is not None:
            h_w = pooled[0]
            ext.update_resident(h_w, weight)
        else:
            h_w = ext.upload_resident(weight)
        self._bump("exp_up", _time.perf_counter() - t0)
        if self._exp_cache_budget <= 0:
            return h_w, matrix.rows, matrix.cols  # caller frees
        entry = (h_w, 0, w_bytes, w_bytes, 0)
        self._cache_put(key, entry)
        self._cache_freq(key)
        return h_w, matrix.rows, matrix.cols

    def _expert_affine(
        self, x: torch.Tensor, matrix: Any, expert: int
    ) -> torch.Tensor:
        import time as _time

        ext = self.ext
        t0 = _time.perf_counter()
        ext.update_resident(self.h_x, x.contiguous().float())
        self._bump("exp_x", _time.perf_counter() - t0)
        uncached = self._exp_cache_budget <= 0
        qtype = self._packed_qtype(matrix)
        h_w, rows, cols = self._expert_handle(matrix, expert)
        try:
            t0 = _time.perf_counter()
            if qtype is None:
                ext.linear_bf16_resident_io(
                    self.h_x, h_w, 0, self.h_y, 1, rows, cols, False
                )
            else:
                getattr(ext, _KQUANT_OP[qtype])(
                    self.h_x, h_w, self.h_y, 1, rows, cols, 1.0, 0
                )
            self._bump("exp_kernel", _time.perf_counter() - t0)
            t0 = _time.perf_counter()
            out = ext.download_resident(self.h_y, [rows])
            self._bump("exp_down", _time.perf_counter() - t0)
            return out
        finally:
            if uncached:
                ext.free_resident(h_w)

    def _expert_affine_batch(
        self, x: torch.Tensor, matrix: Any, expert: int
    ) -> torch.Tensor:
        import time as _time

        ext = self.ext
        tc = x.shape[0]
        uncached = self._exp_cache_budget <= 0
        qtype = self._packed_qtype(matrix)
        h_w, rows, cols = self._expert_handle(matrix, expert)
        h_x = ext.allocate_resident(tc * cols * 4)
        h_y = ext.allocate_resident(tc * rows * 4)
        try:
            t0 = _time.perf_counter()
            ext.update_resident(h_x, x.contiguous().float())
            if qtype is None:
                ext.linear_bf16_resident_io(h_x, h_w, 0, h_y, tc, rows, cols, False)
            else:
                getattr(ext, _KQUANT_OP[qtype])(
                    h_x, h_w, h_y, tc, rows, cols, 1.0, 0
                )
            self._bump("exp_batch_kernel", _time.perf_counter() - t0)
            t0 = _time.perf_counter()
            out = ext.download_resident(h_y, [tc, rows])
            self._bump("exp_batch_down", _time.perf_counter() - t0)
            return out
        finally:
            ext.free_resident(h_x)
            ext.free_resident(h_y)
            if uncached:
                ext.free_resident(h_w)

    # -- MoE (packed K-quant experts, no fused NVFP4 dispatch) ------------
    def _moe_step(self, i: int, h: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        router_w = self._dense(f"{base}.mlp.gate.weight")
        logits = F.linear(h.float(), router_w.float()).float()
        probs = torch.softmax(logits, dim=-1)
        values, index = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)

        shared_g = self._pinned_affine(h, f"{base}.mlp.shared_expert.gate_proj")
        shared_u = self._pinned_affine(h, f"{base}.mlp.shared_expert.up_proj")
        shared = self._pinned_affine(
            F.silu(shared_g) * shared_u,
            f"{base}.mlp.shared_expert.down_proj",
        )
        gate_scalar = self._pinned_affine(
            h, f"{base}.mlp.shared_expert_gate"
        )
        out = torch.sigmoid(gate_scalar) * shared

        gate_mat = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        up_mat = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        down_mat = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        for slot, expert in enumerate(index.tolist()):
            gate = self._expert_affine(h, gate_mat, expert)
            up = self._expert_affine(h, up_mat, expert)
            down = self._expert_affine(F.silu(gate) * up, down_mat, expert)
            out = out + values[slot].item() * down
        return out

    def _moe_chunk(self, i: int, h_chunk: torch.Tensor) -> torch.Tensor:
        """Chunked prefill MoE: one T>1 kernel per distinct (expert, proj)."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        tc = h_chunk.shape[0]
        router_w = self._dense(f"{base}.mlp.gate.weight")
        logits = F.linear(h_chunk.float(), router_w.float()).float()
        probs = torch.softmax(logits, dim=-1)
        values, index = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)

        shared_g = self._pinned_affine_batch(
            h_chunk, f"{base}.mlp.shared_expert.gate_proj"
        )
        shared_u = self._pinned_affine_batch(
            h_chunk, f"{base}.mlp.shared_expert.up_proj"
        )
        shared = self._pinned_affine_batch(
            F.silu(shared_g) * shared_u,
            f"{base}.mlp.shared_expert.down_proj",
        )
        gate_scalar = self._pinned_affine_batch(
            h_chunk, f"{base}.mlp.shared_expert_gate"
        )
        out = torch.sigmoid(gate_scalar) * shared

        gate_mat = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        up_mat = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        down_mat = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        exp_slots: list[list[int]] = index.tolist()
        val_slots: list[list[float]] = values.tolist()
        groups: dict[int, list[int]] = {}
        for t in range(tc):
            for expert in exp_slots[t]:
                groups.setdefault(expert, []).append(t)
        for expert, rows in groups.items():
            xg = h_chunk[rows]
            gate = self._expert_affine_batch(xg, gate_mat, expert)
            up = self._expert_affine_batch(xg, up_mat, expert)
            down = self._expert_affine_batch(F.silu(gate) * up, down_mat, expert)
            for j, t in enumerate(rows):
                wsum = 0.0
                for slot, e in enumerate(exp_slots[t]):
                    if e == expert:
                        wsum += val_slots[t][slot]
                out[t] = out[t] + wsum * down[j]
        return out

    # -- guards ------------------------------------------------------------
    def generate(self, *args, batched: bool = False, **kwargs):
        if batched and not self._gguf_batched:
            print(
                "FTV_GGUF_BATCHED=0: GGUF batched decode disabled, "
                "running the sequential path",
                flush=True,
            )
            batched = False
        return super().generate(*args, batched=batched, **kwargs)


__all__ = ["ResidentGGUF"]
