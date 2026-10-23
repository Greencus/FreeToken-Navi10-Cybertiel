"""Batched decode recorders for the GGUF (packed K-quant) resident decoder.

``ResidentGGUF`` inherits the base ``ResidentNVFP4.decode_step_batched``
walk (GPU-resident GDN batch A, batched attention in/out, one command
buffer submit per layer, host router) and swaps in K-quant-aware recorders
through this mixin (first in the MRO):

- ``_record_gdn_in`` / ``_record_attn_in`` dispatch each dense pin by its
  SOURCE dtype (bf16 -> ``batch_bf16_io``, fp32 -> ``batch_fp32_io``); the
  GGUF's fp32 GDN a/b pins and fp32-widened k/v pins would otherwise hit
  the base recorders' bf16-only dispatches.
- ``_record_moe_in`` records the 8 routed experts through
  ``batch_moe_kquant_record`` (one pybind crossing: 8 in-shader gateup-silu
  on stacked gate+up packed bundles + 8 down accumulates with the router
  value folded into the per-expert acc gscale) plus the shared-expert tail,
  so one decode token drops from ~960 sync upload/kernel/download round
  trips to ~2-4 command buffer submits per layer.

On top of that, ``GgufArenaMixin`` (the class's first base, see
``gguf_arena.py``) swaps the per-expert resident-LRU record for the
fixed-slot K-quant arena + fused crossing: ``_record_moe_in`` below opens
with the arena gate and routes eligible layers to ``_kq_record_moe``, so
this mixin's record body only runs for ineligible/MTP layers (and for
``FTV_GGUF_ARENA=0``).

Contract (enforced loudly, never silently degraded): packed lane on
(``FTV_GGUF_PACKED=1``), resident cache budget > 0, NVFP4 spec pack off
(forced during ``ResidentGGUF.__init__``), NVFP4 GPU router off (falls out
of ``FTV_EXP_ARENA`` being forced off; the GGUF arena has its own
``FTV_GGUF_GPU_ROUTER`` knob).  Env ``FTV_GGUF_BATCHED=0`` disables the
whole path; ``generate(batched=True)`` then downgrades to the sequential
decoder with a printed warning so the correctness gates pass in either
mode.
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from freetoken.models.gguf.packed import packed_gate_up_expert, packed_tensor

from .gguf_arena import GgufArenaMixin
from .gguf_provider import ExpertBank
from .nvfp4_resident import _translate


class GgufBatchedMixin(GgufArenaMixin):
    """K-quant record overrides for ``ResidentNVFP4.decode_step_batched``.

    Only the three per-layer record hooks are replaced; the batch walk,
    GPU GDN kernels, attention host math, lm-head batch and the env-gated
    sequential/MTP/prefill paths all stay the base implementations.  The
    methods below assume the base ``__init__`` has run (resident scratch
    buffers, ``pinned``/``expert_mats``, the ``_exp_cache``/``_pool_get``/
    ``_cache_put`` LRU, ``_packed_qtype``).
    """

    # -- dtype-dispatching pinned matvec (T=1, inside an open batch) ------
    def _batch_pinned_matmul(
        self, x_handle: int, slot: Any, y_handle: int
    ) -> None:
        """Record y = x @ slot^T for one GGUF pin by its source dtype."""
        ext = self.ext
        if slot.dtype == torch.bfloat16:
            ext.batch_bf16_io(
                x_handle, slot.handle, slot.handle, y_handle,
                1, slot.rows, slot.cols, False,
            )
        elif slot.dtype == torch.float32:
            ext.batch_fp32_io(
                x_handle, slot.handle, slot.handle, y_handle,
                1, slot.rows, slot.cols, False,
            )
        elif slot.dtype == torch.float16:
            # ResidentGGUF widens every fp16 pin to fp32 at init when the
            # batched path is on (no batch_fp16 op exists; the fp16/fp32
            # resident shaders share the fma chain, so widening is
            # bit-identical) - reaching this arm means that did not run.
            raise RuntimeError(
                "GGUF batched decode: fp16 pin reached a batch recorder; "
                "k/v pins must be widened to fp32 when FTV_GGUF_BATCHED=1"
            )
        else:
            raise RuntimeError(
                f"GGUF batched decode: unsupported pin dtype {slot.dtype}"
            )

    # -- record overrides (base signature-compatible) --------------------
    def _record_gdn_in(
        self, i: int, ext: Any, download: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Record the 4 linear-attention in-projections into the open
        batch with dtype dispatch (qkv/z bf16, a/b fp32 on this family);
        see the base method for the download=False (GPU-resident GDN)
        contract."""
        base = f"language_model.model.layers.{i}"
        qkv_s = self.pinned[_translate(f"{base}.linear_attn.in_proj_qkv")]
        z_s = self.pinned[_translate(f"{base}.linear_attn.in_proj_z")]
        a_s = self.pinned[_translate(f"{base}.linear_attn.in_proj_a")]
        b_s = self.pinned[_translate(f"{base}.linear_attn.in_proj_b")]
        self._batch_pinned_matmul(self.h_x_fused, qkv_s, self.h_qkv)
        self._batch_pinned_matmul(self.h_x_fused, z_s, self.h_z)
        self._batch_pinned_matmul(self.h_x_fused, a_s, self.h_a)
        self._batch_pinned_matmul(self.h_x_fused, b_s, self.h_b)
        if not download:
            empty = torch.empty(0)
            return empty, empty, empty, empty
        qkv_t = torch.empty(qkv_s.rows)
        z_t = torch.empty(z_s.rows)
        a_t = torch.empty(a_s.rows)
        b_t = torch.empty(b_s.rows)
        ext.batch_download(self.h_qkv, qkv_t)
        ext.batch_download(self.h_z, z_t)
        ext.batch_download(self.h_a, a_t)
        ext.batch_download(self.h_b, b_t)
        return qkv_t, z_t, a_t, b_t

    def _record_attn_in(
        self, i: int, ext: Any, download: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Record q/k/v projections + downloads (full attention) with
        dtype dispatch (q bf16, k/v fp32 after the init-time widening).
        ``download=False`` (FTV_GGUF_GPU_ATTN) keeps all three on the GPU -
        the device prep kernel consumes them in the same batch - and
        returns empties, mirroring the ``_record_gdn_in`` contract."""
        base = f"language_model.model.layers.{i}"
        q_s = self.pinned[_translate(f"{base}.self_attn.q_proj")]
        k_s = self.pinned[_translate(f"{base}.self_attn.k_proj")]
        v_s = self.pinned[_translate(f"{base}.self_attn.v_proj")]
        self._batch_pinned_matmul(self.h_x_fused, q_s, self.h_qg)
        self._batch_pinned_matmul(self.h_x_fused, k_s, self.h_k)
        self._batch_pinned_matmul(self.h_x_fused, v_s, self.h_v)
        if not download:
            empty = torch.empty(0)
            return empty, empty, empty
        qg_t = torch.empty(q_s.rows)
        k_t = torch.empty(k_s.rows)
        v_t = torch.empty(v_s.rows)
        ext.batch_download(self.h_qg, qg_t)
        ext.batch_download(self.h_k, k_t)
        ext.batch_download(self.h_v, v_t)
        return qg_t, k_t, v_t

    def _record_moe_in(
        self, i: int, h: torch.Tensor, ext: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """K-quant MoE record: host router/topk (no GPU round trips), then
        the 8 routed experts + shared-expert tail into the open batch.

        The caller has already uploaded ``h`` into ``self.h_x_fused`` and
        filled ``h_y_acc``; the down products accumulate
        resident-to-resident with the renormalized router value folded
        into each expert's acc gscale (acc += v_e * silu(x@Wg_e^T) *
        (x@Wu_e^T) @ Wd_e^T), so the host reads back exactly (acc, shared
        down, gate scalar) - the same triple the base NVFP4 record
        returns, and the base batch walk consumes it unchanged.
        """
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        if not self._packed_kq:
            raise RuntimeError(
                "GGUF batched decode needs the packed K-quant lane "
                "(FTV_GGUF_PACKED=1); set FTV_GGUF_BATCHED=0 to run the "
                "sequential host-dequant path"
            )
        if self._exp_cache_budget <= 0:
            raise RuntimeError(
                "GGUF batched decode needs a resident expert cache "
                "(FTV_EXP_CACHE_MB>0): expert handles must stay alive "
                "across the open batch"
            )
        if self._kq_arena and self._kq_layer_ok.get(i, False):
            # Fixed-slot K-quant arena (docs/vulkan/gpu-resident-token.md):
            # one fused crossing over arena slots, experts not in the
            # resident LRU at all. Ineligible layers, the MTP block-40
            # walk and FTV_GGUF_ARENA=0 fall through to the LRU record
            # below unchanged.
            return self._kq_record_moe(i, h, ext)
        gate_mat = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        up_mat = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        down_mat = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        router_w = self._dense(f"{base}.mlp.gate.weight")
        logits = F.linear(h.float(), router_w.float()).float()
        probs = torch.softmax(logits, dim=-1)
        values, index = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)
        vlist = values.tolist() if values.dim() == 1 else values.tolist()[0]
        elist = index.tolist() if index.dim() == 1 else index.tolist()[0]

        # Fetch ALL routed handles BEFORE recording anything: a mid-record
        # _cache_put eviction may pool (and, past the pool cap, free) a
        # handle this batch already recorded a copy/dispatch against;
        # fetch-then-record keeps every eviction/free inside the
        # pre-record window, where no command references the victim yet.
        gu_w: list[int] = []
        gu_sc: list[float] = []
        dn_w: list[int] = []
        dn_sc: list[float] = []
        pending: list[tuple[int, torch.Tensor]] = []
        for e, expert in enumerate(elist):
            h_gu, _gr, _gc = self._gguf_gu_handle(
                gate_mat, up_mat, expert, pending)
            h_dn, _dr, _dc = self._gguf_dn_handle(down_mat, expert, pending)
            gu_w.append(h_gu)
            dn_w.append(h_dn)
            gu_sc.append(1.0)  # K-quant packs its scales in-block
            dn_sc.append(float(vlist[e]))  # router value -> acc gscale
        # MRU-touch the fetched set so the NEXT layer's misses cannot
        # evict entries whose dispatches this layer records next.
        for expert in elist:
            self._cache_touch((id(gate_mat), id(up_mat), expert))
            self._cache_touch((id(down_mat), expert))
        if pending:
            # Drain this row's misses with one union copy: the fetch loop
            # recorded no GPU work, so command order is unchanged and the
            # union check emits the same barrier the first conflicting
            # sequential copy did (none when the row touches nothing prior).
            t0 = time.perf_counter()
            ext.batch_upload_multi(
                [h for h, _ in pending],
                [w for _, w in pending],
                [0] * len(pending),
            )
            self._bump("exp_up", time.perf_counter() - t0)
        gu_rows, gu_cols = gate_mat.rows, gate_mat.cols
        dn_rows, dn_cols = down_mat.rows, down_mat.cols
        q_g = self._packed_qtype(gate_mat)
        q_d = self._packed_qtype(down_mat)
        acc_t = torch.empty(cfg.hidden_size)
        sd_t = torch.empty(cfg.hidden_size)
        gsc = self.pinned[_translate(f"{base}.mlp.shared_expert_gate")]
        gsc_t = torch.empty(gsc.rows)
        if q_d == q_g:
            # One qtype per layer (true for every qwen35moe ICE mix: the
            # whole block shares one K-quant format) -> one pybind
            # crossing for the 8 gateup-silu + 8 acc dispatches.  The
            # gateup bundle holds gate rows [0, N) then up rows [N, 2N);
            # each expert owns its own buffer, so every slot0 is 0.
            ext.batch_moe_kquant_record(
                self.h_x_fused, self.h_y_acc, q_g,
                self.h_silu_slots,
                gu_w, gu_sc, gu_rows, gu_cols,
                dn_w, dn_sc, dn_rows, dn_cols,
                [0] * len(gu_w), [0] * len(dn_w),
            )
        else:
            # Mixed gate/down quant format (never hit on this checkpoint;
            # the gate+up BUNDLE format itself is checked in
            # _gguf_gu_handle since the gateup shader takes one qtype):
            # fall back to per-side crossings with each side's shader.
            for e in range(len(gu_w)):
                ext.batch_kquant_io(
                    q_g, "gateup", gu_w[e], self.h_x_fused,
                    self.h_silu_slots[e], 1, gu_rows, gu_cols, gu_sc[e], 0,
                )
                ext.batch_kquant_io(
                    q_d, "acc", dn_w[e], self.h_silu_slots[e],
                    self.h_y_acc, 1, dn_rows, dn_cols, dn_sc[e], 0,
                )

        # Shared-expert tail: the same buffers and dispatches the NVFP4
        # batch_moe_record folds into its fused crossing, recorded here
        # separately because the K-quant record stops at the routed 16.
        sg = self.pinned[_translate(f"{base}.mlp.shared_expert.gate_proj")]
        su = self.pinned[_translate(f"{base}.mlp.shared_expert.up_proj")]
        sdp = self.pinned[_translate(f"{base}.mlp.shared_expert.down_proj")]
        self._batch_pinned_matmul(self.h_x_fused, sg, self.h_sg)
        self._batch_pinned_matmul(self.h_x_fused, su, self.h_su)
        self._batch_pinned_matmul(self.h_x_fused, gsc, self.h_gsc)
        ext.batch_swiglu_io(self.h_sg, self.h_su, self.h_x3, sg.rows)
        self._batch_pinned_matmul(self.h_x3, sdp, self.h_sdown)
        ext.batch_download(self.h_y_acc, acc_t)
        ext.batch_download(self.h_sdown, sd_t)
        ext.batch_download(self.h_gsc, gsc_t)
        return acc_t, sd_t, gsc_t

    # -- packed-expert handle fetch (resident LRU, in-batch upload) ------
    def _gguf_gu_handle(
        self,
        gate_mat: ExpertBank,
        up_mat: ExpertBank,
        expert: int,
        pending: list[tuple[int, torch.Tensor]] | None = None,
    ) -> tuple[int, int, int]:
        """Packed stacked gate+up bundle handle (2N rows, gate first).

        Returns ``(h_w, 2N, K)``; the bundle lives in the resident LRU
        under ``(id(gate), id(up), expert)`` - a distinct key from the
        per-matrix keys prefill's ``_expert_handle`` uses, so decode warm
        misses never clobber the sequential path's cache entries.
        """
        if not isinstance(gate_mat, ExpertBank) or not isinstance(
            up_mat, ExpertBank
        ):
            raise TypeError(
                "GGUF batched MoE expects ExpertBank pairs, got "
                f"{type(gate_mat).__name__}/{type(up_mat).__name__}"
            )
        qtype = self._packed_qtype(gate_mat)
        if qtype is None or self._packed_qtype(up_mat) != qtype:
            raise RuntimeError(
                f"layer {gate_mat.layer}: gate/up pack mismatch "
                f"({getattr(gate_mat, 'qtype', None)} vs "
                f"{getattr(up_mat, 'qtype', None)}); the gateup shader "
                f"unpacks one quant format per stacked bundle"
            )
        key = (id(gate_mat), id(up_mat), expert)
        entry = self._exp_cache.get(key)
        if entry is not None:
            self._cache_touch(key)
            self._cache_freq(key)
            self.stat_calls["cache_hit"] = (
                self.stat_calls.get("cache_hit", 0) + 1)
            return entry[0], gate_mat.rows * 2, gate_mat.cols
        self.stat_calls["cache_miss"] = (
            self.stat_calls.get("cache_miss", 0) + 1)
        t0 = time.perf_counter()
        slab = packed_gate_up_expert(
            self.tensors.index, gate_mat.layer, expert)
        if slab.shape[0] != gate_mat.rows * 2:
            raise RuntimeError(
                f"layer {gate_mat.layer} expert {expert}: gate+up bundle "
                f"has {slab.shape[0]} rows, expected {gate_mat.rows * 2} "
                f"(M-shaped MoE without up tensors cannot use the stacked "
                f"gateup shader)"
            )
        weight = packed_tensor(slab)
        if qtype == "q8_0":
            # The Q8_0 unpacker rebuilds unaligned dwords and reads up to
            # 4 B past the last packed row (same slack _expert_handle).
            weight = torch.cat([weight, weight.new_zeros(4)])
        self._bump("exp_packed", time.perf_counter() - t0)
        h_w = self._gguf_cache_resident(key, weight, pending)
        return h_w, gate_mat.rows * 2, gate_mat.cols

    def _gguf_dn_handle(
        self,
        down_mat: ExpertBank,
        expert: int,
        pending: list[tuple[int, torch.Tensor]] | None = None,
    ) -> tuple[int, int, int]:
        """Packed down-matrix handle via the resident LRU.

        Returns ``(h_w, rows, K=intermediate)``.  Keyed per matrix like
        the sequential ``_expert_handle`` but distinct from the bundle
        keys above.
        """
        if not isinstance(down_mat, ExpertBank):
            raise TypeError(
                "GGUF batched MoE expects ExpertBank down matrices, got "
                f"{type(down_mat).__name__}"
            )
        qtype = self._packed_qtype(down_mat)
        if qtype is None:
            raise RuntimeError(
                f"layer {down_mat.layer}: down bank has no in-shader "
                f"unpacker; the batched path is packed-K-quant only "
                f"(FTV_GGUF_BATCHED=0 runs the sequential host path)"
            )
        key = (id(down_mat), expert)
        entry = self._exp_cache.get(key)
        if entry is not None:
            self._cache_touch(key)
            self._cache_freq(key)
            self.stat_calls["cache_hit"] = (
                self.stat_calls.get("cache_hit", 0) + 1)
            return entry[0], down_mat.rows, down_mat.cols
        self.stat_calls["cache_miss"] = (
            self.stat_calls.get("cache_miss", 0) + 1)
        t0 = time.perf_counter()
        slab = self.tensors.expert_packed(
            down_mat.layer, down_mat.proj, expert)
        weight = packed_tensor(slab)
        if qtype == "q8_0":
            weight = torch.cat([weight, weight.new_zeros(4)])
        self._bump("exp_packed", time.perf_counter() - t0)
        h_w = self._gguf_cache_resident(key, weight, pending)
        return h_w, down_mat.rows, down_mat.cols

    def _gguf_cache_resident(
        self,
        key: tuple,
        weight: torch.Tensor,
        pending: list[tuple[int, torch.Tensor]] | None = None,
    ) -> int:
        """Insert ``weight`` into the resident LRU, uploading INSIDE the
        open batch (pooled-handle reuse or a fresh allocation + recorded
        copy), so a decode cache miss costs no sync submit/fence - the
        hazard barrier the batch engine emits before the first consumer
        dispatch orders the copy.

        With ``pending`` the destination is allocated and cached but the
        copy is deferred: the caller drains the collected misses with one
        ``batch_upload_multi`` right after its fetch loop, so ``weight``
        must stay alive in ``pending`` until then.
        """
        ext = self.ext
        w_bytes = weight.numel() * weight.element_size()
        t0 = time.perf_counter()
        pooled = self._pool_get(w_bytes, 0)
        if pooled is not None:
            h_w = pooled[0]
        else:
            h_w = ext.allocate_resident(w_bytes)
        if pending is None:
            ext.batch_upload(h_w, weight)
        else:
            pending.append((h_w, weight))
        self._bump("exp_up", time.perf_counter() - t0)
        self._cache_put(key, (h_w, 0, w_bytes, w_bytes, 0))
        self._cache_freq(key)
        return h_w


__all__ = ["GgufBatchedMixin"]
