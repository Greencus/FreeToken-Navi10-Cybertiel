"""Fixed-slot K-quant expert arena for the packed GGUF batched decoder.

Engine half of the GPU-resident token (``docs/vulkan/gpu-resident-token.md``
steps 1, 2, 4 and 6): decode MoE experts stop living in the VRAM LRU - they
live in a preallocated per-layer SLOT arena sized from the layer's own
packed row bytes, and the 8 routed experts of a layer are read through one
``batch_moe_kquant_fused_record`` crossing (2 dispatches instead of 16).

Why GGUF-owned (vs ``FTV_EXP_ARENA``, the NVFP4 arena):
  the base arena sizes its buffers from ``bank[0][0].shape`` as NVFP4
  PACKED columns plus scales VkBuffers (``_arena_slot_wf_b`` etc.).  A GGUF
  ``ExpertBank`` reports the LOGICAL K, and K-quant has no separate scales
  buffer at all - so the base's layout would be wrong for both the payload
  stride and the buffer set.  ``FTV_EXP_ARENA`` therefore stays forced off
  during ``ResidentGGUF.__init__`` and this mixin allocates its own buffers
  from ``ExpertBank.row_bytes``: slot ``s`` of layer ``i`` lives at
  ``s * slot_stride_bytes``, exactly the ``Ids[j] * rows_per_slot`` addressing
  the ``linear_q{4,5}k_{gateup8,acc8}_f32.comp`` shaders use.

Contract (loud, never silent):

- packed lane on (``FTV_GGUF_PACKED=1``) and the batched recorders on
  (``FTV_GGUF_BATCHED=1``); otherwise the arena never engages and the LRU
  path is untouched.
- a layer only joins the arena when gate/up/down are ONE in-shader qtype in
  {q4_k, q5_k} (the fused record takes a single ``qtype``, and there is no
  q8_0 gateup8/acc8 shader); ineligible layers print a warning and keep the
  LRU path.
- slot 0 is the reserved no-content sentinel (never assigned); its content
  is zero-filled at allocation so a misread slot is a deterministic zero,
  not stale bytes.
- every routed expert owns a live slot with correct content at record time
  (the pin fixpoint below); a routed expert with no slot is a
  ``RuntimeError``, never a silent slot-0 fallback.

Env (all read once at ``ResidentGGUF.__init__``):

``FTV_GGUF_ARENA``          1 (default) = arena on; 0 = the old per-expert
                            resident LRU path.
``FTV_GGUF_ARENA_SLOTS``    slots per layer (default 28 -> 2047.5 MiB total
                            for this checkpoint; slot 0 reserved).
``FTV_GGUF_FUSED``          1 (default) = one fused crossing per layer;
                            0 = the arena with the 16-dispatch record (the
                            A/B that separates "arena" from "fusion").
``FTV_GGUF_PREFETCH``       1 (default) = host router + slot assign +
                            queue-2 content uploads BEFORE batch 3 (step 6),
                            so a miss's DMA overlaps the record window;
                            0 = everything inside the open batch.
``FTV_GGUF_GPU_ROUTER``     0 (default) = host router (bit-identical to the
                            current output); 1 = the GPU-resident
                            RMS/router/topk lane feeding the same fused
                            record (step 3).  Off by default: its host
                            shadow router costs what the host router costs.
``FTV_GGUF_GPU_ROUTER_CHECK``  1 = cross-validate GPU vs host per layer.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Any

import torch

from freetoken.models.gguf.packed import packed_gate_up_expert, packed_tensor

from .gguf_provider import ExpertBank
from .nvfp4_resident import _translate

# The topk256 kernel's NO_SLOT word (0x7FFFFFFF) lives in the GPU-router
# MAP buffer only. It must NEVER be written into the record ids buffer:
# the shaders compute `Ids[j] * rows_per_slot` with no range check, so a
# sentinel id lands far outside the arena. The host-lane record below
# points unused ids at a routed slot with acc gscale 0 instead.

# Once the arena owns the expert bytes, a prefill/sequential arm that
# refills the LRU must not stack its bytes on top of it (arena + 2048 MiB
# LRU + pinned dense > 7.5 GB on the 8 GB card).  Cap, do not delete: the
# sequential fallback keeps a warm working set and the LRU machinery
# (_cache_touch/_pool_get) keeps its invariants.
_KQ_LRU_CAP_MB = 512

_KQ_FUSED_TYPES = ("q4_k", "q5_k")


class GgufArenaMixin:
    """K-quant arena + fused record, first in the MRO after ``ResidentGGUF``.

    Only the MoE record/entry hooks are overridden; the batch walk, GPU GDN,
    host attention and lm-head batch stay the base implementations.  Methods
    assume the base ``__init__`` has run (resident scratch, ``pinned`` /
    ``expert_mats``, ``_exp_cache`` / ``_pool_get`` / ``_cache_put``,
    ``_packed_qtype``, ``_host_pack_get``).
    """

    # ------------------------------------------------------------------
    # setup / teardown
    # ------------------------------------------------------------------
    def _kq_arena_setup(self) -> None:
        """Read the arena env and register per-layer facts (NO allocation).

        Allocation is lazy (``_kq_arena_ensure`` at the first decode token,
        after the prefill LRU has been reclaimed) so prefill keeps the whole
        VRAM budget for the LRU and the arena can never push a prefill past
        the 7.5 GB line.
        """
        self._kq_arena = False
        self._kq_prefetch_on = False
        self._kq_fused = True
        self._kq_allocated = False
        self._kq_slots = 0
        self._kq_bytes = 0
        self._arena_slot_rows_gu = 0   # rows per gate+up slot (2*N)
        self._arena_slot_rows_dn = 0   # rows per down slot (H)
        self._kq_slot_wf_b: list[int] = []   # per-layer gu slot stride (B)
        self._kq_slot_wd_b: list[int] = []   # per-layer down slot stride (B)
        self._kq_layer_ok: dict[int, bool] = {}
        self._kq_layer_c: dict[int, tuple[int, int, int, int]] = {}
        self._moe_pref: dict[int, tuple[list[int], list[float]]] = {}
        if not (self._packed_kq and self._gguf_batched):
            return
        env = os.environ
        if env.get("FTV_GGUF_ARENA", "1").strip() == "0":
            print("FTV_GGUF_ARENA=0: GGUF decode uses the per-expert "
                  "resident LRU (no fixed-slot arena)", flush=True)
            return
        try:
            slots = int(env.get("FTV_GGUF_ARENA_SLOTS", "28"))
        except (TypeError, ValueError):
            slots = 28
        slots = max(9, slots)
        if slots - 1 < self.config.num_experts_per_tok:
            raise RuntimeError(
                f"FTV_GGUF_ARENA_SLOTS={slots} (slot 0 reserved -> "
                f"{slots - 1} usable) < num_experts_per_tok="
                f"{self.config.num_experts_per_tok}: the fused record needs "
                "one slot per routed expert")
        self._kq_slots = slots
        # The base slot allocator (_arena_assign) hands out 1.._arena_n-1
        # and its capacity check reads _arena_n; this arena owns its own
        # slot count, so the two must agree or a fresh assign could index
        # past these smaller buffers (the base arena's own 32 slots are
        # never allocated here - FTV_EXP_ARENA stays forced off).
        self._arena_n = slots
        self._kq_fused = env.get("FTV_GGUF_FUSED", "1").strip() != "0"
        self._kq_prefetch_on = env.get("FTV_GGUF_PREFETCH", "1").strip() != "0"
        try:
            self._gpu_router = int(env.get("FTV_GGUF_GPU_ROUTER", "0")) != 0
        except (TypeError, ValueError):
            self._gpu_router = False
        try:
            self._gpu_router_check = int(
                env.get("FTV_GGUF_GPU_ROUTER_CHECK", "0")) != 0
        except (TypeError, ValueError):
            self._gpu_router_check = False
        bad: list[str] = []
        for i in range(len(self.config.layer_types)):
            if self._kq_probe_layer(i) is None:
                self._kq_layer_ok[i] = False
                bad.append(str(i))
            else:
                self._kq_layer_ok[i] = True
        if bad:
            print(
                "GGUF arena: layer(s) " + ",".join(bad) +
                " are not one in-shader qtype (q4_k/q5_k) across "
                "gate/up/down - those layers keep the resident LRU path; "
                "the rest use the arena", flush=True)
        if not any(self._kq_layer_ok.values()):
            self._kq_arena = False
            print("GGUF arena: no eligible layer, disabled (LRU path)",
                  flush=True)
            return
        if self._kq_prefetch_on and self._gpu_router:
            # The GPU-router lane has its own prewarm (the base
            # _gpu_router_prewarm + my async override) before batch_begin;
            # a second host prefetch would run the router twice.
            self._kq_prefetch_on = False
        self._kq_arena = True
        if self._gpu_router:
            self._kq_gpu_router_setup()

    def _kq_probe_layer(self, i: int) -> tuple[int, int, str] | None:
        """``(gu_slot_bytes, dn_slot_bytes, qtype)`` or None (ineligible)."""
        base = f"language_model.model.layers.{i}"
        try:
            gm = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
            um = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
            dm = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        except KeyError:
            return None
        if not (isinstance(gm, ExpertBank) and isinstance(um, ExpertBank)
                and isinstance(dm, ExpertBank)):
            return None
        qg, qu, qd = gm.qtype, um.qtype, dm.qtype
        if qg != qu or qg != qd or qg not in _KQ_FUSED_TYPES:
            return None
        if not (gm.row_bytes and dm.row_bytes):
            return None
        # The bundle layout and the gateup shader assume N gate rows then
        # N up rows (slot0 + n / slot0 + n + N), and the fused acc reads
        # its K as the silu vector length (= gate rows).
        if gm.cols != um.cols or gm.rows != um.rows:
            return None
        if dm.cols != gm.rows:
            return None
        gu_stride = (gm.rows + um.rows) * gm.row_bytes
        dn_stride = dm.rows * dm.row_bytes
        self._kq_layer_c[i] = (gm.rows, gm.cols, dm.rows, dm.cols)
        return gu_stride, dn_stride, qg

    def _kq_arena_ensure(self, i: int) -> None:
        """Allocate layer ``i``'s arena buffers + maps (idempotent, grows)."""
        if i in self._kq_layer_ok and not self._kq_layer_ok[i]:
            raise RuntimeError(
                f"layer {i} is not arena-eligible but reached the arena "
                "record path (routing/eligibility desync)")
        if i < len(self._arena_wf) and self._arena_wf[i]:
            return
        probe = self._kq_probe_layer(i)
        if probe is None:
            raise RuntimeError(
                f"layer {i} cannot join the K-quant arena on demand "
                "(probe failed after setup; bank set changed?)")
        gu_stride, dn_stride, qtype = probe
        ext = self.ext
        n = self._kq_slots
        # Slot strides in ROWS for the recorders: gateup slot0 = slot*2N
        # (gate N rows + up N rows), down slot0 = slot*H. The probe's byte
        # strides and these must agree across every layer (this family has
        # one shape per proj kind); a mismatch would silently address the
        # wrong rows, so refuse it loudly.
        gu_rows, _gu_cols, dn_rows, _dn_cols = self._kq_layer_c[i]
        gu_slot_rows = 2 * gu_rows
        dn_slot_rows = dn_rows
        if self._arena_slot_rows_gu not in (0, gu_slot_rows):
            raise RuntimeError(
                f"layer {i}: gate+up slot rows {gu_slot_rows} != the arena's "
                f"{self._arena_slot_rows_gu}; per-layer shape mix is not "
                "supported by the fixed-slot K-quant arena")
        if self._arena_slot_rows_dn not in (0, dn_slot_rows):
            raise RuntimeError(
                f"layer {i}: down slot rows {dn_slot_rows} != the arena's "
                f"{self._arena_slot_rows_dn}; per-layer shape mix is not "
                "supported by the fixed-slot K-quant arena")
        self._arena_slot_rows_gu = gu_slot_rows
        self._arena_slot_rows_dn = dn_slot_rows
        # Pad the lists with zeros so index alignment with the base's
        # per-layer lists (arena_ids/gs/map) survives the layer-40 growth
        # the MTP head does AFTER init. Slot byte strides are PER LAYER:
        # this checkpoint's Q5_K blocks (0-14) pack 1408 B/row and its
        # Q4_K blocks (15-39) 1152 B/row, so one scalar stride would give
        # the wrong upload offset to whichever group was ensured last.
        while len(self._arena_wf) <= i:
            self._arena_wf.append(0)
            self._arena_wd.append(0)
            self._arena_ids.append(0)
            self._arena_gs.append(0)
            self._arena_map.append(OrderedDict())
            self._kq_slot_wf_b.append(0)
            self._kq_slot_wd_b.append(0)
        while len(self._arena_map_dirty) <= i:
            self._arena_map_dirty.append(True)
        if self._arena_wf[i]:
            return  # already allocated by a previous ensure
        self._arena_wf[i] = ext.allocate_resident(gu_stride * n)
        self._arena_wd[i] = ext.allocate_resident(dn_stride * n)
        self._arena_ids[i] = ext.allocate_resident(8 * 4)   # u32[8]
        self._arena_gs[i] = ext.allocate_resident(16 * 4)   # float[16]
        self._kq_slot_wf_b[i] = gu_stride
        self._kq_slot_wd_b[i] = dn_stride
        self._kq_bytes += (gu_stride + dn_stride) * n + 8 * 4 + 16 * 4
        self.stat_calls["arena_layers"] = (
            self.stat_calls.get("arena_layers", 0) + 1)

    def _kq_arena_alloc_all(self) -> None:
        """Allocate every eligible layer and zero-fill the reserved slot 0.

        One batched zero upload (not one submit per buffer): slot 0 is never
        assigned, so a misread must be a deterministic zero rather than the
        mmap bytes a fresh VkBuffer would hold.
        """
        ext = self.ext
        for i in range(len(self.config.layer_types)):
            if self._kq_layer_ok.get(i):
                self._kq_arena_ensure(i)
        zeros_gu: list[int] = []
        zeros_dn: list[int] = []
        t_gu: list[torch.Tensor] = []
        t_dn: list[torch.Tensor] = []
        for i, ok in sorted(self._kq_layer_ok.items()):
            if not ok or not self._arena_wf[i]:
                continue
            zeros_gu.append(self._arena_wf[i])
            zeros_dn.append(self._arena_wd[i])
            t_gu.append(torch.zeros(self._kq_slot_wf_b[i], dtype=torch.uint8))
            t_dn.append(torch.zeros(self._kq_slot_wd_b[i], dtype=torch.uint8))
        if not zeros_gu:
            return
        ext.batch_begin()
        ext.batch_upload_multi(zeros_gu + zeros_dn, t_gu + t_dn,
                               [0] * (len(zeros_gu) + len(zeros_dn)))
        ext.batch_submit()

    def _kq_reclaim_lru(self) -> int:
        """Free the resident LRU + handle pool (bytes the arena replaces).

        Runs only OUTSIDE an open batch and only on the arena path: the
        arena never stores its experts in ``_exp_cache``, so anything in the
        cache or pool is prefill's or a sequential arm's dead weight.
        """
        ext = self.ext
        freed = 0
        for entry in list(self._exp_cache.values()):
            h_w, h_s = entry[0], entry[1]
            for h in (h_w, h_s):
                if h:
                    try:
                        ext.free_resident(h)
                    except RuntimeError:
                        pass
            freed += int(entry[3])
        self._exp_cache.clear()
        self._exp_cache_bytes = 0
        for key, lst in list(self._exp_pool.items()):
            for h_w, h_s in lst:
                for h in (h_w, h_s):
                    if h:
                        try:
                            ext.free_resident(h)
                        except RuntimeError:
                            pass
            del self._exp_pool[key]
        self._exp_pool_bytes = 0
        if freed or self.stat_calls.get("arena_reclaim"):
            self.stat_calls["arena_reclaim"] = (
                self.stat_calls.get("arena_reclaim", 0) + 1)
            self.stat_calls["arena_reclaim_bytes"] = (
                self.stat_calls.get("arena_reclaim_bytes", 0) + freed)
        return freed

    def _kq_decode_entry(self) -> None:
        """Per-token arena housekeeping (before any batch of the token).

        ONE-TIME (first decode token): drain any faulted queue-2 state
        before freeing handles, free the prefill/sequential resident LRU
        + handle pool (bytes the arena replaces), cap the LRU budget so a
        later refill cannot stack on top of the arena, and allocate +
        zero slot 0 of every eligible layer.  The LRU is reclaimed ONCE,
        not per token: after the arena engages the only things that
        re-enter ``_exp_cache`` are the MTP block-40 experts (and any
        sequential arm), whose warm working set must survive across
        tokens - a per-token reclaim would re-upload it every draft step.
        EVERY TOKEN: a routing prediction is only valid for the token
        that produced it.
        """
        self._moe_pref.clear()
        if self._kq_allocated:
            return
        ext = self.ext
        ext.batch_upload_async_drain()
        freed = self._kq_reclaim_lru()
        if self._exp_cache_budget > _KQ_LRU_CAP_MB:
            self._exp_cache_budget = _KQ_LRU_CAP_MB
        self._kq_arena_alloc_all()
        self._kq_allocated = True
        print(
            f"GGUF arena: {sum(1 for v in self._kq_layer_ok.values() if v)}"
            f" layers x {self._kq_slots} slots, "
            f"{self._kq_bytes / 2**20:.1f} MiB resident "
            f"(reclaimed {freed / 2**20:.1f} MiB of LRU, "
            f"LRU budget now {self._kq_cache_cap()} MiB, "
            f"fused={int(self._kq_fused)} "
            f"prefetch={int(self._kq_prefetch_on)} "
            f"gpu_router={int(self._gpu_router)})", flush=True)

    def _kq_cache_cap(self) -> int:
        return int(self._exp_cache_budget)

    # ------------------------------------------------------------------
    # packs + slot assignment
    # ------------------------------------------------------------------
    def _kq_pack_gu(self, gate_mat: ExpertBank, up_mat: ExpertBank,
                    expert: int) -> torch.Tensor:
        """Stacked gate+up packed bytes (cached in the host pack cache)."""
        key = (id(gate_mat), id(up_mat), expert)

        def build():
            slab = packed_gate_up_expert(
                self.tensors.index, gate_mat.layer, expert)
            if slab.shape[0] != gate_mat.rows * 2:
                raise RuntimeError(
                    f"layer {gate_mat.layer} expert {expert}: gate+up "
                    f"bundle has {slab.shape[0]} rows, expected "
                    f"{gate_mat.rows * 2} (M-shaped MoE without up tensors "
                    "cannot use the stacked gateup bundle)")
            return packed_tensor(slab), torch.empty(0, dtype=torch.int32)

        packed, _scales, _hit = self._host_pack_get(key, build)
        return packed

    def _kq_pack_dn(self, down_mat: ExpertBank, expert: int) -> torch.Tensor:
        """Down packed bytes (cached in the host pack cache)."""
        key = (id(down_mat), expert)

        def build():
            slab = self.tensors.expert_packed(
                down_mat.layer, down_mat.proj, expert)
            return packed_tensor(slab), torch.empty(0, dtype=torch.int32)

        packed, _scales, _hit = self._host_pack_get(key, build)
        return packed

    def _kq_assign_routed(self, i: int, elist: list[int]) -> list[tuple[int, int]]:
        """Pin loop: every routed expert owns a slot; return the misses.

        A re-assignment evicts at most one entry, which may itself be routed,
        so the loop strictly grows routed membership and <= 9 passes settle
        (same fixpoint argument as the base NVFP4 pin loop).  Stale pairs
        (the map no longer maps the expert to the slot it was given) are
        refused loudly: they would write content into a slot the record does
        not read, or read a slot that holds another expert.
        """
        mp = self._arena_map[i]
        pending: list[tuple[int, int]] = []
        for _pin in range(9):
            missing = [e for e in elist if mp.get(e) is None]
            if not missing:
                break
            for e in missing:
                pending.append((e, self._arena_assign(i, e)))
        stale = [(e, s) for e, s in pending if mp.get(e) != s]
        if stale:
            raise RuntimeError(
                f"layer {i}: arena pin produced stale (expert, slot) pairs "
                f"{stale}; current map "
                f"{{{', '.join(f'{e}:{mp.get(e)}' for e, _ in stale)}}}")
        gone = [e for e in elist if mp.get(e) is None]
        if gone:
            raise RuntimeError(
                f"layer {i}: routed expert(s) {gone} have no arena slot "
                f"(slots={self._kq_slots}, routed={len(elist)}); refusing "
                "silent slot-0 fallback")
        if self._kq_slots - 1 < len(elist):
            raise RuntimeError(
                f"layer {i}: arena has {self._kq_slots - 1} usable slots "
                f"(slot 0 reserved) but {len(elist)} routed experts need "
                "one each")
        return pending

    def _kq_upload_pending(self, i: int,
                           pending: list[tuple[int, int]],
                           *, async_: bool) -> int:
        """Write the packed content of ``pending`` (expert, slot) pairs.

        ``async_=True`` emits queue-2 copies OUTSIDE the open batch and
        returns the flush cookie (-1 when nothing is pending); the consumer
        batch records ``batch_cross_barrier`` at ``batch_begin`` and submits
        with ``batch_submit([cookie])`` (the NVFP4 prewarm protocol).
        ``async_=False`` drains them with one in-batch union copy.
        """
        if not pending:
            return -1
        t0 = time.perf_counter()
        base = f"language_model.model.layers.{i}"
        gm = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        um = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        dm = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        handles: list[int] = []
        tensors: list[torch.Tensor] = []
        offsets: list[int] = []
        for e, sidx in pending:
            if self._arena_map[i].get(e) != sidx:
                raise RuntimeError(
                    f"layer {i}: pending (expert {e}, slot {sidx}) is STALE "
                    f"(map has {self._arena_map[i].get(e)}); refusing a "
                    "stale-slot content write")
            fp = self._kq_pack_gu(gm, um, e)
            dq = self._kq_pack_dn(dm, e)
            handles += [self._arena_wf[i], self._arena_wd[i]]
            tensors += [fp, dq]
            offsets += [sidx * self._kq_slot_wf_b[i],
                        sidx * self._kq_slot_wd_b[i]]
        ext = self.ext
        if async_:
            for h, t, o in zip(handles, tensors, offsets):
                ext.batch_upload_async(h, t, o)
            self._bump("exp_up", time.perf_counter() - t0)
            self.stat_calls["arena_pref_up"] = (
                self.stat_calls.get("arena_pref_up", 0) + 1)
            return int(ext.batch_upload_async_flush())
        ext.batch_upload_multi(handles, tensors, offsets)
        self._bump("exp_up", time.perf_counter() - t0)
        self.stat_calls["arena_up"] = (
            self.stat_calls.get("arena_up", 0) + 1)
        return -1

    # ------------------------------------------------------------------
    # prefetch (step 6): host router + assign + queue-2 content, pre-batch
    # ------------------------------------------------------------------
    def _kq_prefetch_layer(self, i: int, h2: torch.Tensor) -> int:
        """Route layer ``i`` on the host BEFORE batch 3 opens.

        Returns the queue-2 flush cookie (-1 when nothing was copied).  The
        routed prediction is stashed for ``_kq_record_moe``, which then does
        NOT route again - so the total host router cost per layer per token
        is unchanged; only WHEN it runs moves (into the window where the
        GPU is still draining the previous batch's readbacks, and where the
        miss copies can be issued before the record).
        """
        if not self._kq_layer_ok.get(i, False):
            return -1
        self._kq_arena_ensure(i)
        elist, vlist = self._gpu_router_host_topk(i, h2)
        self._moe_pref[i] = (elist, vlist)
        mp = self._arena_map[i]
        pend: list[tuple[int, int]] = []
        for e in elist:
            if e in mp:
                mp.move_to_end(e)
            else:
                pend.append((e, self._arena_assign(i, e)))
        return self._kq_upload_pending(i, pend, async_=True)

    # ------------------------------------------------------------------
    # the fused record (step 2)
    # ------------------------------------------------------------------
    def _kq_check_fused_tail(self, i: int, sg: Any, su: Any,
                             gsc: Any, sdp: Any) -> None:
        """Refuse a fused record whose tail pins the C++ recorder cannot read.

        ``batch_moe_kquant_fused_record`` dispatches the shared tail itself:
        sg/su/sdp as BF16 (the GGUF provider pins them BF16) and gsc either
        BF16 or FP32 via the ``gsc_fp32`` flag (this family pins
        ``shared_expert_gate`` FP32). Anything else - fp16 was already
        widened at init - would be read as the wrong type and corrupt the
        gate scalar, so fail loudly instead.
        """
        for name, pin in (("sg", sg), ("su", su), ("sdp", sdp)):
            if pin.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"layer {i}: shared-expert {name} pin is {pin.dtype}, "
                    "but batch_moe_kquant_fused_record reads sg/su/sdp as "
                    "BF16; run FTV_GGUF_FUSED=0 (16-dispatch arena record "
                    "with dtype-dispatched tail) or widen the pin at init")
        if gsc.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(
                f"layer {i}: shared_expert_gate pin is {gsc.dtype}; the "
                "fused record supports BF16 or FP32 (gsc_fp32 flag)")

    def _kq_record_moe(
        self, i: int, h: torch.Tensor, ext: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Host router (unless predicted) + arena slots + fused record.

        Returns the same ``(acc, sd, gsc)`` triple the base record returns,
        so the batch walk consumes it unchanged.  The caller has already
        uploaded ``h`` into ``self.h_x_fused`` and zero-filled ``h_y_acc``.
        """
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        self._kq_arena_ensure(i)
        gate_mat = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        up_mat = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        down_mat = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        pref = self._moe_pref.pop(i, None)
        if pref is None:
            # MTP multi-row walk / FTV_GGUF_PREFETCH=0: route here (same
            # math and the same canonical sort as the prefetch).
            elist, vlist = self._gpu_router_host_topk(i, h)
        else:
            elist, vlist = pref
        pending = self._kq_assign_routed(i, elist)
        self._kq_upload_pending(i, pending, async_=False)

        mp = self._arena_map[i]
        qtype = self._packed_qtype(gate_mat)
        gu_rows, gu_cols, dn_rows, dn_cols = self._kq_layer_c[i]
        if dn_cols != gu_rows:
            raise RuntimeError(
                f"layer {i}: down K {dn_cols} != gate rows {gu_rows} "
                "(the fused acc reads a silu vector of length gu_rows)")
        # ids = slot index per canonical expert; gs = [fgs_0..7, v_0..7] with
        # the K-quant gateup gscale at 1.0 (scales live in-block) and the
        # renormalized router value folded into the acc gscale - the exact
        # values the 16-dispatch path passes as push-constant gscale, so the
        # fused dot's per-row s is bit-identical.
        n = len(elist)
        # Top-k < 8 lanes (not this family): default them to a routed slot
        # with acc gscale 0. A sentinel id would compute
        # `0x7FFFFFFF * rows_per_slot` and read outside the arena; gs 0
        # against real (finite) content contributes exactly 0.
        ids = ([mp[elist[0]]] * 8) if n else [0] * 8
        gs = [0.0] * 16
        for j in range(8):
            if j < n:
                ids[j] = mp[elist[j]]
                gs[j] = 1.0
                gs[8 + j] = float(vlist[j])
        ext.batch_upload(self._arena_ids[i],
                         torch.tensor(ids, dtype=torch.int32), 0)
        ext.batch_upload(self._arena_gs[i],
                         torch.tensor(gs, dtype=torch.float32), 0)
        acc_t = torch.empty(cfg.hidden_size)
        sd_t = torch.empty(cfg.hidden_size)
        gsc = self.pinned[_translate(f"{base}.mlp.shared_expert_gate")]
        gsc_t = torch.empty(gsc.rows)
        if self._kq_fused:
            sg = self.pinned[_translate(
                f"{base}.mlp.shared_expert.gate_proj")]
            su = self.pinned[_translate(
                f"{base}.mlp.shared_expert.up_proj")]
            sdp = self.pinned[_translate(
                f"{base}.mlp.shared_expert.down_proj")]
            self._kq_check_fused_tail(i, sg, su, gsc, sdp)
            ext.batch_moe_kquant_fused_record(
                self.h_x_fused, self.h_y_acc,
                self.h_sg, self.h_su, self.h_gsc,
                self.h_x3, self.h_sdown,
                qtype, self.h_silu_all,
                self._arena_wf[i], self._arena_wd[i],
                self._arena_ids[i], self._arena_ids[i], self._arena_gs[i],
                gu_rows, gu_cols, self._arena_slot_rows_gu,
                dn_rows, dn_cols, self._arena_slot_rows_dn,
                gu_rows,                 # silu_N = the down K
                sg.handle, sg.rows, sg.cols,
                su.handle, su.rows, su.cols,
                gsc.handle, gsc.rows, gsc.cols,
                sdp.handle, sdp.rows, sdp.cols,
                sg.rows,                # gate_rows -> batch_swiglu width N
                acc_t, sd_t, gsc_t,
                gsc_fp32=(gsc.dtype == torch.float32),
            )
        else:
            # Arena WITHOUT the fusion (the A/B): 16 dispatches reading the
            # arena slots, then the shared tail - the current batched
            # record with per-expert buffers swapped for slot offsets.
            self._kq_record_16(i, ext, elist, vlist, qtype, gu_rows,
                               gu_cols, dn_rows, dn_cols,
                               acc_t, sd_t, gsc_t, gsc)
        return acc_t, sd_t, gsc_t

    def _kq_record_16(
        self, i: int, ext: Any, elist: list[int], vlist: list[float],
        qtype: str, gu_rows: int, gu_cols: int,
        dn_rows: int, dn_cols: int,
        acc_t: torch.Tensor, sd_t: torch.Tensor, gsc_t: torch.Tensor,
        gsc: Any,
    ) -> None:
        """16 routed dispatches over the arena + the shared-expert tail."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        mp = self._arena_map[i]
        gu_slot0 = [mp[e] * self._arena_slot_rows_gu for e in elist]
        dn_slot0 = [mp[e] * self._arena_slot_rows_dn for e in elist]
        n = len(elist)
        wf = [self._arena_wf[i]] * n
        wd = [self._arena_wd[i]] * n
        ext.batch_moe_kquant_record(
            self.h_x_fused, self.h_y_acc, qtype,
            self.h_silu_slots[:n],
            wf, [1.0] * n, gu_rows, gu_cols,
            wd, [float(v) for v in vlist[:n]], dn_rows, dn_cols,
            gu_slot0, dn_slot0,
        )
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

    # ------------------------------------------------------------------
    # entry hooks
    # ------------------------------------------------------------------
    def decode_step_batched(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        if self._kq_arena:
            self._kq_decode_entry()
        return super().decode_step_batched(x, pos)

    def decode_rows_batched(self, *args, **kwargs):
        if self._kq_arena:
            self._kq_decode_entry()
        return super().decode_rows_batched(*args, **kwargs)

    # NOTE: there is deliberately NO ``_record_moe_in`` override here.  The
    # MoE record hook resolves on ``GgufBatchedMixin`` first (it sits above
    # this mixin in the MRO), so ITS ``_record_moe_in`` opens with the
    # arena gate and calls ``self._kq_record_moe`` for eligible layers,
    # falling through to its own K-quant LRU record otherwise - a mixin
    # override here would either be shadowed or (via super()) land in the
    # NVFP4 base record, which cannot read GGUF ExpertBank tuples.

    # ------------------------------------------------------------------
    # GPU-router lane (FTV_GGUF_GPU_ROUTER=1, step 3)
    # ------------------------------------------------------------------
    def _kq_gpu_router_setup(self) -> None:
        """The base init's GPU-router block, in K-quant shapes.

        The base ran it only under ``FTV_GPU_ROUTER`` + ``FTV_EXP_ARENA``
        (both off for this class), so the resident router/norm/exp-scale/map
        buffers are built here instead - the decode walk then finds them
        where it expects them.  ``_moe_layer_c`` keeps LOGICAL K for K-quant
        (no ``* 2``: a GGUF bank's cols is already the GEMM's K, whereas
        NVFP4 banks report packed K/2).
        """
        cfg = self.config
        ext = self.ext
        E = cfg.num_experts
        self.h_logits = ext.allocate_resident(E * 4)
        self.h_topk_ids = ext.allocate_resident(8 * 4)
        self.h_topk_vals = ext.allocate_resident(8 * 4)
        self._gpu_router_vram = E * 4 + 8 * 4 + 8 * 4
        for i in range(len(cfg.layer_types)):
            base = f"language_model.model.layers.{i}"
            self._h2_maxdiff[i] = 0.0
            router_w = self._dense(f"{base}.mlp.gate.weight").contiguous()
            h_w = ext.upload_resident(router_w)
            self._router_w_res.append(h_w)
            self._gpu_router_vram += router_w.numel() * 4
            norm_w = self._dense(
                f"{base}.post_attention_layernorm.weight").contiguous()
            h_n = ext.upload_resident(norm_w)
            self._norm_w_res.append(h_n)
            self._gpu_router_vram += norm_w.numel() * 4
            # K-quant carries its global scale in-block: both tables are 1.0,
            # so the topk writes fgs=1.0 / dgs*v = v - the host path's gs.
            tbl = torch.ones(E, 2, dtype=torch.float32)
            h_t = ext.upload_resident(tbl)
            self._exp_scale_res.append(h_t)
            self._gpu_router_vram += E * 2 * 4
            self._arena_map_res.append(ext.allocate_resident(E * 4))
            self._arena_map_dirty.append(True)
            self._gpu_router_vram += E * 4
            self._moe_layer_c[i] = self._kq_layer_c.get(i, (0, 0, 0, 0))

    def _gpu_router_prewarm_async_uploads(self, i: int) -> int:
        """Base protocol, K-quant payloads (gate+up bundle + down)."""
        pend = self._prewarm_pending.get(i)
        if not pend:
            return -1
        self._prewarm_pending[i] = []
        cookie = self._kq_upload_pending(i, pend, async_=True)
        self.stat_calls["prewarm_async"] = (
            self.stat_calls.get("prewarm_async", 0) + 1)
        return cookie

    def _gpu_router_prewarm(self, i: int, elist_p: list[int]) -> None:
        """Base walk, with this arena's lazy layer-40 growth first.

        The base indexes ``_arena_map[i]`` while seeding ``_arena_ids``,
        which is only safe once ``_kq_arena_ensure`` has padded the lists.
        """
        if self._kq_arena:
            self._kq_arena_ensure(i)
        return super()._gpu_router_prewarm(i, elist_p)

    def _record_moe_gpu(
        self, i: int, ext: Any, elist: list[int] | None,
        vlist: list[float] | None, *, upload_ids_gs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Base protocol, K-quant payload (single qtype, no scales arenas)."""
        if not self._kq_arena:
            return super()._record_moe_gpu(
                i, ext, elist, vlist, upload_ids_gs=upload_ids_gs)
        # The GPU-router lane records the dispatch itself; the host-lane
        # gate in GgufBatchedMixin._record_moe_in does not apply here, so
        # this is the place to refuse a mixed-qtype/MTP layer loudly
        # instead of indexing the arena with the wrong slot stride.
        if not self._kq_layer_ok.get(i, False):
            raise RuntimeError(
                f"layer {i} is not arena-eligible but reached "
                "_record_moe_gpu on the K-quant arena path "
                f"(FTV_GGUF_GPU_ROUTER requires every walked layer to be "
                "uniform q4_k/q5_k; MTP/ineligible layers must decode on "
                "the host-router lane)")
        self._kq_arena_ensure(i)
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        gate_mat = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        up_mat = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        down_mat = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        if elist is not None:
            pending = self._kq_assign_routed(i, elist)
            self._kq_upload_pending(i, pending, async_=False)
        if upload_ids_gs and elist is not None:
            mp = self._arena_map[i]
            # Same unused-lane contract as _kq_record_moe: a routed slot
            # with acc gscale 0, never the raw sentinel (the shader's
            # `Ids[j] * rows_per_slot` has no range check).
            n = len(elist)
            ids = ([mp[elist[0]]] * 8) if n else [0] * 8
            gs = [0.0] * 16
            for j in range(min(8, n)):
                ids[j] = mp[elist[j]]
                gs[j] = 1.0
                gs[8 + j] = float(down_mat[elist[j]][2]) * float(vlist[j])
            ext.batch_upload(self._arena_ids[i],
                             torch.tensor(ids, dtype=torch.int32), 0)
            ext.batch_upload(self._arena_gs[i],
                             torch.tensor(gs, dtype=torch.float32), 0)
        gu_rows, gu_cols, dn_rows, dn_cols = self._kq_layer_c[i]
        qtype = self._packed_qtype(gate_mat)
        acc_t = torch.empty(cfg.hidden_size)
        sd_t = torch.empty(cfg.hidden_size)
        gsc = self.pinned[_translate(f"{base}.mlp.shared_expert_gate")]
        gsc_t = torch.empty(gsc.rows)
        sg = self.pinned[_translate(f"{base}.mlp.shared_expert.gate_proj")]
        su = self.pinned[_translate(f"{base}.mlp.shared_expert.up_proj")]
        sdp = self.pinned[_translate(f"{base}.mlp.shared_expert.down_proj")]
        self._kq_check_fused_tail(i, sg, su, gsc, sdp)
        ext.batch_moe_kquant_fused_record(
            self.h_x_fused, self.h_y_acc,
            self.h_sg, self.h_su, self.h_gsc,
            self.h_x3, self.h_sdown,
            qtype, self.h_silu_all,
            self._arena_wf[i], self._arena_wd[i],
            self._arena_ids[i], self._arena_ids[i], self._arena_gs[i],
            gu_rows, gu_cols, self._arena_slot_rows_gu,
            dn_rows, dn_cols, self._arena_slot_rows_dn,
            gu_rows,
            sg.handle, sg.rows, sg.cols,
            su.handle, su.rows, su.cols,
            gsc.handle, gsc.rows, gsc.cols,
            sdp.handle, sdp.rows, sdp.cols,
            sg.rows,           # gate_rows -> batch_swiglu width N
            acc_t, sd_t, gsc_t,
            gsc_fp32=(gsc.dtype == torch.float32),
        )
        return acc_t, sd_t, gsc_t


__all__ = ["GgufArenaMixin"]
