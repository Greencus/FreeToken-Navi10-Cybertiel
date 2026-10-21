"""Hybrid streaming decode for the Ressl ModelOpt NVFP4 checkpoint.

Reuses the validated FreeToken decode architecture (`ResidentQwen35`:
GDN recurrence, full-attention SDPA, MoE routing, norms, sampling) and
overrides ONLY the weight-source layer:

- Dense BF16 (attention, shared experts, router, norms, conv, embed,
  lm_head): pinned once to VRAM, run on the proven
  ``linear_bf16_resident_io`` kernel. No dequant risk at all.
- NVFP4 switch experts (256/layer, E2M1 + e4m3 blocks + fp32 global):
  kept as CPU mmap views; the 8 active experts per layer are
  dequantized to BF16 per token, uploaded, run, freed.

Stem translation (methods keep speaking the affine-style names, this
class maps them to Ressl keys):
``language_model.lm_head`` -> ``lm_head``,
``language_model.model.layers.*`` -> ``model.language_model.layers.*``.
"""
from __future__ import annotations

import pathlib
import os
import math
import itertools
import queue
import threading
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F

from .loader import ShardedSafetensors
from .config import load_qwen35_config
from .attention import (
    apply_partial_rotary,
    compute_text_rope,
    rms_norm_head,
)
from .linear_attn import l2norm

# Shared absent-expert sentinel for the GPU-router arena map (oracle-8
# fix 1): ONE constant used by BOTH the host map fill (_gpu_router_sync_map)
# and the topk256_k8_f32.comp shader (NO_SLOT).  The kernel maps it to the
# reserved no-content slot 0; slot indices handed out by _arena_assign are
# 1..arena_n-1, so a 0 in arena_ids is never a valid assignment.
_ARENA_NO_SLOT = 0x7FFFFFFF
from .loader import AffineQuantMatrix
from .resident import ResidentQwen35

from .nvfp4 import PREFIX as RPREFIX
from .nvfp4 import dequantize_nvfp4, pack_scales_u32
from ._runtime import get_ext


class _Bf16Pinned:
    """One BF16 matrix pinned in VRAM."""

    __slots__ = ("handle", "rows", "cols")

    def __init__(self, ext, tensor: torch.Tensor):
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"expected BF16 dense weight, got {tensor.dtype}")
        contiguous = tensor.contiguous()
        self.handle = ext.upload_resident(contiguous)
        self.rows = contiguous.shape[-2]
        self.cols = contiguous.shape[-1]


def _translate(name: str) -> str:
    """Affine-style name -> Ressl key. Works on bare stems (no suffix)
    and full tensor names alike: only the prefix is rewritten, the
    remainder (including any `.weight` / `A_log` / `dt_bias` suffix)
    passes through verbatim."""
    if name == "language_model.lm_head":
        return "lm_head"
    if name.startswith("language_model.model."):
        return "model.language_model." + name[len("language_model.model."):]
    if name.startswith("language_model."):
        return "model.language_model." + name[len("language_model."):]
    raise KeyError(f"no Ressl mapping for {name}")


class ResidentNVFP4(ResidentQwen35):
    """Streaming decoder over the Ressl NVFP4 checkpoint (see module doc)."""

    # Optional debug/stop hook: decode loops truncate after this many layers
    # (negative/None = run all). Set by tests only.
    _bs_stop: int | None = None

    # Error-message descriptor for the checkpoint flavor (overridden below).
    _open_desc = "Ressl NVFP4"

    def _open(self, model_dir):
        """Config + tensor source for this checkpoint flavor.

        Hook so ``ResidentGGUF`` swaps the weight source (a single mmap'd
        GGUF) without touching the rest of __init__."""
        return load_qwen35_config(model_dir), ShardedSafetensors(model_dir)

    def __init__(
        self,
        model_dir: str | pathlib.Path,
        ext=None,
        max_len: int = 4096,
        kv_dtype: str = "fp32",
    ) -> None:
        torch.set_num_threads(8)
        # ext=None resolves the pybind module lazily via _runtime (see
        # ResidentQwen35.__init__); explicit ext keeps fork behavior.
        # Normalize the LOCAL first: __init__ keeps calling ext.* for the
        # resident allocations below, so leaving it None crashes the
        # implicit-resolution path (CLI/engine) that tests never exercise.
        if ext is None:
            ext = get_ext()
        self.ext = ext
        if kv_dtype not in ("fp32", "bf16"):
            raise ValueError(f"kv_dtype must be fp32|bf16, got {kv_dtype!r}")
        self.kv_dtype = kv_dtype
        try:
            self.config, self.tensors = self._open(model_dir)
            cfg = self.config
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"cannot open {self._open_desc} model at {model_dir}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.max_len = max_len

        max_in = max(
            cfg.hidden_size,
            cfg.linear_num_value_heads * cfg.linear_value_head_dim,
            cfg.num_attention_heads * cfg.head_dim,
        )
        self.h_x = ext.allocate_resident(max_in * 4)
        self.h_y = ext.allocate_resident(cfg.vocab_size * 4)
        # Dedicated buffer for the shared MoE input h (task #10): the
        # fused gate+up calls reuse it across all 8 experts of a layer,
        # while the per-expert down calls keep using self.h_x for their
        # (different) silu(gate)*up input.
        self.h_x_fused = ext.allocate_resident(cfg.hidden_size * 4)
        # Silu-fold scratch (task #11): the fused gate+up kernel writes
        # Y = silu(gate)*up here; the down kernel reads it as its X,
        # resident-to-resident, so no host round trip per expert.
        # Sized for chunked prefill: max_in rows x 512 gate rows x fp32.
        self.h_silu = ext.allocate_resident(max_in * 512 * 4)
        # Task #12 (batched decode path): per-expert silu slots expose the
        # fold product to the accumulate-down kernel while the 8 experts of a
        # layer all live in one command buffer (no overwrite hazards).
        # `gate rows` = hidden_size/4 (all layers, 512).
        gate_rows = cfg.hidden_size // 4
        self.h_silu_slots = [
            ext.allocate_resident(gate_rows * 4) for _ in range(8)
        ]
        # Accumulator for the 8 routed-expert down calls of one layer
        # (cleared via vkCmdFillBuffer at the top of the layer batch).
        self.h_y_acc = ext.allocate_resident(cfg.hidden_size * 4)
        # Shared-expert scratch: gate/up proj outputs, sigmoid scalar,
        # and the shared down output (added host-side with acc).
        self.h_sg  = ext.allocate_resident(gate_rows * 4)
        self.h_su  = ext.allocate_resident(gate_rows * 4)
        self.h_gsc = ext.allocate_resident(4)
        self.h_sdown = ext.allocate_resident(cfg.hidden_size * 4)
        # Dense batched-step scratch (task #12): attn out_proj input +
        # output per layer batch B, and the shared down input. The out_proj
        # input is the pre-out_proj gated vector: rows nv*hv (GDN) or
        # hq*hd (full attn), i.e. up to max_proj below, not hidden_size.
        max_proj = max(
            cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim,
            cfg.num_attention_heads * cfg.head_dim * 2,
        )
        self.h_x2 = ext.allocate_resident(max_proj * 4)
        self.h_x3 = ext.allocate_resident(gate_rows * 4)
        self.h_ao = ext.allocate_resident(cfg.hidden_size * 4)
        # Batched dense in-projection scratch (task #12): linear_attn
        # qkv/z/a/b (max 8192/4096) and full-attn qg/k/v (max 8192/512).
        vdim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        self.h_qkv = ext.allocate_resident(max_proj * 4)
        self.h_z = ext.allocate_resident(vdim * 4)
        self.h_a = ext.allocate_resident(vdim * 4)
        self.h_b = ext.allocate_resident(vdim * 4)
        self.h_qg = ext.allocate_resident(max_proj * 4)
        self.h_k = ext.allocate_resident(
            cfg.num_key_value_heads * cfg.head_dim * 4)
        self.h_v = ext.allocate_resident(
            cfg.num_key_value_heads * cfg.head_dim * 4)
        # Task #13 (GPU-resident GDN): shared scratch for the merged
        # linear-attention batch. h_mixed holds the conv+silu output [C],
        # h_x_resid the layer input x [H] (for x1 = x + ao on GPU),
        # h_x1 the post-attention residual [H] (downloaded as x1).
        _gdn_c = (cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
                  + cfg.linear_num_value_heads * cfg.linear_value_head_dim)
        self.h_mixed = ext.allocate_resident(_gdn_c * 4)
        self.h_x1 = ext.allocate_resident(cfg.hidden_size * 4)
        self.h_x_resid = ext.allocate_resident(cfg.hidden_size * 4)
        # Always-on stage timers (perf_counter ~50ns vs ms ops).
        self.stats: dict[str, float] = {}
        self.stat_calls: dict[str, int] = {}

        # Heterogeneous impl types by design (base holds affine slots and
        # matrices); Any keeps the override checker honest about that.
        self.pinned: dict[str, Any] = {}
        # fp32 materialization cache for _dense() (router/norm/conv taps):
        # read-only views, cached once so the per-layer hot path stops
        # re-copying BF16->fp32 on every token.
        self._dense_cache: dict[str, torch.Tensor] = {}
        # Per-expert MoE statics: (gate_mat, up_mat, down_mat, expert) ->
        # (gu_rows, gu_cols, dn_rows, dn_cols, fused_gs, down_gs).
        # Shapes + gscales never change; cached once so decode stops
        # re-running float()/shape reads per layer per token.
        self._moe_statics: dict[tuple, tuple] = {}
        # Stable stem registry: id(bundle_list) -> expert_mats stem.
        # Built in _pin_all; lets hotsets survive across processes
        # (id()s don't). Must exist before _pin_all() runs below.
        self._mat_stems: dict[int, str] = {}
        # expert_mats[affine-style stem] = list of 256 NVFP4 bundles
        # (packed, scales, global_scale) as CPU mmap views.
        self.expert_mats: dict[str, Any] = {}
        # LRU resident cache for NVFP4 packed+scales (task #8). Key is
        # (id of the per-layer bundle list, expert); the lists live for
        # the model lifetime so id() is stable. Budget in MB via env
        # FTV_EXP_CACHE_MB (default 4096; 0 disables caching). 8 GB VRAM
        # total: pinned bf16 ~1 GB + KV ~0.4 GB leaves ~4 GB free.
        # Entry tuple: (h_w, h_s, width_bytes, w_bytes, s_bytes).
        self._exp_cache: OrderedDict[tuple[int, ...], tuple[int, int, int, int, int]] = OrderedDict()
        self._exp_cache_bytes = 0
        # Free-buffer pool keyed by (w_bytes, s_bytes): eviction drops into
        # it instead of vkFreeMemory/vkAllocateMemory churn (task #12 opt:
        # batch uploads re-use pooled handles on the next re-miss).
        self._exp_pool: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._exp_pool_bytes = 0
        # Host-side packed+scales_u32 cache (task #12 opt): a re-miss after
        # VRAM eviction re-runs torch.cat + pack_scales_u32 unless we keep
        # the packed CPU tensors. Budget 256 MiB (default; N=128 A/B at 4 GB
        # VRAM showed no benefit at 768 - FTV_HOST_PACK_MB overrides,
        # 0 disables storing new entries).
        self._host_pack: OrderedDict[tuple, tuple] = OrderedDict()
        self._host_pack_bytes = 0
        # Dedicated per-key MISS counter for _host_pack_get's admission
        # gate (FTV_HOST_PACK_GATE): misses build+discard until the gate-th
        # miss, which caches, so one-shot routing keys never add bytes.
        # _exp_freq is NOT reused: it counts hits too and would promote
        # entries that never needed packing.
        self._pack_miss: dict[tuple, int] = {}
        try:
            host_pack_mb = int(os.environ.get("FTV_HOST_PACK_MB", "256"))
        except (ValueError, TypeError):
            host_pack_mb = 256
        self._host_pack_budget = host_pack_mb * 1024 * 1024
        # Speculative pre-pack (task b'): FTV_SPEC_PACK (default OFF after
        # the decisive A/B: wins ~1/4 of waste, TAIL16 2-5 ms/tok slower;
        # =1 opts in when a workload wants it). A daemon packer thread
        # pre-builds CPU packed+scales for the experts a layer is
        # predicted to route to at its MoE batch, from the PRE-ATTENTION
        # input h (available at batchA/B1 record time);
        # _host_pack_get's consult pops the ready map. When unset/0,
        # _spec_buf is None and the hot path is today's path + one attr
        # check (verified bit-identical).
        try:
            spec_pack = int(os.environ.get("FTV_SPEC_PACK", "0"))
        except (TypeError, ValueError):
            spec_pack = 0
        self._spec_buf: dict[tuple, tuple] | None = {} if spec_pack else None
        self._spec_buf_bytes = 0
        self._spec_peak_bytes = 0
        # Per-layer slot ring: the pre-batch1 lookahead writes cells for
        # layer L into _spec_slots[L]; the pre-batch3 drain of the SAME
        # token moves finished cells into the ready map; stale
        # generations are discarded.
        self._spec_slots: list[dict[tuple, dict]] = [
            {} for _ in range(len(self.config.layer_types))]
        self._spec_gen: list[int] = [0] * len(self.config.layer_types)
        # Half-B armed flags: a layer only speculates when the PREVIOUS
        # token actually missed the VRAM cache (re-arm on miss; cache-
        # stable layers never waste pack work).
        self._spec_armed: list[bool] = [False] * len(self.config.layer_types)
        # Worst-case transient bound: 40 layers x 16 keys (~70 MB).
        self._spec_max_cells = 40 * 16
        # Oracle amend 3 fallback: measured GDN late-rate was >80%, so
        # GDN layers enqueue ONLY when >= this many predicted experts are
        # ec/hp-missing (concentrate pack work on miss-heavy layers).
        # Default ON (measured-triggered): set FTV_SPEC_PACK_GDN3=0 to
        # disable; MIN=3.
        try:
            self._spec_gdn3 = int(
                os.environ.get("FTV_SPEC_PACK_GDN3", "1")) != 0
        except (TypeError, ValueError):
            self._spec_gdn3 = True
        self._spec_gdn_min = 3
        self._spec_queue = queue.Queue()
        self._spec_worker: threading.Thread | None = None
        # Expert-frequency map for hotset prewarm (task #12 opt): key -> hits.
        self._exp_freq: dict[tuple, int] = {}
        try:
            cache_mb = int(os.environ.get("FTV_EXP_CACHE_MB", "4096"))
        except (ValueError, TypeError):
            cache_mb = 4096
        self._exp_cache_budget = max(0, cache_mb) * 1024 * 1024
        # Mat-id -> layer-index reverse map (populated in _pin_all).
        self._mat_layer: dict[int, int] = {}
        self._pin_all()

        # ---- Fixed-slot per-layer expert arena (task: desc-churn lever)
        # Decode MoE experts live in preallocated per-layer fixed
        # VkBuffers; contents rewritten via batch_upload, NO alloc/free
        # on decode.  Desc keys (pipe, slot.W, slot.S, X, silu[e]) are
        # stable -> desc misses ~0/token in steady state.  The global
        # _exp_cache (LRU/LFU) is the hit-overlay; the arena per-layer
        # OrderedDicts are the assignment maps (slot reassignment on
        # eviction, never free_resident).
        self._arena_enabled = os.environ.get(
            "FTV_EXP_ARENA", "1").strip() != "0"
        try:
            self._arena_n = int(
                os.environ.get("FTV_EXP_ARENA_SLOTS", "32"))
        except (TypeError, ValueError):
            self._arena_n = 32
        # Arena capacity floor: the FUSED MoE dispatch reads ALL routed
        # experts' arena slots in one pass (fused gateup8 + fused acc8
        # per layer), so each layer's arena must hold at least
        # num_experts_per_tok slots — fewer would silently drop routed
        # experts from the fused GEMM.  Floor at 9 so the usable capacity
        # (slot 0 is RESERVED as the no-content marker — oracle-8 fix 1:
        # a written 0 in arena_ids unambiguously means "absent", never a
        # valid assignment; _arena_assign hands out slots 1..arena_n-1)
        # stays >= 8 (the default top-k); the explicit check below still
        # fires for models whose top-k exceeds the floor.
        self._arena_n = max(9, self._arena_n)
        if (self._arena_enabled
                and self._arena_n - 1 < cfg.num_experts_per_tok):
            raise RuntimeError(
                f"FTV_EXP_ARENA_SLOTS={self._arena_n} (usable "
                f"{self._arena_n - 1} slot-0 reserved) < num_experts_per_tok="
                f"{cfg.num_experts_per_tok}: the fused arena MoE path needs "
                f"one arena slot per routed expert (see _moe_batched_decode)")
        # Per-LAYER arena buffers (NOT per-slot): each layer owns ONE
        # fused-packed, ONE fused-scales, ONE down-packed, ONE down-
        # scales VkBuffer; the _arena_n bundles live at fixed BYTE
        # offsets slot*slotbytes inside.  Desc keys therefore carry only
        # the (stable, layer-constant) arena handles -> ~2 keys/layer/
        # shader instead of one per (slot, silu-position) pair; the
        # 3072-set desc cache no longer overflows on decode.  The slot0
        # push constant addresses the right rows inside the arena.
        # Sizes from actual bundles (shape-peek, mmap view, no data).
        _gm = self.expert_mats.get(
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj")
        _dm = self.expert_mats.get(
            "language_model.model.layers.0.mlp.switch_mlp.down_proj")
        if _gm and _dm:
            _gp, _, _ = _gm[0]
            _dp, _, _ = _dm[0]
            _rows_fused = _gp.shape[-2] * 2  # gate+up stacked
            _rows_down = _dp.shape[-2]
            _packed_cols = _gp.shape[-1]       # K_gate/2
            # pack_scales_u32 emits [rows, K/64] u32 words (4 e4m3 scale
            # bytes per word); K/64 == packed_cols/32, NOT packed_cols/8.
            _scales_u32_cols = _packed_cols // 32
            _wf_b = int(_rows_fused * _packed_cols)            # u8 packed
            _sf_b = int(_rows_fused * _scales_u32_cols * 4)    # u32 scales
            _wd_b = int(_rows_down * _dp.shape[-1])            # u8 packed
            _kd = _dp.shape[-1] * 2  # hidden cols for down
            _sd_b = int(_rows_down * (_kd // 64) * 4)          # u32 scales
        else:
            # Fallback: derive from config (Qwen3.5-Nano dims).
            _inter = cfg.moe_intermediate_size
            _h = cfg.hidden_size
            _rows_fused = _inter * 2
            _rows_down = _h
            _wf_b = _inter * 2 * (_h // 2)
            _sf_b = _inter * 2 * (_h // 64) * 4
            _wd_b = _h * (_inter // 2)
            _sd_b = _h * (_inter // 64) * 4
        self._arena_rows_fused = _rows_fused  # slot0 stride (rows)
        self._arena_rows_down = _rows_down    # slot0 stride (rows)
        self._arena_slot_wf_b = _wf_b  # slot stride in bytes
        self._arena_slot_sf_b = _sf_b
        self._arena_slot_wd_b = _wd_b
        self._arena_slot_sd_b = _sd_b
        self._arena_bytes_fused = _wf_b + _sf_b  # 1.125 MiB/slot
        self._arena_bytes_down  = _wd_b + _sd_b   # 0.5625 MiB/slot
        self._arena_wf: list[int] = []     # per layer: fused u8 arena
        self._arena_sf: list[int] = []     # per layer: fused scales arena
        self._arena_wd: list[int] = []     # per layer: down u8 arena
        self._arena_sd: list[int] = []     # per layer: down scales arena
        self._arena_map: list[OrderedDict] = []  # per layer: expert→slot_idx
        self._arena_ids: list[int] = []       # per layer: resident u32[8]
        self._arena_gs: list[int] = []        # per layer: resident float[16]
        self._arena_keys: set = set()          # all arena-backed cache keys
        _n_layers = len(cfg.layer_types)
        if self._arena_enabled:
            for _i in range(_n_layers):
                self._arena_wf.append(
                    ext.allocate_resident(_wf_b * self._arena_n))
                self._arena_sf.append(
                    ext.allocate_resident(_sf_b * self._arena_n))
                self._arena_wd.append(
                    ext.allocate_resident(_wd_b * self._arena_n))
                self._arena_sd.append(
                    ext.allocate_resident(_sd_b * self._arena_n))
                self._arena_map.append(OrderedDict())
                self._arena_ids.append(
                    ext.allocate_resident(8 * 4))  # u32[8]
                # Fused dispatches read per-expert fp32 global scales
                # (gateup gscale at [0..8), acc gscale*router at [8..16))
                # from this per-layer resident buffer (32 B upload/token).
                self._arena_gs.append(
                    ext.allocate_resident(16 * 4))  # float[16]
        self._arena_total_bytes = (
            (_wf_b + _sf_b + _wd_b + _sd_b)
            * self._arena_n * _n_layers if self._arena_enabled else 0)
        # Fused MUL_MAT_ID-style MoE scratch: one contiguous buffer
        # holding all 8 experts' silu(gate)*up side-by-side (expert j at
        # rows [j*gate_rows, (j+1)*gate_rows)).  The fused gateup8
        # kernel writes here and the fused acc8 kernel reads it with
        # X base = j*gate_rows; the legacy 16-dispatch path keeps using
        # the 8 separate h_silu_slots buffers.
        self.h_silu_all = ext.allocate_resident(gate_rows * 8 * 4)

        # ---- GPU-router setup (task #22, oracle-3 R1-R5) ---------------
        # FTV_GPU_ROUTER=1 routes decode MoE through GPU-resident
        # RMSNorm -> fp32 router GEMM -> topk-256x8 -> arena ids/gs write
        # feeding the fused MoE dispatches with GPU-written routing.
        # =0 keeps the ORIGINAL host h2/router/topk + host ids/gs upload
        # path bit-for-bit (fallback).  Requires the fixed-slot arena
        # (fused dispatches read the GPU-written ids/gs) — silently
        # falls back to the host path when the arena is disabled.
        # FTV_GPU_ROUTER_CHECK=1 additionally keeps the HOST h2+router
        # active per layer for cross-validation gates (h2 <=1e-5/layer,
        # paired (id, renormalized scale) <=1e-6 after canonical sort);
        # that is validation-harness mode, not steady-state mode.
        try:
            self._gpu_router = int(
                os.environ.get("FTV_GPU_ROUTER", "0")) != 0
        except (TypeError, ValueError):
            self._gpu_router = False
        try:
            self._gpu_router_check = int(
                os.environ.get("FTV_GPU_ROUTER_CHECK", "0")) != 0
        except (TypeError, ValueError):
            self._gpu_router_check = False
        if self._gpu_router and not self._arena_enabled:
            self._gpu_router = False
        self._gpu_router_vram = 0
        self._router_w_res: list[int] = []     # per layer: fp32 [E, H]
        self._norm_w_res: list[int] = []       # per layer: fp32 [H]
        self._exp_scale_res: list[int] = []    # per layer: fp32 [E, 2]
        self._arena_map_res: list[int] = []    # per layer: u32 [E]
        self._arena_map_dirty: list[bool] = []
        self._prewarm_pending: dict[int, list] = {}  # fix-3 warm-up queue
        self._moe_layer_c: dict[int, tuple] = {}  # layer shapes for record
        # Cross-validation accumulators (CHECK mode + harness).
        self._h2_maxdiff: dict[int, float] = {}
        self._pair_maxdiff = 0.0
        self._lg_maxdiff_h = 0.0
        self._lg_maxdiff_gemm = 0.0
        self._gpu_router_fails = 0
        self._gpu_router_ties = 0
        self._gpu_router_fallback = False
        self._gpu_router_host_list: tuple = ([], [])
        self._last_h2g: torch.Tensor | None = None
        self._last_lg: torch.Tensor | None = None
        if self._gpu_router:
            E = cfg.num_experts
            self.h_logits = ext.allocate_resident(E * 4)
            self.h_topk_ids = ext.allocate_resident(8 * 4)
            self.h_topk_vals = ext.allocate_resident(8 * 4)
            self._gpu_router_vram += E * 4 + 8 * 4 + 8 * 4
            for _i in range(len(cfg.layer_types)):
                base = f"language_model.model.layers.{_i}"
                self._h2_maxdiff[_i] = 0.0
                router_w = self._dense(
                    f"{base}.mlp.gate.weight").contiguous()
                h_w = ext.upload_resident(router_w)
                self._router_w_res.append(h_w)
                self._gpu_router_vram += router_w.numel() * 4
                norm_w = self._dense(
                    f"{base}.post_attention_layernorm.weight").contiguous()
                h_n = ext.upload_resident(norm_w)
                self._norm_w_res.append(h_n)
                self._gpu_router_vram += norm_w.numel() * 4
                gm = self.expert_mats.get(
                    f"{base}.mlp.switch_mlp.gate_proj")
                dm = self.expert_mats.get(
                    f"{base}.mlp.switch_mlp.down_proj")
                tbl = torch.zeros(E, 2, dtype=torch.float32)
                for _e in range(E):
                    try:
                        tbl[_e, 0] = float(gm[_e][2])
                        tbl[_e, 1] = float(dm[_e][2])
                    except (TypeError, ValueError, RuntimeError,
                             IndexError):
                        tbl[_e, 0] = 0.0
                        tbl[_e, 1] = 0.0
                h_t = ext.upload_resident(tbl.contiguous())
                self._exp_scale_res.append(h_t)
                self._gpu_router_vram += E * 2 * 4
                h_m = ext.allocate_resident(E * 4)
                self._arena_map_res.append(h_m)
                self._gpu_router_vram += E * 4
                self._arena_map_dirty.append(True)
                _gp, _, _ = gm[0]
                _dp, _, _ = dm[0]
                self._moe_layer_c[_i] = (
                    _gp.shape[-2], _gp.shape[-1] * 2,
                    _dp.shape[-2], _dp.shape[-1] * 2)

        self.gdn_state: dict[int, torch.Tensor] = {}
        self.conv_buf: dict[int, torch.Tensor] = {}
        key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        conv_dim = 2 * key_dim + value_dim
        for i, t in enumerate(cfg.layer_types):
            if t == "linear_attention":
                self.gdn_state[i] = torch.zeros(
                    cfg.linear_num_value_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_value_head_dim,
                )
                self.conv_buf[i] = torch.zeros(conv_dim, 3)
        # Task #13 (GPU-resident GDN): per-linear-layer pinned params
        # (fp32) + resident state/conv history. ~2 MB state + 96 KB conv
        # per layer (~63 MB for 30 layers). Host torch state stays the
        # reference; _gdn_gpu_dirty tracks divergence (set by reset /
        # prefill / sequential decode, cleared by the batched sync).
        self.gdn_taps: dict[int, int] = {}
        self.gdn_alog: dict[int, int] = {}
        self.gdn_dtb: dict[int, int] = {}
        self.gdn_normw: dict[int, int] = {}
        self.gdn_state_gpu: dict[int, int] = {}
        self.conv_buf_gpu: dict[int, int] = {}
        for i, t in enumerate(cfg.layer_types):
            if t != "linear_attention":
                continue
            base = f"language_model.model.layers.{i}"
            try:
                taps = self.tensors.get(
                    _translate(f"{base}.linear_attn.conv1d.weight")
                ).float()[:, 0, :].contiguous()
                alog = self.tensors.get(
                    _translate(f"{base}.linear_attn.A_log")
                ).float().contiguous()
                dtb = self.tensors.get(
                    _translate(f"{base}.linear_attn.dt_bias")
                ).float().contiguous()
                normw = self.tensors.get(
                    _translate(f"{base}.linear_attn.norm.weight")
                ).float().contiguous()
            except (KeyError, RuntimeError) as exc:
                raise RuntimeError(
                    f"missing GDN params for layer {i}: {exc}") from exc
            self.gdn_taps[i] = ext.upload_resident(taps)
            self.gdn_alog[i] = ext.upload_resident(alog)
            self.gdn_dtb[i] = ext.upload_resident(dtb)
            self.gdn_normw[i] = ext.upload_resident(normw)
            self.gdn_state_gpu[i] = ext.allocate_resident(
                cfg.linear_num_value_heads * cfg.linear_key_head_dim
                * cfg.linear_value_head_dim * 4)
            self.conv_buf_gpu[i] = ext.allocate_resident(conv_dim * 3 * 4)
        self._gdn_C = conv_dim
        self._gdn_key_dim = key_dim
        self._gdn_rep = (
            cfg.linear_num_value_heads // cfg.linear_num_key_heads)
        self._gdn_scale = 1.0 / math.sqrt(cfg.linear_key_head_dim)
        self._gdn_gpu_dirty = True

        if max_len > 262144:
            raise ValueError(f"max_len {max_len} exceeds model max_pos 262144")
        self.k_cache: dict[int, torch.Tensor] = {}
        self.v_cache: dict[int, torch.Tensor] = {}
        kv_torch_dtype = torch.float32 if kv_dtype == "fp32" else torch.bfloat16
        for i, t in enumerate(cfg.layer_types):
            if t == "full_attention":
                self.k_cache[i] = torch.zeros(
                    max_len, cfg.num_key_value_heads, cfg.head_dim,
                    dtype=kv_torch_dtype,
                )
                self.v_cache[i] = torch.zeros(
                    max_len, cfg.num_key_value_heads, cfg.head_dim,
                    dtype=kv_torch_dtype,
                )
        self._seq_len = 0
        if self._spec_buf is not None:
            self._spec_worker = threading.Thread(
                target=self._spec_worker_main,
                name="spec-pack",
                daemon=True,
            )
            self._spec_worker.start()

        pos = torch.arange(max_len, dtype=torch.float32)
        cos, sin = compute_text_rope(
            pos, cfg.rope_theta, cfg.head_dim, cfg.partial_rotary_factor
        )
        self.rope_cos = cos
        self.rope_sin = sin
        self.rotary_dim = self.rope_cos.shape[-1]
        self.reset()
        # NOTE (task #12 opt): hotset prewarm is intentionally NOT done
        # here. Prefill touches hundreds of bundles and LRU-evicts
        # init-time prewarms before decode starts. Call load_hotset()
        # explicitly AFTER prefill, right before decode, e.g.:
        #   model.prefill_chunked(prompt); model.load_hotset(path)

    # -- weight staging -------------------------------------------------
    def _pin_bf16(self, rstem: str) -> _Bf16Pinned:
        try:
            tensor = self.tensors.get(rstem + ".weight")
        except (KeyError, RuntimeError) as exc:
            raise ValueError(f"missing BF16 weight {rstem}.weight: {exc}") from exc
        slot = _Bf16Pinned(self.ext, tensor.float() if tensor.dtype != torch.bfloat16 else tensor)
        self.pinned[rstem] = slot
        return slot

    def _pin_stem(self, stem: str) -> None:
        """Pin one affine-style stem: BF16 direct, or NVFP4 triplet
        dequantized once to BF16 (small matrices only -- switch experts
        stream per token instead, see _pin_all)."""
        rstem = _translate(stem)
        if self.tensors.contains(rstem + ".weight_scale"):
            packed = self.tensors.get(rstem + ".weight")
            scales = self.tensors.get(rstem + ".weight_scale")
            gs = self.tensors.get(rstem + ".weight_scale_2")
            try:
                dense = dequantize_nvfp4(packed, scales, gs).to(torch.bfloat16)
            except (ValueError, TypeError, RuntimeError) as exc:
                raise ValueError(f"NVFP4 dequant failed for {rstem}: {exc}") from exc
            slot = _Bf16Pinned(self.ext, dense)
            self.pinned[rstem] = slot
        elif self.tensors.contains(rstem + ".weight"):
            self._pin_bf16(rstem)
        else:
            raise ValueError(f"no weight for {stem} (tried {rstem})")

    def _pin_all(self) -> None:
        cfg = self.config
        for i, layer_type in enumerate(cfg.layer_types):
            base = f"language_model.model.layers.{i}"
            rbase = f"{RPREFIX}.layers.{i}"
            if layer_type == "linear_attention":
                suffixes = (
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.in_proj_a",
                    "linear_attn.in_proj_b",
                    "linear_attn.out_proj",
                )
            else:
                suffixes = (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                )
            for suffix in suffixes:
                self._pin_stem(f"{base}.{suffix}")
            for suffix in (
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj",
                "mlp.shared_expert_gate",
            ):
                self._pin_stem(f"{base}.{suffix}")
            # Switch experts: keep CPU views, stream active 8 per token.
            for suffix, short in (
                ("mlp.switch_mlp.gate_proj", "gate_proj"),
                ("mlp.switch_mlp.up_proj", "up_proj"),
                ("mlp.switch_mlp.down_proj", "down_proj"),
            ):
                bundles = []
                for e in range(cfg.num_experts):
                    estem = f"{rbase}.mlp.experts.{e}.{short}"
                    try:
                        packed = self.tensors.get(estem + ".weight")
                        scales = self.tensors.get(estem + ".weight_scale")
                        gs = self.tensors.get(estem + ".weight_scale_2")
                    except (KeyError, RuntimeError) as exc:
                        raise ValueError(f"missing NVFP4 expert {estem}: {exc}") from exc
                    try:
                        gscale = float(gs.reshape(-1)[0].item())
                    except (ValueError, TypeError, RuntimeError, IndexError) as exc:
                        raise ValueError(f"bad global scale for {estem}: {exc}") from exc
                    bundles.append((packed, scales, gscale))
                self.expert_mats[f"{base}.{suffix}"] = bundles
                # Stable stem registry for cross-session hotsets (task #12
                # opt): id()s die with the process, stems don't.
                self._mat_stems[id(bundles)] = f"{base}.{suffix}"
                self._mat_layer[id(bundles)] = i
        self._pin_stem("language_model.lm_head")

    # -- translated accessors --------------------------------------------
    def reset(self) -> None:
        """Clear recurrent state and KV caches; marks GPU GDN state
        dirty so the batched path re-syncs host->device on next use."""
        super().reset()
        self._gdn_gpu_dirty = True
        # Speculative pre-pack (b'): token-scoped state. Fresh sessions
        # start empty + disarmed; layers re-arm on their first real miss.
        for _i in range(len(self._spec_armed)):
            self._spec_armed[_i] = False
            self._spec_slots[_i] = {}
            self._spec_gen[_i] = 0
        if self._spec_buf is not None:
            self._spec_buf.clear()
            self._spec_buf_bytes = 0

    def _sync_gdn_gpu(self) -> None:
        """Upload host GDN state + conv history to resident buffers once
        when dirty (after reset/prefill/sequential decode). One batch,
        one submit (~60 MB, ~15 ms, amortized over the decode run)."""
        if not self._gdn_gpu_dirty:
            return
        ext = self.ext
        ext.batch_begin()
        for i, t in enumerate(self.config.layer_types):
            if t != "linear_attention":
                continue
            ext.batch_upload(
                self.gdn_state_gpu[i],
                self.gdn_state[i].contiguous().float())
            ext.batch_upload(
                self.conv_buf_gpu[i],
                self.conv_buf[i].contiguous().float())
        ext.batch_submit()
        self._gdn_gpu_dirty = False

    def _dense(self, name: str) -> torch.Tensor:
        # Callers pass exact tensor keys (`.weight`, `A_log`, `dt_bias`,
        # ...); translate the prefix only, never touch the suffix.
        # Results are READ-ONLY views; cache the materialized fp32 so the
        # per-layer, per-token hot path (router, norm scales, conv taps)
        # never re-copies BF16->fp32 (up to ~80 MB/token for the router).
        cached = self._dense_cache.get(name)
        if cached is not None:
            return cached
        tensor = self.tensors.get(_translate(name))
        if tensor.dtype == torch.bfloat16:
            out = tensor.float()
        else:
            out = tensor.float().contiguous()
        self._dense_cache[name] = out
        return out

    def embed_row(self, token_id: int) -> torch.Tensor:
        """BF16 embed row straight to fp32 (no dequant anywhere)."""
        try:
            row = self.tensors.get(RPREFIX + ".embed_tokens.weight")[token_id]
        except (KeyError, RuntimeError, IndexError) as exc:
            raise ValueError(f"bad embed row {token_id}: {exc}") from exc
        return row.float().contiguous()

    def _bump(self, key: str, seconds: float) -> None:
        self.stats[key] = self.stats.get(key, 0.0) + seconds
        self.stat_calls[key] = self.stat_calls.get(key, 0) + 1

    def report_stats(self) -> str:
        lines = []
        for key in sorted(self.stats):
            total = self.stats[key]
            calls = self.stat_calls.get(key, 0)
            avg = total / calls if calls else 0.0
            lines.append(f"{key}: total={total:.3f}s calls={calls} avg={avg*1000:.3f}ms")
        return "\n".join(lines)

    def _pinned_affine(self, x: torch.Tensor, stem: str) -> torch.Tensor:
        import time as _time
        slot = self.pinned[_translate(stem)]
        ext = self.ext
        t0 = _time.perf_counter()
        ext.update_resident(self.h_x, x.contiguous().float())
        self._bump("pin_up", _time.perf_counter() - t0)
        t0 = _time.perf_counter()
        ext.linear_bf16_resident_io(
            self.h_x, slot.handle, 0, self.h_y,
            1, slot.rows, slot.cols, False,
        )
        self._bump("pin_kernel", _time.perf_counter() - t0)
        t0 = _time.perf_counter()
        out = ext.download_resident(self.h_y, [slot.rows])
        self._bump("pin_down", _time.perf_counter() - t0)
        return out

    def _pinned_affine_batch(self, x: torch.Tensor, stem: str) -> torch.Tensor:
        # Same translate-to-Ressl lookup as _pinned_affine, but batched
        # Tc>1 via temp resident buffers (base class does not translate).
        import time as _time
        slot = self.pinned[_translate(stem)]
        tc = x.shape[0]
        ext = self.ext
        h_x = ext.allocate_resident(tc * slot.cols * 4)
        h_y = ext.allocate_resident(tc * slot.rows * 4)
        try:
            t0 = _time.perf_counter()
            ext.update_resident(h_x, x.contiguous().float())
            ext.linear_bf16_resident_io(
                h_x, slot.handle, 0, h_y,
                tc, slot.rows, slot.cols, False,
            )
            self._bump("pin_batch_kernel", _time.perf_counter() - t0)
            t0 = _time.perf_counter()
            out = ext.download_resident(h_y, [tc, slot.rows])
            self._bump("pin_batch_down", _time.perf_counter() - t0)
            return out
        finally:
            ext.free_resident(h_x)
            ext.free_resident(h_y)

    def _expert_affine(
        self, x: torch.Tensor, matrix: AffineQuantMatrix | list, expert: int
    ) -> torch.Tensor:
        """One expert matvec via the native NVFP4 kernel, with an LRU
        resident-buffer cache for packed+scales (task #8).

        On a cache hit only the kernel runs; on a miss the packed E2M1
        bytes + packed e4m3 scales are uploaded once and kept. Budget
        ``FTV_EXP_CACHE_MB`` (default 2048 MB VRAM; 0 disables caching).
        Set ``FTV_NVFP4_GPU=0`` to fall back to the CPU-dequant BF16 path.

        ``matrix`` is the per-layer bundle list (positionally compatible
        with the base signature, which takes one fused matrix here).
        """
        if not isinstance(matrix, list):
            raise TypeError(
                "ResidentNVFP4 expects per-layer bundle lists, got "
                f"{type(matrix).__name__}"
            )
        import os
        import time as _time
        packed, scales, gscale = matrix[expert]
        rows = packed.shape[-2]
        cols = packed.shape[-1] * 2
        ext = self.ext
        t0 = _time.perf_counter()
        ext.update_resident(self.h_x, x.contiguous().float())
        self._bump("exp_x", _time.perf_counter() - t0)
        if os.environ.get("FTV_NVFP4_GPU", "1") == "1":
            key = (id(matrix), expert)
            entry = self._exp_cache.get(key)
            if entry is None:
                t0 = _time.perf_counter()
                scales_u32 = pack_scales_u32(scales)
                self._bump("exp_scale_pack", _time.perf_counter() - t0)
                t0 = _time.perf_counter()
                pooled = self._pool_get(
                    packed.numel() * packed.element_size(),
                    scales_u32.numel() * scales_u32.element_size(),
                )
                if pooled is not None:
                    h_w, h_s = pooled
                    ext.update_resident(h_w, packed.contiguous())
                    ext.update_resident(h_s, scales_u32.contiguous())
                else:
                    h_w = ext.upload_resident(packed.contiguous())
                    h_s = ext.upload_resident(scales_u32.contiguous())
                self._bump("exp_up", _time.perf_counter() - t0)
                entry = self._mk_entry(h_w, h_s, packed, scales_u32)
                self._cache_put(key, entry)
                self._cache_freq(key)
                self.stat_calls["cache_miss"] = self.stat_calls.get("cache_miss", 0) + 1
            else:
                self._cache_touch(key)
                self._cache_freq(key)
                self.stat_calls["cache_hit"] = self.stat_calls.get("cache_hit", 0) + 1
            h_w, h_s, _width, _wb, _sb = entry
            try:
                t0 = _time.perf_counter()
                ext.linear_nvfp4_resident_io(
                    self.h_x, h_w, h_s, self.h_y,
                    1, rows, cols, gscale,
                )
                self._bump("exp_kernel", _time.perf_counter() - t0)
                t0 = _time.perf_counter()
                out = ext.download_resident(self.h_y, [rows])
                self._bump("exp_down", _time.perf_counter() - t0)
                return out
            finally:
                pass  # cached handles must stay alive
        dense = dequantize_nvfp4(packed, scales, gscale).to(torch.bfloat16)
        t0 = _time.perf_counter()
        h_w = ext.upload_resident(dense.contiguous())
        self._bump("exp_up", _time.perf_counter() - t0)
        try:
            t0 = _time.perf_counter()
            ext.linear_bf16_resident_io(
                self.h_x, h_w, 0, self.h_y, 1, rows, cols, False,
            )
            self._bump("exp_kernel", _time.perf_counter() - t0)
            t0 = _time.perf_counter()
            out = ext.download_resident(self.h_y, [rows])
            self._bump("exp_down", _time.perf_counter() - t0)
            return out
        finally:
            ext.free_resident(h_w)

    # -- LRU expert cache ----------------------------------------------
    @staticmethod
    def _mk_entry(
        h_w: int, h_s: int, packed, scales_u32: torch.Tensor
    ) -> tuple[int, int, int, int, int]:
        """Build cache entry (h_w, h_s, width, w_bytes, s_bytes)."""
        return (
            h_w, h_s,
            packed.numel() + scales_u32.numel() * 4,
            packed.numel() * packed.element_size(),
            scales_u32.numel() * scales_u32.element_size(),
        )

    def _cache_put(self, key: tuple, entry: tuple) -> None:
        budget = self._exp_cache_budget
        if budget <= 0:
            return
        self._exp_cache[key] = entry
        self._exp_cache_bytes += entry[2]
        try:
            lfu = os.environ.get("FTV_CACHE_POLICY", "lru") == "lfu"
        except (TypeError, ValueError):
            lfu = False
        while self._exp_cache_bytes > budget and self._exp_cache:
            if lfu:
                # LFU-sampled eviction (FTV_CACHE_POLICY=lfu): among the
                # oldest 64 keys (OrderedDict iteration order is
                # least-recently-touched first; _cache_touch move_to_end),
                # evict the least-frequently-used key. min() returns the
                # first minimal over the oldest-first slice, so frequency
                # ties go to the oldest. O(64) per eviction.
                _k = min(
                    itertools.islice(self._exp_cache.items(), 64),
                    key=lambda kv: self._exp_freq.get(kv[0], 0),
                )[0]
                _hw, _hs, width, wb, sb = self._exp_cache.pop(_k)
                self.stat_calls["cache_evict_lfu"] = (
                    self.stat_calls.get("cache_evict_lfu", 0) + 1)
            else:
                _k, (_hw, _hs, width, wb, sb) = next(iter(self._exp_cache.items()))
                del self._exp_cache[_k]
            self.stat_calls["cache_evict"] = (
                self.stat_calls.get("cache_evict", 0) + 1)
            self._exp_cache_bytes -= width
            # Arena entries: drop key and release slot but NEVER pool/free
            # the fixed VkBuffers (desc cache stays valid; buffer identity
            # is stable).  Non-arena: pool and eventually free normally.
            if _k in self._arena_keys:
                self._arena_drop(_k)
            else:
                # Pool the freed handles keyed by exact byte size so a
                # re-miss of the same bundle re-uploads into them.
                key_sz = (wb, sb)
                self._exp_pool.setdefault(key_sz, []).append((_hw, _hs))
                self._exp_pool_bytes += wb + sb
                # Bound the pool to 25% of the cache budget; beyond that free.
                if self._exp_pool_bytes > budget // 4:
                    for _i in range(len(self._exp_pool.get(key_sz, [])) - 1):
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

    def _cache_touch(self, key: tuple) -> None:
        if key in self._exp_cache:
            self._exp_cache.move_to_end(key)

    def _host_pack_get(
        self, key: tuple, build
    ) -> tuple:
        """Return cached (packed, scales_u32) CPU tensors for ``key`` or
        build them via ``build()`` and cache. Lets a VRAM-evicted re-miss
        skip torch.cat + pack_scales_u32 entirely (task #12 opt).

        Returns ``(packed, scales_u32, was_hit)``; ``was_hit`` is True when
        ``key`` was already cached (callers bill time against
        ``host_pack_hit`` vs ``exp_scale_pack`` accordingly).

        Admission gate ``FTV_HOST_PACK_GATE`` (default 2, parsed per call):
        a MISS increments the per-key ``_pack_miss`` counter and only the
        gate-th+ miss stores the entry -- earlier misses build but do NOT
        store, so transient one-shot keys add zero bytes. gate=0 stores on
        the first miss (gate disabled). FTV_HOST_PACK_MB=0 keeps hits but
        never stores new entries."""
        hit = self._host_pack.get(key)
        if hit is not None:
            self._host_pack.move_to_end(key)
            self.stats["host_pack_hits"] = self.stats.get("host_pack_hits", 0) + 1
            return hit[0], hit[1], True
        # Speculative pre-pack consult (b'): the worker may have already
        # built this key's packed+scales (byte-identical to build()); pop
        # it so each key is consumed at most once. Billed under
        # spec_pack_win + the layer-type split (GDN vs full-attn), NEVER
        # host_pack_hit -- the caller sees was_hit=False (exp_scale_pack
        # gets ~0 time: nothing was built). A consult-win IS a _host_pack
        # miss, so admission works exactly like a build: _pack_miss += 1,
        # store per gate. The drain NEVER inserts into _host_pack (key
        # identity stays structural) -- this is the only insertion site.
        sb = self._spec_buf
        if sb is not None:
            spec = sb.pop(key, None)
            if spec is not None:
                packed, scales_u32, ltype = spec
                self._spec_buf_bytes -= (
                    packed.numel() * packed.element_size()
                    + scales_u32.numel() * scales_u32.element_size())
                if self._spec_buf_bytes < 0:
                    self._spec_buf_bytes = 0
                # Completion split by layer type (oracle amend 3): the
                # ready-map entry carries its source layer type so
                # window-fit is measurable per type.
                self._bump("spec_pack_win", 0.0)
                if ltype == "linear_attention":
                    self._bump("spec_win_gdn", 0.0)
                else:
                    self._bump("spec_win_full", 0.0)
                try:
                    gate = int(os.environ.get("FTV_HOST_PACK_GATE", "2"))
                except (ValueError, TypeError):
                    gate = 2
                self._pack_miss[key] = self._pack_miss.get(key, 0) + 1
                if (self._host_pack_budget > 0
                        and self._pack_miss[key] >= gate):
                    self._host_pack_store(key, packed, scales_u32)
                return packed, scales_u32, False
        try:
            gate = int(os.environ.get("FTV_HOST_PACK_GATE", "2"))
        except (ValueError, TypeError):
            gate = 2
        packed, scales_u32 = build()
        self._pack_miss[key] = self._pack_miss.get(key, 0) + 1
        if self._host_pack_budget <= 0 or self._pack_miss[key] < gate:
            return packed, scales_u32, False
        self._host_pack_store(key, packed, scales_u32)
        return packed, scales_u32, False

    def _host_pack_store(self, key: tuple, packed, scales_u32) -> None:
        """Insert (packed, scales_u32) under the LRU budget. Shared by the
        build-miss and spec-consult paths so the 1-vs-0 control (gate d:
        sorted(_host_pack.keys()) + per-key _pack_miss + byte-hash) is
        identical between FTV_SPEC_PACK=0 and =1."""
        self._host_pack[key] = (packed, scales_u32)
        self._host_pack_bytes += (
            packed.numel() * packed.element_size()
            + scales_u32.numel() * scales_u32.element_size()
        )
        while self._host_pack_bytes > self._host_pack_budget and self._host_pack:
            _k, (_pk, _su) = next(iter(self._host_pack.items()))
            del self._host_pack[_k]
            self._host_pack_bytes -= (
                _pk.numel() * _pk.element_size()
                + _su.numel() * _su.element_size()
            )

    # -- speculative pre-pack (b') -------------------------------------
    # Pack bytes are a pure function of the (immutable) expert weights, so
    # a pack produced speculatively from the pre-attention h is
    # byte-identical to the one the committed path would build from the
    # true MoE input h2 later in the SAME layer. Safety case: only the
    # committed path ever consumes an entry ("e in actual-8 AND e in
    # b'-packed"); _record_moe_in stays UNCHANGED (router still on the true
    # h2; the committed batch records purely from the actual-8).
    @staticmethod
    def _build_fused_pack(
        g_packed, u_packed, g_scales, u_scales,
    ) -> tuple:
        """Shared fused gate+up CPU build (stack rows + u32 scales pack).
        THE single implementation: the committed path (via
        _fused_gateup_handles) and the spec worker both call this so
        worker output is byte-identical to build()."""
        p = torch.cat([g_packed, u_packed], dim=-2).contiguous()
        s = pack_scales_u32(torch.cat([g_scales, u_scales], dim=-2))
        return p, s

    @staticmethod
    def _build_down_pack(packed, scales) -> tuple:
        """Shared down build. THE single implementation: the committed
        path (_down_handles_batch) and the spec worker both call it."""
        return packed, pack_scales_u32(scales)

    def _spec_worker_main(self) -> None:
        """Daemon packer (b'): pops lookahead cells, builds the CPU
        packed+scales (pure weight function), writes ONLY its own cell +
        Event. No ext calls; no _host_pack/_pack_miss/_exp_cache reads or
        writes (main precomputes everything). torch set to 1 thread so
        pack work never steals the main thread's intra-op pool
        (thread-local in torch >= 2.x; verified empirically)."""
        import torch as _torch
        _torch.set_num_threads(1)
        q = self._spec_queue
        while True:
            cell = q.get()
            if cell is None:
                return
            try:
                packed, scales_u32 = self._spec_build_cell(cell)
            except Exception:
                packed = scales_u32 = None
            cell["packed"] = packed
            cell["scales_u32"] = scales_u32
            cell["event"].set()

    def _spec_build_cell(self, cell: dict) -> tuple:
        kind = cell["kind"]
        views = cell["views"]
        if kind == "fused":
            return self._build_fused_pack(*views)
        if kind == "down":
            return self._build_down_pack(*views)
        raise ValueError(f"unknown spec pack kind {kind!r}")

    def _spec_pending(self) -> int:
        """Cells awaiting drain/consumption (queued + in slots). Bounds
        the transient spec memory: worst case ~40 layers x 16 keys."""
        n = self._spec_queue.qsize()
        for _s in self._spec_slots:
            n += len(_s)
        return n

    def _spec_drain(self, layer: int) -> None:
        """Pre-record drain of ``layer``'s lookahead slot (cells written
        moments ago by the pre-batch1 lookahead of THIS token). Single
        non-blocking Event.is_set() pass; unset cells are discarded
        (never blocks on the worker). Ready cells move (packed,
        scales_u32, layer_type) into the ready map keyed by the expert
        key, which _host_pack_get consults (pops). Drain NEVER inserts
        into _host_pack (key identity stays structural: only
        _host_pack_get applies the admission gate). Counters:
          spec_pack_late / spec_late_gdn / spec_late_full =
                             worker hadn't finished when the slot drained
                             (wasted pack work; result dropped);
          spec_pack_unused = worker finished but the entry was never
                             consulted before being superseded (prediction
                             error / stale gen)."""
        sb = self._spec_buf
        cells = self._spec_slots[layer]
        if not cells:
            return
        ltype = self.config.layer_types[layer]
        late_key = "spec_late_gdn" if ltype == "linear_attention" \
            else "spec_late_full"
        gen = self._spec_gen[layer]
        for key, cell in list(cells.items()):
            if cell.get("gen") != gen:
                # Stale generation (token aborted before its drain): void.
                if cell["event"].is_set():
                    self._bump("spec_pack_unused", 0.0)
                else:
                    self._bump("spec_pack_late", 0.0)
                    self._bump(late_key, 0.0)
                cell["discarded"] = True
                continue
            if not cell["event"].is_set():
                # Too slow: worker will finish into a discarded cell.
                cell["discarded"] = True
                self._bump("spec_pack_late", 0.0)
                self._bump(late_key, 0.0)
                continue
            packed = cell.get("packed")
            scales_u32 = cell.get("scales_u32")
            if packed is None or scales_u32 is None:
                cell["discarded"] = True
                self._bump("spec_pack_late", 0.0)
                self._bump(late_key, 0.0)
                continue
            old = sb.get(key)
            if old is not None:
                # A prior unconsumed prediction of this same key is being
                # superseded -> its pack was wasted.
                self._bump("spec_pack_unused", 0.0)
                self._spec_buf_bytes -= (
                    old[0].numel() * old[0].element_size()
                    + old[1].numel() * old[1].element_size())
            sz = (packed.numel() * packed.element_size()
                  + scales_u32.numel() * scales_u32.element_size())
            sb[key] = (packed, scales_u32, ltype)
            self._spec_buf_bytes += sz
            if self._spec_buf_bytes > self._spec_peak_bytes:
                self._spec_peak_bytes = self._spec_buf_bytes
        # Slot is consumed (moved to the ready map / discarded). Clearing
        # here guarantees no path ever re-processes an already-drained
        # cell (leftover double-insert would corrupt _spec_buf_bytes and
        # inflate unused/late). The next lookahead writes a fresh dict.
        cells.clear()

    def _spec_lookahead(self, layer: int, h: torch.Tensor) -> None:
        """Pre-batch1 lookahead (b'): run the layer router on the
        PRE-ATTENTION input h(L,T) (available at batchA/batch1 record
        time) and enqueue CPU pack cells for the predicted experts,
        missing from BOTH _exp_cache and _host_pack (Half-A: pack only
        predicted-missing). The worker packs during the layer's own
        attention window (batchA/batch1 + host recurrence + batch2) and
        the pre-batch3 drain of the SAME token consumes them -- the MoE
        batch still routes from the true h2 (_record_moe_in UNCHANGED);
        only predicted experts that ARE in the actual-8 are consumed.
        Runs only when armed (Half-B, re-armed on miss) and the
        pending-cell bound has headroom. With FTV_SPEC_PACK_GDN3=1 the
        fallback (oracle amend 3) applies to GDN layers: enqueue only
        when >= 3 predicted experts are missing from ec/hp, else void
        the slot (cleared so the drain never re-processes leftovers)."""
        if not self._spec_armed[layer]:
            # Nothing new was generated; void leftovers so the pre-batch3
            # drain of THIS token cannot re-process last token's cells.
            self._spec_slots[layer] = {}
            return
        if self._spec_pending() >= self._spec_max_cells:
            self._spec_slots[layer] = {}
            return
        cfg = self.config
        ltype = self.config.layer_types[layer]
        base = f"language_model.model.layers.{layer}"
        gate_mat = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        up_mat = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        down_mat = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        router_w = self._dense(f"{base}.mlp.gate.weight")
        logits = F.linear(h.float(), router_w.float()).float()
        probs = torch.softmax(logits, dim=-1)
        _, index = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
        elist = index.tolist() if index.dim() == 1 else index.tolist()[0]
        ec = self._exp_cache
        hp = self._host_pack
        # Half-A: pack only predicted experts that are missing from BOTH
        # ec and hp now (the authoritative main-side state at enqueue
        # time); the drain decides window-fit, the actual-8 decides value.
        miss_any: list = []
        for expert in elist:
            gkey = (id(gate_mat), id(up_mat), expert)
            dkey = (id(down_mat), expert)
            gfused = gkey in ec or gkey in hp
            gdown = dkey in ec or dkey in hp
            if not (gfused and gdown):
                miss_any.append((expert, gkey, dkey, gfused, gdown))
        if (self._spec_gdn3 and ltype == "linear_attention"
                and len(miss_any) < self._spec_gdn_min):
            # Miss mass too thin (measured late-rate >80%): skip the pack
            # AND void the slot so the drain doesn't re-process leftovers.
            self._spec_slots[layer] = {}
            self._spec_gen[layer] += 1
            return
        self._spec_gen[layer] += 1
        gen = self._spec_gen[layer]
        q = self._spec_queue
        cells: dict[tuple, dict] = {}
        for expert, gkey, dkey, gfused, gdown in miss_any:
            if not gfused:
                g_packed, g_scales, _gg = gate_mat[expert]
                u_packed, u_scales, _ug = up_mat[expert]
                cell = {
                    "kind": "fused", "gen": gen, "key": gkey,
                    "ltype": ltype,
                    "event": threading.Event(),
                    "views": (g_packed, u_packed, g_scales, u_scales),
                    "packed": None, "scales_u32": None, "discarded": False,
                }
                cells[gkey] = cell
                q.put(cell)
            if not gdown:
                d_packed, d_scales, _dg = down_mat[expert]
                cell = {
                    "kind": "down", "gen": gen, "key": dkey,
                    "ltype": ltype,
                    "event": threading.Event(),
                    "views": (d_packed, d_scales),
                    "packed": None, "scales_u32": None, "discarded": False,
                }
                cells[dkey] = cell
                q.put(cell)
        self._spec_slots[layer] = cells

    def _pool_get(
        self, wb: int, sb: int
    ) -> tuple[int, int] | None:
        """Pop a pooled (h_w, h_s) with exact byte size, else None."""
        lst = self._exp_pool.get((wb, sb))
        if lst:
            hw, hs = lst.pop()
            self._exp_pool_bytes -= wb + sb
            return hw, hs
        return None

    def _cache_freq(self, key: tuple) -> None:
        self._exp_freq[key] = self._exp_freq.get(key, 0) + 1

    # ---- Arena slot assignment (task: fixed-slot per-layer arena) ----
    def _arena_drop(self, key: tuple) -> None:
        """Release an arena key AFTER it was evicted from _exp_cache by
        _cache_put (the caller already removed it and decremented bytes).
        Drops the sibling (other-kind) entry if still present and releases
        the slot ONLY when both entries are gone.  The fixed VkBuffers are
        NEVER freed (desc cache stays valid); the slot buffers keep their
        content until reassigned by _arena_assign."""
        if len(key) == 3:
            gid, _uid, expert = key
            layer = self._mat_layer.get(gid)
        elif len(key) == 2:
            did, expert = key
            layer = self._mat_layer.get(did)
        else:
            return  # unknown key shape; skip
        if layer is None:
            return
        self._arena_keys.discard(key)
        _gm = self.expert_mats.get(
            f"language_model.model.layers.{layer}"
            f".mlp.switch_mlp.gate_proj")
        _um = self.expert_mats.get(
            f"language_model.model.layers.{layer}"
            f".mlp.switch_mlp.up_proj")
        _dm = self.expert_mats.get(
            f"language_model.model.layers.{layer}"
            f".mlp.switch_mlp.down_proj")
        _fk = (id(_gm), id(_um), expert) if (_gm and _um) else None
        _dk = (id(_dm), expert) if _dm else None
        _other = _dk if len(key) == 3 else _fk
        if _other is not None and _other in self._exp_cache:
            _w, _s, _wid, _wb, _sb = self._exp_cache.pop(_other)
            self._exp_cache_bytes -= _wid
            self._arena_keys.discard(_other)
        # Release the slot only when neither entry remains cached.  The
        # per-layer arena VkBuffer is NEVER freed (desc cache stays valid,
        # VRAM budget fixed); the slot bytes keep their content until
        # reassigned by _arena_assign.
        if ((_fk is None or _fk not in self._exp_cache)
                and (_dk is None or _dk not in self._exp_cache)):
            mp = self._arena_map[layer]
            if expert in mp:
                mp.pop(expert)
            if getattr(self, "_gpu_router", False):
                self._arena_map_dirty[layer] = True
            self.stat_calls["arena_evict"] = (
                self.stat_calls.get("arena_evict", 0) + 1)

    def _arena_assign(self, layer: int, expert: int) -> int:
        """Return the slot INDEX for (layer, expert).  Assigns a new slot
        (LRU-evict per layer) on first touch.  Eviction drops the
        victim's fused+down cache entries but does NOT free the per-layer
        arena VkBuffers.

        SLOT 0 IS RESERVED as the no-content marker (oracle-8 fix 1):
        slots handed out here are 1..arena_n-1, so a 0 in the arena ids
        buffer (GPU topk output for an absent expert, or the fallback
        default in _fused_slot0/_down_slot0) unambiguously means "no
        arena content" — never a valid assignment."""
        mp = self._arena_map[layer]
        sidx = mp.get(expert)
        if sidx is not None:
            # Existing assignment: touch for LRU order.
            mp.move_to_end(expert)
            return sidx
        # New assignment: use a free slot or evict oldest.  Capacity is
        # arena_n - 1 (slot 0 reserved); next free index is len(mp) + 1.
        if len(mp) >= self._arena_n - 1:
            # Evict LRU expert of this layer.
            victim = next(iter(mp))
            vsidx = mp.pop(victim)
            sidx = vsidx
            # Drop victim's cache entries (pair).
            _gm = self.expert_mats.get(
                f"language_model.model.layers.{layer}"
                f".mlp.switch_mlp.gate_proj")
            _um = self.expert_mats.get(
                f"language_model.model.layers.{layer}"
                f".mlp.switch_mlp.up_proj")
            _dm = self.expert_mats.get(
                f"language_model.model.layers.{layer}"
                f".mlp.switch_mlp.down_proj")
            for _k in ((id(_gm), id(_um), victim) if (_gm and _um) else (),
                       (id(_dm), victim) if _dm else ()):
                if _k and _k in self._exp_cache:
                    _w, _s, _wid, _wb, _sb = self._exp_cache.pop(_k)
                    self._exp_cache_bytes -= _wid
                    self._arena_keys.discard(_k)
        else:
            sidx = len(mp) + 1  # slot 0 reserved; first use is slot 1
        mp[expert] = sidx
        mp.move_to_end(expert)
        if getattr(self, "_gpu_router", False):
            self._arena_map_dirty[layer] = True
        return sidx

    def _fused_slot0(self, gate_mat: list, expert: int) -> int:
        """Row offset inside the fused arena for this expert's slot
        (slots hold gate+up stacked: stride = 2*N rows).  Returns 0 when
        the arena is disabled or the layer is unknown (legacy path reads
        rows [0, N) of its per-expert bundle — bit-transparent).  The 0
        fallback for an arena-enabled-but-unassigned expert is now the
        RESERVED no-content slot (oracle-8 fix 1), never a valid
        assignment — reading it yields no-content rows, and the fused
        path raises rather than silently decoding against it."""
        layer = self._mat_layer.get(id(gate_mat))
        if not self._arena_enabled or layer is None:
            return 0
        return self._arena_map[layer].get(expert, 0) * self._arena_rows_fused

    def _down_slot0(self, down_mat: list, expert: int) -> int:
        """Row offset inside the down arena for this expert's slot
        (stride = N_down rows per expert)."""
        layer = self._mat_layer.get(id(down_mat))
        if not self._arena_enabled or layer is None:
            return 0
        return self._arena_map[layer].get(expert, 0) * self._arena_rows_down

    def save_hotset(self, path: str, top_n: int = 800) -> None:
        """Persist the most-frequent expert bundles to JSON using STABLE
        stem keys (id()s die with the process). Format: list of
        [kind, stem_a, stem_b_or_null, expert, count] where kind is
        "fused" (gate+up) or "down". For prewarm on the next session
        via load_hotset (task #12 opt)."""
        import json as _json
        top = sorted(self._exp_freq.items(), key=lambda kv: -kv[1])[:top_n]
        out = []
        for key, cnt in top:
            if len(key) == 2:
                mat_id, expert = key
                stem = self._mat_stems.get(mat_id)
                if stem is None:
                    continue
                out.append(["down", stem, None, expert, cnt])
            elif len(key) == 3:
                gid, uid, expert = key
                gs = self._mat_stems.get(gid)
                us = self._mat_stems.get(uid)
                if gs is None or us is None:
                    continue
                out.append(["fused", gs, us, expert, cnt])
        try:
            with open(path, "w") as fh:
                _json.dump(out, fh, indent=0)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot write hotset file {path}: {exc}") from exc
        self.stats["hotset_saved"] = len(out)

    def load_hotset(self, path: str, limit: int = 800) -> int:
        """Prewarm the resident expert cache from a save_hotset file.
        All uploads are recorded into ONE batch (one submit), so startup
        prewarm costs ~one fence wait, not hundreds of sync uploads.
        Returns how many bundles were uploaded. Missing stems (arch
        mismatch) are skipped."""
        import json as _json
        try:
            with open(path) as fh:
                items = _json.load(fh)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read hotset file {path}: {exc}") from exc
        ext = self.ext
        ext.batch_begin()
        n = 0
        try:
            for item in items[:limit]:
                try:
                    kind, sa, sb, expert, _cnt = item
                except (TypeError, ValueError, IndexError):
                    continue
                if kind == "down":
                    down_mat = self.expert_mats.get(sa)
                    if down_mat is None:
                        continue
                    key = (id(down_mat), expert)
                    if key in self._exp_cache:
                        self._cache_touch(key)
                        continue
                    self._down_handles_batch(down_mat, expert, batch=True)
                elif kind == "fused":
                    gate_mat = self.expert_mats.get(sa)
                    up_mat = self.expert_mats.get(sb) if sb else None
                    if gate_mat is None or up_mat is None:
                        continue
                    key = (id(gate_mat), id(up_mat), expert)
                    if key in self._exp_cache:
                        self._cache_touch(key)
                        continue
                    self._fused_gateup_handles(
                        gate_mat, up_mat, expert, batch=True)
                else:
                    continue
                n += 1
        finally:
            ext.batch_submit()
        self.stats["hotset_prewarmed"] = n
        return n


    # -- chunked prefill (task #9) --------------------------------------
    def _expert_affine_batch(
        self, x: torch.Tensor, matrix: list, expert: int
    ) -> torch.Tensor:
        """Batched (Tc>1) expert matvec via the NVFP4 kernel with the same
        LRU cache as the single-token path. ``x`` is ``[Tc, K]``; returns
        ``[Tc, N]``. Uses temp resident buffers (freed on return)."""
        import time as _time
        packed, scales, gscale = matrix[expert]
        rows = packed.shape[-2]
        cols = packed.shape[-1] * 2
        tc = x.shape[0]
        key = (id(matrix), expert)
        entry = self._exp_cache.get(key)
        if entry is None:
            t0 = _time.perf_counter()
            scales_u32 = pack_scales_u32(scales)
            self._bump("exp_scale_pack", _time.perf_counter() - t0)
            t0 = _time.perf_counter()
            pooled = self._pool_get(
                packed.numel() * packed.element_size(),
                scales_u32.numel() * scales_u32.element_size(),
            )
            if pooled is not None:
                h_w, h_s = pooled
                self.ext.update_resident(h_w, packed.contiguous())
                self.ext.update_resident(h_s, scales_u32.contiguous())
            else:
                h_w = self.ext.upload_resident(packed.contiguous())
                h_s = self.ext.upload_resident(scales_u32.contiguous())
            self._bump("exp_up", _time.perf_counter() - t0)
            entry = self._mk_entry(h_w, h_s, packed, scales_u32)
            self._cache_put(key, entry)
            self._cache_freq(key)
            self.stat_calls["cache_miss"] = self.stat_calls.get("cache_miss", 0) + 1
        else:
            self._cache_touch(key)
            self._cache_freq(key)
            self.stat_calls["cache_hit"] = self.stat_calls.get("cache_hit", 0) + 1
        h_w, h_s, _width, _wb, _sb = entry
        ext = self.ext
        h_x = ext.allocate_resident(tc * cols * 4)
        h_y = ext.allocate_resident(tc * rows * 4)
        try:
            t0 = _time.perf_counter()
            ext.update_resident(h_x, x.contiguous().float())
            ext.linear_nvfp4_resident_io(
                h_x, h_w, h_s, h_y, tc, rows, cols, gscale,
            )
            self._bump("exp_kernel", _time.perf_counter() - t0)
            t0 = _time.perf_counter()
            out = ext.download_resident(h_y, [tc, rows])
            self._bump("exp_down", _time.perf_counter() - t0)
            return out
        finally:
            ext.free_resident(h_x)
            ext.free_resident(h_y)

    # -- fused gate+up (candidate #10) ----------------------------------
    # gate_proj and up_proj take the SAME input x per token, and the
    # per-expert global scale (weight_scale_2) is identical for gate and
    # up on all 30,720 matrices (verified 10,239/10,239 across 40 layers;
    # downstream_scale/cwise_scale are folded into the same scalar here).
    # So we can stack gate+up packed rows ([2*N, K/2] u8 + [2*N, K/64]
    # scales u32) and compute BOTH with ONE kernel call: one x upload,
    # one dispatch, one download, half the decode calls per layer.
    # The fused bundle gets its own LRU key (id(gate_list), id(up_list),
    # expert) so the two source lists stay live for model lifetime.
    def _fused_gateup_handles(
        self, gate_mat: list, up_mat: list, expert: int,
        batch: bool = False,
    ) -> tuple[int, int, int, int, int]:
        """Return cached (h_w, h_s, width, wb, sb) bundle entry.
        ``batch=True`` records the packed+scales uploads into the currently
        open command-buffer batch instead of a synchronous upload_resident
        (the cache entry is valid once the batch is submitted)."""
        import os
        import time as _time
        if os.environ.get("FTV_NVFP4_GPU", "1") != "1":
            raise RuntimeError("fused gate+up requires the NVFP4 kernel")
        key = (id(gate_mat), id(up_mat), expert)
        entry = self._exp_cache.get(key)
        if entry is None:
            g_packed, g_scales, _g_gs = gate_mat[expert]
            u_packed, u_scales, u_gs = up_mat[expert]

            def _build_fused():
                # One shared implementation (b'): worker spec packs must
                # be byte-identical to the committed path's build.
                return self._build_fused_pack(
                    g_packed, u_packed, g_scales, u_scales)

            t0 = _time.perf_counter()
            packed, scales_u32, was_hit = self._host_pack_get(key, _build_fused)
            if was_hit:
                self._bump("host_pack_hit", _time.perf_counter() - t0)
            else:
                self._bump("exp_scale_pack", _time.perf_counter() - t0)
            t0 = _time.perf_counter()
            # Arena path (arena enabled, layer known): upload into the
            # per-LAYER arena at this expert's slot byte offset.  Handle
            # identity is layer-constant, so the desc key
            # (pipe, arenaW[L], arenaS[L], X, silu[e]) is stable across
            # slot reassignments — the antidote to the per-slot-buffer
            # key explosion that overflowed the 3072-set desc cache.
            if self._arena_enabled:
                layer = self._mat_layer.get(id(gate_mat))
                if layer is not None:
                    sidx = self._arena_assign(layer, expert)
                    h_w = self._arena_wf[layer]
                    h_s = self._arena_sf[layer]
                    w_off = sidx * self._arena_slot_wf_b
                    s_off = sidx * self._arena_slot_sf_b
                    if batch:
                        self.ext.batch_upload(h_w, packed, w_off)
                        self.ext.batch_upload(
                            h_s, scales_u32.contiguous(), s_off)
                    else:
                        self.ext.update_resident(h_w, packed, w_off)
                        self.ext.update_resident(
                            h_s, scales_u32.contiguous(), s_off)
                    entry = self._mk_entry(h_w, h_s, packed, scales_u32)
                    self._cache_put(key, entry)
                    self._cache_freq(key)
                    self._arena_keys.add(key)
                    self._bump("exp_up", _time.perf_counter() - t0)
                    self.stat_calls["cache_miss"] = (
                        self.stat_calls.get("cache_miss", 0) + 1)
                    return entry
            # Legacy path (arena disabled / layer unknown)
            pooled = self._pool_get(
                packed.numel() * packed.element_size(),
                scales_u32.numel() * scales_u32.element_size(),
            )
            if pooled is not None:
                h_w, h_s = pooled
                # Pooled handles are recycled in place; upload into them
                # in the open batch too (task #17) so a re-miss never pays
                # a sync submit+fence per bundle on the decode path.
                if batch:
                    self.ext.batch_upload(h_w, packed)
                    self.ext.batch_upload(h_s, scales_u32.contiguous())
                else:
                    self.ext.update_resident(h_w, packed)
                    self.ext.update_resident(h_s, scales_u32.contiguous())
            elif batch:
                h_w = self.ext.allocate_resident(
                    packed.numel() * packed.element_size())
                h_s = self.ext.allocate_resident(
                    scales_u32.numel() * scales_u32.element_size())
                self.ext.batch_upload(h_w, packed)
                self.ext.batch_upload(h_s, scales_u32.contiguous())
            else:
                h_w = self.ext.upload_resident(packed)
                h_s = self.ext.upload_resident(scales_u32.contiguous())
            self._bump("exp_up", _time.perf_counter() - t0)
            entry = self._mk_entry(h_w, h_s, packed, scales_u32)
            self._cache_put(key, entry)
            self._cache_freq(key)
            self.stat_calls["cache_miss"] = self.stat_calls.get("cache_miss", 0) + 1
        else:
            self._cache_touch(key)
            self._cache_freq(key)
            self.stat_calls["cache_hit"] = self.stat_calls.get("cache_hit", 0) + 1
        return entry

    def _down_handles_batch(
        self, down_mat: list, expert: int, batch: bool = False
    ) -> tuple[int, int]:
        """Return cached (h_w, h_s) for one down matrix, uploading into the
        open batch on first touch (allocate + batch_upload) so a cold cache
        does not force per-expert synchronous submits."""
        import time as _time
        packed, scales, _gscale = down_mat[expert]
        key = (id(down_mat), expert)
        entry = self._exp_cache.get(key)
        if entry is not None:
            self._cache_touch(key)
            self._cache_freq(key)
            self.stat_calls["cache_hit"] = self.stat_calls.get("cache_hit", 0) + 1
            return entry[0], entry[1]
        t0 = _time.perf_counter()
        # Closure hygiene: freeze the packed/scales locals under fresh
        # names BEFORE the lambda so a future rebind of `packed` /
        # `scales_u32` below can never leak into the captured build.
        p, s = packed, scales
        packed, scales_u32, _hit = self._host_pack_get(
            key, lambda: self._build_down_pack(p, s))
        if _hit:
            self._bump("host_pack_hit", _time.perf_counter() - t0)
        else:
            self._bump("exp_scale_pack", _time.perf_counter() - t0)
        t0 = _time.perf_counter()
        # Arena path (arena enabled, layer known): upload into the
        # per-LAYER down arena at this expert's slot byte offset.
        if self._arena_enabled:
            layer = self._mat_layer.get(id(down_mat))
            if layer is not None:
                sidx = self._arena_assign(layer, expert)
                h_w = self._arena_wd[layer]
                h_s = self._arena_sd[layer]
                w_off = sidx * self._arena_slot_wd_b
                s_off = sidx * self._arena_slot_sd_b
                if batch:
                    self.ext.batch_upload(h_w, packed.contiguous(), w_off)
                    self.ext.batch_upload(
                        h_s, scales_u32.contiguous(), s_off)
                else:
                    self.ext.update_resident(h_w, packed.contiguous(), w_off)
                    self.ext.update_resident(
                        h_s, scales_u32.contiguous(), s_off)
                self._bump("exp_up", _time.perf_counter() - t0)
                self._cache_put(key, (
                    h_w, h_s,
                    packed.numel() + scales_u32.numel() * 4,
                    packed.numel() * packed.element_size(),
                    scales_u32.numel() * scales_u32.element_size()))
                self._cache_freq(key)
                self._arena_keys.add(key)
                self.stat_calls["cache_miss"] = (
                    self.stat_calls.get("cache_miss", 0) + 1)
                return h_w, h_s
        # Legacy path (arena disabled / layer unknown)
        pooled = self._pool_get(
            packed.numel() * packed.element_size(),
            scales_u32.numel() * scales_u32.element_size(),
        )
        if pooled is not None:
            h_w, h_s = pooled
            # Same in-batch rule as fused gate+up (task #17): pooled
            # handles get batch_upload on the decode path, never sync.
            if batch:
                self.ext.batch_upload(h_w, packed.contiguous())
                self.ext.batch_upload(h_s, scales_u32.contiguous())
            else:
                self.ext.update_resident(h_w, packed.contiguous())
                self.ext.update_resident(h_s, scales_u32.contiguous())
        elif batch:
            h_w = self.ext.allocate_resident(
                packed.numel() * packed.element_size())
            h_s = self.ext.allocate_resident(
                scales_u32.numel() * scales_u32.element_size())
            self.ext.batch_upload(h_w, packed.contiguous())
            self.ext.batch_upload(h_s, scales_u32.contiguous())
        else:
            h_w = self.ext.upload_resident(packed.contiguous())
            h_s = self.ext.upload_resident(scales_u32.contiguous())
        self._bump("exp_up", _time.perf_counter() - t0)
        entry = self._mk_entry(h_w, h_s, packed, scales_u32)
        self._cache_put(key, entry)
        self._cache_freq(key)
        self.stat_calls["cache_miss"] = self.stat_calls.get("cache_miss", 0) + 1
        return h_w, h_s

    def _expert_affine_gateup_silu(
        self, x: torch.Tensor, gate_mat: list, up_mat: list, expert: int,
        skip_x: bool = False,
    ) -> None:
        """Fused gate+up with the SiLU fold (task #11): ONE kernel call
        computes gate[n] and up[n] from the stacked bundle (rows n and
        n+N) and writes Y = silu(gate)*up into SELF.H_SILU (resident
        scratch, no download). The down kernel then reads that scratch as
        its X, resident-to-resident. ``skip_x=True`` reuses the h the
        caller already uploaded to self.h_x_fused (same h for all 8
        experts of a layer)."""
        import time as _time
        g_packed, _g_scales, g_gs = gate_mat[expert]
        rows = g_packed.shape[-2]
        cols = g_packed.shape[-1] * 2
        try:
            fused_gs = float(g_gs)
        except (TypeError, ValueError, RuntimeError):
            raise RuntimeError(f"bad fused gate+up gscale for expert {expert}") from None
        h_w, h_s, _width, _wb, _sb = self._fused_gateup_handles(gate_mat, up_mat, expert)
        ext = self.ext
        if not skip_x:
            t0 = _time.perf_counter()
            ext.update_resident(self.h_x_fused, x.contiguous().float())
            self._bump("exp_x", _time.perf_counter() - t0)
        t0 = _time.perf_counter()
        ext.linear_nvfp4_gateup_silu_io(
            self.h_x_fused, h_w, h_s, self.h_silu,
            1, rows, cols, fused_gs,
            self._fused_slot0(gate_mat, expert),
        )
        self._bump("exp_kernel", _time.perf_counter() - t0)

    def _expert_affine_gateup_silu_batch(
        self, x: torch.Tensor, gate_mat: list, up_mat: list, expert: int
    ) -> None:
        """Silu-fold gate+up for Tc>1 tokens: writes [Tc, N]
        silu(gate)*up into self.h_silu via the T-parametrized kernel.
        Caller must then call _expert_affine_down_from_silu with the
        same tc before reusing h_silu."""
        import time as _time
        g_packed, _g_scales, g_gs = gate_mat[expert]
        rows = g_packed.shape[-2]
        cols = g_packed.shape[-1] * 2
        try:
            fused_gs = float(g_gs)
        except (TypeError, ValueError, RuntimeError):
            raise RuntimeError(f"bad fused gate+up batch gscale for expert {expert}") from None
        h_w, h_s, _width, _wb, _sb = self._fused_gateup_handles(gate_mat, up_mat, expert)
        tc = x.shape[0]
        ext = self.ext
        h_x = ext.allocate_resident(tc * cols * 4)
        try:
            t0 = _time.perf_counter()
            ext.update_resident(h_x, x.contiguous().float())
            ext.linear_nvfp4_gateup_silu_io(
                h_x, h_w, h_s, self.h_silu,
                tc, rows, cols, fused_gs,
                self._fused_slot0(gate_mat, expert),
            )
            self._bump("exp_kernel", _time.perf_counter() - t0)
        finally:
            ext.free_resident(h_x)

    def _expert_affine_down_from_silu(
        self, down_mat: list, expert: int, tc: int
    ) -> torch.Tensor:
        """Down projection reading its input from self.h_silu (the
        silu-fold scratch), no host upload: Y[Tc, N_down] =
        silu(gate)*up @ Wd^T.  Same per-expert down cache as the classic
        path (shared _down_handles_batch cache so prefill + decode hit
        the SAME arena entries — no legacy pooled leftovers)."""
        import time as _time
        packed, scales, gscale = down_mat[expert]
        rows = packed.shape[-2]
        cols = packed.shape[-1] * 2
        # Shared cache+arena path (prefill -> arena, bit-transparent
        # via slot0=0 when the arena is disabled).
        h_w, h_s = self._down_handles_batch(down_mat, expert, batch=False)
        slot0 = self._down_slot0(down_mat, expert)
        ext = self.ext
        try:
            down_gs = float(gscale)
        except (TypeError, ValueError, RuntimeError):
            raise RuntimeError(f"bad down gscale for expert {expert}") from None
        t0 = _time.perf_counter()
        ext.linear_nvfp4_resident_io(
            self.h_silu, h_w, h_s, self.h_y,
            tc, rows, cols, down_gs, slot0,
        )
        self._bump("exp_kernel", _time.perf_counter() - t0)
        t0 = _time.perf_counter()
        out = ext.download_resident(self.h_y, [tc, rows])
        self._bump("exp_down", _time.perf_counter() - t0)
        return out

    def _moe_step(self, i: int, h: torch.Tensor) -> torch.Tensor:
        """Decode-path MoE with fused gate+up (candidate #10): the 8
        selected experts per layer each need gate=Wg·h and up=Wu·h on the
        SAME h; stacking the two NVFP4 matrices lets one kernel call (one
        x upload, one dispatch, one download) produce both. Down stays a
        separate per-expert kernel on silu(gate)·up."""
        import os
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
        fused = os.environ.get("FTV_NVFP4_GPU", "1") == "1"
        if fused:
            # Same h for all 8 experts: one upload, 8 kernel calls.
            import time as _time
            t0 = _time.perf_counter()
            self.ext.update_resident(self.h_x_fused, h.contiguous().float())
            self._bump("exp_x", _time.perf_counter() - t0)
        for slot, expert in enumerate(index.tolist()):
            if fused:
                self._expert_affine_gateup_silu(
                    h, gate_mat, up_mat, expert, skip_x=True,
                )
                down = self._expert_affine_down_from_silu(down_mat, expert, 1)[0]
            else:
                gate = self._expert_affine(h, gate_mat, expert)
                up = self._expert_affine(h, up_mat, expert)
                down = self._expert_affine(F.silu(gate) * up, down_mat, expert)
            out = out + values[slot].item() * down
        return out

    def _gdn_chunk(self, i: int, x_chunk: torch.Tensor) -> torch.Tensor:
        """Batched linear-attention layer over a whole chunk.

        The five dense in-projections are computed once for all Tc rows
        (one GPU affine each via ``_pinned_affine_batch``); the conv1d
        history + GDN state recurrence stays strictly sequential per row
        (pure torch, no per-row uploads); ``out_proj`` is one batched
        affine. Returns ``[Tc, H]``.
        """
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        qkv = self._pinned_affine_batch(x_chunk, f"{base}.linear_attn.in_proj_qkv")
        z = self._pinned_affine_batch(x_chunk, f"{base}.linear_attn.in_proj_z")
        a = self._pinned_affine_batch(x_chunk, f"{base}.linear_attn.in_proj_a")
        b = self._pinned_affine_batch(x_chunk, f"{base}.linear_attn.in_proj_b")

        conv_w = self._dense(f"{base}.linear_attn.conv1d.weight")
        if conv_w.dim() == 3 and conv_w.shape[1] == 1:
            taps = conv_w[:, 0, :].float()  # [C, K] (loader permuted [C,K,1])
        else:
            taps = conv_w.float().reshape(conv_w.shape[0], -1)
        nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        key_dim = nk * hk
        value_dim = nv * hv

        buf = self.conv_buf[i]
        a_log = self._dense(f"{base}.linear_attn.A_log")
        dt_bias = self._dense(f"{base}.linear_attn.dt_bias")
        norm_w = self._dense(f"{base}.linear_attn.norm.weight")
        scale = 1.0 / math.sqrt(hk)
        state = self.gdn_state[i]
        tc = x_chunk.shape[0]
        gated_all = torch.empty((tc, nv * hv), dtype=torch.float32)
        for t in range(tc):
            qv = qkv[t]
            mixed = (taps[:, :3] * buf).sum(-1) + taps[:, 3] * qv
            buf[:, 0] = buf[:, 1]
            buf[:, 1] = buf[:, 2]
            buf[:, 2] = qv
            mixed = F.silu(mixed)
            query, key, value = torch.split(
                mixed, [key_dim, key_dim, value_dim], dim=-1
            )
            query = query.reshape(nk, hk).repeat_interleave(nv // nk, dim=0)
            key = key.reshape(nk, hk).repeat_interleave(nv // nk, dim=0)
            value = value.reshape(nv, hv)
            beta = torch.sigmoid(b[t])
            g = -torch.exp(a_log) * F.softplus(a[t] + dt_bias)
            qt = l2norm(query) * scale
            kt = l2norm(key)
            state.mul_(torch.exp(g)[:, None, None])
            kv_mem = (state * kt[:, :, None]).sum(dim=1)
            delta = (value - kv_mem) * beta[:, None]
            state.add_(kt[:, :, None] * delta[:, None, :])
            core = (state * qt[:, :, None]).sum(dim=1)
            var = core.pow(2).mean(-1, keepdim=True)
            normed = core * torch.rsqrt(var + cfg.rms_norm_eps)
            gated_all[t] = (
                normed * norm_w * F.silu(z[t].reshape(nv, hv))
            ).reshape(nv * hv)
        return self._pinned_affine_batch(gated_all, f"{base}.linear_attn.out_proj")

    def _full_attn_chunk(
        self, i: int, x_chunk: torch.Tensor, pos: int
    ) -> torch.Tensor:
        """Chunked full-attention layer (task #9). Batches q/k/v/o
        projections across Tc, writes the chunk into the KV caches, and
        runs causal SDPA over the whole cache prefix."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        hq, hkv, hd = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        tc = x_chunk.shape[0]
        qg = self._pinned_affine_batch(x_chunk, f"{base}.self_attn.q_proj")
        query, gate = qg.view(tc, hq, 2 * hd).chunk(2, dim=-1)
        key = self._pinned_affine_batch(
            x_chunk, f"{base}.self_attn.k_proj"
        ).view(tc, hkv, hd)
        value = self._pinned_affine_batch(
            x_chunk, f"{base}.self_attn.v_proj"
        ).view(tc, hkv, hd)

        q_norm = self._dense(f"{base}.self_attn.q_norm.weight")
        k_norm = self._dense(f"{base}.self_attn.k_norm.weight")
        query = rms_norm_head(query, q_norm, cfg.rms_norm_eps).float()
        key = rms_norm_head(key, k_norm, cfg.rms_norm_eps).float()

        cos = self.rope_cos[pos : pos + tc].unsqueeze(1)
        sin = self.rope_sin[pos : pos + tc].unsqueeze(1)
        query = apply_partial_rotary(query, cos, sin, self.rotary_dim)
        key = apply_partial_rotary(key, cos, sin, self.rotary_dim)

        if self.k_cache[i].dtype == torch.bfloat16:
            self.k_cache[i][pos : pos + tc].copy_(key.to(torch.bfloat16))
            self.v_cache[i][pos : pos + tc].copy_(value.to(torch.bfloat16))
            keys = self.k_cache[i][: pos + tc].float()
            values = self.v_cache[i][: pos + tc].float()
        else:
            self.k_cache[i][pos : pos + tc].copy_(key)
            self.v_cache[i][pos : pos + tc].copy_(value)
            keys = self.k_cache[i][: pos + tc]
            values = self.v_cache[i][: pos + tc]
        P = pos + tc
        repeat = hq // hkv
        if repeat > 1:
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)
        q = query.transpose(0, 1)  # [Hq, Tc, P]
        k = keys.transpose(0, 1)  # [Hq, P, D]
        v = values.transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)  # [Hq, Tc, P]
        row = torch.arange(tc) + pos
        col = torch.arange(P)
        allowed = col.unsqueeze(0) <= row.unsqueeze(1)  # [Tc, P]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        context = (probs @ v).transpose(0, 1).reshape(tc, hq * hd)
        context = context * torch.sigmoid(gate.reshape(tc, hq * hd))
        return self._pinned_affine_batch(context, f"{base}.self_attn.o_proj")

    def _moe_chunk(self, i: int, h_chunk: torch.Tensor) -> torch.Tensor:
        """Batched MoE layer (task #9): one router affine for Tc rows, one
        batched shared-expert pass, then grouped expert calls — every
        distinct (layer, expert, proj) runs the T>1 kernel once with all
        chunk tokens that route to it, instead of Tc×8 T=1 calls."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        tc = h_chunk.shape[0]
        router_w = self._dense(f"{base}.mlp.gate.weight")
        logits = F.linear(h_chunk.float(), router_w.float()).float()
        probs = torch.softmax(logits, dim=-1)
        values, index = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)

        shared_g = self._pinned_affine_batch(h_chunk, f"{base}.mlp.shared_expert.gate_proj")
        shared_u = self._pinned_affine_batch(h_chunk, f"{base}.mlp.shared_expert.up_proj")
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
        # Materialize routing as plain Python lists once (cheap, Tc×8)
        # so the grouping loop never touches torch scalars.
        exp_slots: list[list[int]] = index.tolist()
        val_slots: list[list[float]] = values.tolist()
        # Group (token, slot) by expert; each expert gets one T>1 kernel.
        groups: dict[int, list[int]] = {}
        for t in range(tc):
            for expert in exp_slots[t]:
                groups.setdefault(expert, []).append(t)
        for expert, rows in groups.items():
            xg = h_chunk[rows]
            self._expert_affine_gateup_silu_batch(xg, gate_mat, up_mat, expert)
            down = self._expert_affine_down_from_silu(down_mat, expert, len(rows))
            for j, t in enumerate(rows):
                wsum = 0.0
                for slot, e in enumerate(exp_slots[t]):
                    if e == expert:
                        wsum += val_slots[t][slot]
                out[t] = out[t] + wsum * down[j]
        return out

    def prefill_chunked(
        self, ids: list[int], chunk_size: int = 128, log_progress: bool = False
    ) -> torch.Tensor:
        """Chunked prefill: process the whole prompt with batched Tc>1
        affines plus chunked full-attn SDPA; GDN state recurrence stays
        exact per row. Returns logits for the LAST prompt token."""
        if not ids:
            raise ValueError("ids must be non-empty")
        cfg = self.config
        embed_rows = [self.embed_row(tid) for tid in ids]
        logits = None
        for start in range(0, len(ids), chunk_size):
            piece = embed_rows[start : start + chunk_size]
            x = torch.stack(piece)  # [Tc, H]
            for i, layer_type in enumerate(cfg.layer_types):
                base = f"language_model.model.layers.{i}"
                h = self._rms(
                    x, self._dense(f"{base}.input_layernorm.weight"),
                    cfg.rms_norm_eps,
                )
                if layer_type == "linear_attention":
                    x = x + self._gdn_chunk(i, h)
                else:
                    x = x + self._full_attn_chunk(i, h, start)
                h2 = self._rms(
                    x, self._dense(f"{base}.post_attention_layernorm.weight"),
                    cfg.rms_norm_eps,
                )
                x = x + self._moe_chunk(i, h2)
            x = self._rms(
                x, self._dense(f"language_model.model.norm.weight"),
                cfg.rms_norm_eps,
            )
            last = x[-1:]
            logits = self._pinned_affine(last, f"language_model.lm_head")
            self._seq_len = start + len(piece)
            if log_progress:
                print(f"prefill {self._seq_len}/{len(ids)}...", flush=True)
            if self._seq_len >= self.max_len:
                break
        if logits is None:
            raise RuntimeError("empty prompt in prefill_chunked")
        # Host GDN state advanced -> GPU mirrors are stale (task #13).
        self._gdn_gpu_dirty = True
        return logits

    def generate(
        self, prompt_ids: list[int], max_new: int = 64, eos: int | None = None,
        prefill_chunk: int = 1024, log_progress: bool = False,
        temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
        seed: int = 1234, batched: bool = False,
    ) -> list[int]:
        """NVFP4 generate: chunked prefill for prompts (task #9), then the
        per-token decode loop from the base class.

        ``prefill_chunk`` is a real batching parameter here, not just a
        progress-log stride: >1 picks the batched Tc>1 path (5 GDN in-projs
        per layer become one affine each, full-attn q/k/v/o are batched,
        experts are grouped so each distinct (layer, expert, proj) runs the
        T>1 kernel once). Tiny prompts (<= chunk) still work and now use the
        batched path below the truncation.

        ``batched=True`` switches the DECODE loop to the task #12 batched
        path (``decode_step_batched``): each layer = 2 command-buffer
        submissions with resident-to-resident expert accumulation, instead
        of ~28 submit/fence roundtrips.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if temperature < 0.0 or top_k < 0 or not 0.0 < top_p <= 1.0:
            raise ValueError(
                f"bad sampling params temperature={temperature} "
                f"top_k={top_k} top_p={top_p}"
            )
        rng = torch.Generator().manual_seed(seed) if temperature > 0.0 else None
        self.reset()
        out: list[int] = []
        pos = 0
        decode = self.decode_step_batched if batched else self.decode_step
        chunk = max(1, min(prefill_chunk, len(prompt_ids)))
        if chunk > 1:
            # Batched chunked prefill; returns logits for the final token
            # and leaves KV/state advanced for the whole prompt.
            logits = self.prefill_chunked(
                prompt_ids, chunk_size=chunk, log_progress=log_progress
            )
            pos = self._seq_len
        else:
            logits = torch.zeros(self.config.vocab_size)
            for token_id in prompt_ids:
                logits = decode(self.embed_row(token_id), pos)
                pos += 1
                self._seq_len = pos
                if log_progress and (pos % max(1, prefill_chunk) == 0):
                    print(f"prefill {pos}/{len(prompt_ids)}...", flush=True)
                if pos >= self.max_len:
                    break
        for _ in range(max_new):
            if pos >= self.max_len:
                break
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "non-finite logits at position "
                    f"{pos}: check weight loads and kernel outputs"
                )
            try:
                next_id = self._sample(logits, rng, temperature, top_k, top_p)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"sampling failed at position {pos}"
                ) from exc
            out.append(next_id)
            if eos is not None and next_id == eos:
                break
            logits = decode(self.embed_row(next_id), pos)
            pos += 1
        return out

    # -- task #12: batched decode path ---------------------------------
    # Task #22 (GPU router, oracle-3 R1-R5): GPU-resident RMSNorm -> fp32
    # router GEMM -> topk-256x8 -> arena ids/gs write feeding the fused
    # MoE dispatches.  Helpers below: _gpu_router_sync_map,
    # _gpu_router_host_topk, _gpu_router_crosscheck, _record_moe_gpu.

    def _gpu_router_sync_map(self, i: int) -> None:
        """Upload the host arena map (expert -> slot) to the resident map
        buffer so the GPU topk resolves slot indices against CURRENT
        assignments.  Dirty is set by _arena_assign/_arena_drop (which
        run in prefill and in the pin/prewarm loops); the upload is one
        1 KiB vkCmdCopyBuffer inside the open batch-3 before the topk.

        Absent experts are filled with _ARENA_NO_SLOT (0x7FFFFFFF) — the
        SINGLE sentinel shared with the topk256_k8_f32.comp kernel's
        NO_SLOT constant (oracle-8 fix 1); the kernel maps it to the
        reserved no-content slot 0, never a real assignment."""
        if not self._gpu_router or not self._arena_map_dirty[i]:
            return
        E = self.config.num_experts
        mp = self._arena_map[i]
        arr = torch.full((E,), _ARENA_NO_SLOT, dtype=torch.int32)
        for _e, _slot in mp.items():
            if _slot < _ARENA_NO_SLOT:
                arr[_e] = _slot
        self.ext.batch_upload(self._arena_map_res[i], arr, 0)
        self._arena_map_dirty[i] = False

    def _gpu_router_prewarm(self, i: int, elist_p: list[int]) -> None:
        """Arena warm-up for the CURRENT token's PREDICTED routed set
        (oracle-8 fix 3): the host-shadow router ran on host h2 (same
        math as the GPU topk, validated bit-parity on non-tie rows) and
        this assigns any predicted expert missing from the arena map NOW
        — BEFORE _gpu_router_sync_map uploads the map for this token's
        batch-3 topk — so the GPU topk resolves those experts to REAL
        slots and the in-batch fused MoE record reads content we upload
        as cheap copies (no re-record submit).  Also touches in-map
        predicted experts to the LRU end (recently-routed protection:
        the map LRU order now follows routing recency, so eviction never
        hits a stable member of the recent window).

        Newly-assigned (expert, slot) pairs are queued on
        self._prewarm_pending[i] and emitted as QUEUE-2 async content
        uploads by _gpu_router_prewarm_async_uploads (task #29) right
        after this, still before batch_begin; the consumer batch's
        batch_cross_barrier + batch_submit([cookie]) make the copies
        visible to the fused MoE dispatches.

        SAFETY: an expert evicted here to make room was NOT predicted by
        the host this token, so the GPU topk (identical math) will not
        route it either (except rare near-tie flips, which read slot 0
        and hit the post-submit pin re-record).  No silent slot reuse:
        the topk reads the map uploaded AFTER these assignments."""
        mp = self._arena_map[i]
        pend = []
        for e in elist_p:
            if e in mp:
                mp.move_to_end(e)   # recently-routed protection
            else:
                sidx = self._arena_assign(i, e)
                pend.append((e, sidx))
        self._prewarm_pending[i] = pend

    def _gpu_router_prewarm_async_uploads(self, i: int) -> int:
        """Emit the arena CONTENT uploads queued by _gpu_router_prewarm
        (fused + down packs at each newly-assigned expert's slot offset)
        as queue-2 async H2D copies and flush them.

        Task #29 (async H2D overlap): the copies go through
        batch_upload_async on the SECOND queue family (dual-buffered
        upload command buffers in C++) instead of being recorded into
        batch-3 — the DMA overlaps this host record window and the main
        queue no longer executes them serially ahead of the MoE
        dispatches.  Returns the flush cookie (-1 when nothing is
        pending).  MUST run OUTSIDE the open main batch
        (batch_upload_async requires it); the consumer batch records
        batch_cross_barrier at batch_begin (queue-2 -> main-queue memory
        visibility) and submits with batch_submit([cookie]) (semaphore
        ordering), so the fused MoE dispatches still see the content.

        Reads the host pack cache (HIT-happy — prefill populated it).
        The I1 (oracle-10) stale-slot check is preserved: a pending
        (expert, slot) pair that is no longer the map's CURRENT
        assignment would silently write content into a slot the topk no
        longer maps the expert to.  No mutation happens on this path
        today, so it is a defensive invariant, but if it ever fires it
        must be LOUD."""
        pend = self._prewarm_pending.get(i)
        if not pend:
            return -1
        self._prewarm_pending[i] = []
        # I1 (oracle-10): close the prewarm assign->content window.
        # Re-verify each pending (expert, slot) pair is STILL the map's
        # CURRENT assignment before writing content; a stale pair
        # (expert evicted/reassigned between _gpu_router_prewarm and
        # here) would silently write content into a slot the topk no
        # longer maps the expert to.  (Message kept verbatim from the
        # in-batch version.)
        mp = self._arena_map[i]
        for e, sidx in pend:
            if mp.get(e) != sidx:
                raise RuntimeError(
                    f"layer {i}: prewarm pending (expert {e}, slot {sidx}) "
                    f"is STALE; arena map now has "
                    f"{mp.get(e)} (expert evicted/reassigned between "
                    f"prewarm assign and content upload); refusing a "
                    f"silent stale-slot content write")
        # I1 (oracle-10): close the prewarm assign->content window.
        # Re-verify each pending (expert, slot) pair is STILL the map's
        # CURRENT assignment before writing content; a stale pair (expert
        # evicted/reassigned between _gpu_router_prewarm and here) would
        # silently write content into a slot the topk no longer maps the
        # expert to.  No mutation happens on this path today, so this is
        # a defensive invariant, but if it ever fires it must be LOUD.
        mp = self._arena_map[i]
        for e, sidx in pend:
            if mp.get(e) != sidx:
                raise RuntimeError(
                    f"layer {i}: prewarm pending (expert {e}, slot {sidx}) "
                    f"is STALE; arena map now has "
                    f"{mp.get(e)} (expert evicted/reassigned between "
                    f"prewarm assign and content upload); refusing a "
                    f"silent stale-slot content write")
        base = f"language_model.model.layers.{i}"
        gm = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        um = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        dm = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        _n_experts = 0
        for e, sidx in pend:
            _fk = (id(gm), id(um), e)
            _dk = (id(dm), e)
            _gp, _gu, _ = gm[e]
            _up, _uu, _ = um[e]
            _fp, _fs, _fh = self._host_pack_get(
                _fk, lambda: self._build_fused_pack(_gp, _up, _gu, _uu))
            self.ext.batch_upload_async(
                self._arena_wf[i], _fp, sidx * self._arena_slot_wf_b)
            self.ext.batch_upload_async(
                self._arena_sf[i], _fs.contiguous(),
                sidx * self._arena_slot_sf_b)
            _dp, _ds, _ = dm[e]
            _dq, _ds32, _dh = self._host_pack_get(
                _dk, lambda: self._build_down_pack(_dp, _ds))
            self.ext.batch_upload_async(
                self._arena_wd[i], _dq.contiguous(),
                sidx * self._arena_slot_wd_b)
            self.ext.batch_upload_async(
                self._arena_sd[i], _ds32.contiguous(),
                sidx * self._arena_slot_sd_b)
            _n_experts += 1
        self.stat_calls["prewarm_async"] = (
            self.stat_calls.get("prewarm_async", 0) + 1)
        self.stat_calls["prewarm_async_experts"] = (
            self.stat_calls.get("prewarm_async_experts", 0) + _n_experts)
        return int(self.ext.batch_upload_async_flush())

    def _gpu_router_host_topk(
        self, i: int, h2: torch.Tensor,
    ) -> tuple[list[int], list[float]]:
        """Host router + topk + canonical-sort on a given h2 (the
        ORIGINAL _record_moe_in math, verbatim — same fp32 F.linear,
        softmax, torch.topk, renorm, canonical sort).  Used TWO ways:
        (1) the arena prewarm shadow (oracle-8 fix 3) runs it on host h2
        EVERY token to predict the GPU topk's routed set (bit-parity on
        non-tie rows, validated by test_topk_parity) so the arena can be
        warmed in-batch; (2) the hard-fall-to-host path and CHECK-mode
        cross-validation (mismatched ids -> _gpu_router_host_list).
        Returns canonical (elist, vlist)."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        router_w = self._dense(f"{base}.mlp.gate.weight")
        logits = F.linear(h2.float(), router_w.float()).float()
        probs = torch.softmax(logits, dim=-1)
        values, index = torch.topk(
            probs, cfg.num_experts_per_tok, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)
        vlist = (values.tolist() if values.dim() == 1
                 else values.tolist()[0])
        elist = (index.tolist() if index.dim() == 1
                 else index.tolist()[0])
        _pairs = sorted(zip(elist, vlist))
        elist = [_e for _e, _v in _pairs]
        vlist = [_v for _e, _v in _pairs]
        return elist, vlist

    def _gpu_router_crosscheck(
        self, i: int, h2_host: torch.Tensor, h2_gpu: torch.Tensor,
        lg_gpu: torch.Tensor, elist: list[int], vlist: list[float],
        e_h: list[int] | None = None, v_h: list[float] | None = None,
    ) -> None:
        """CHECK-mode cross-validation: GPU-h2 vs host-h2 (<=1e-5/layer)
        and paired (id, renormalized scale) vs host after canonical sort
        (END-TO-END gate <=1e-5 RELATIVE — oracle-8 fix 2: <=1e-6 is
        BELOW the fp32 K=2048 GEMM accumulation noise floor once h2
        differences (RMS order, ~1e-7) amplify through the 256-wide
        router GEMM and the renorm; the kernel-level pair parity on
        IDENTICAL logits is still checked at <=1e-6 in test_topk_parity
        (no h2/GEMM noise there).  Also attributes the logit diff to the
        RMS vs the fp32 GEMM accumulation order (lg_on_gpu_h2 measures
        GEMM alone).  Mismatched ids => tie case -> hard-fall-to-host
        (the caller re-records with _gpu_router_host_list)."""
        cfg = self.config
        d = (h2_host - h2_gpu).abs().max().item()
        self._h2_maxdiff[i] = max(self._h2_maxdiff[i], d)
        if d > 1e-5:
            self._gpu_router_fails += 1
        base = f"language_model.model.layers.{i}"
        router_w = self._dense(f"{base}.mlp.gate.weight")
        lg_h = F.linear(h2_host.float(), router_w.float()).float()
        lg_g = lg_gpu.float()
        self._lg_maxdiff_h = max(self._lg_maxdiff_h,
                                 (lg_h - lg_g).abs().max().item())
        # GEMM-only attribution: host GEMM on GPU h2.
        lg_on_gpu_h2 = F.linear(h2_gpu.float(), router_w.float()).float()
        self._lg_maxdiff_gemm = max(
            self._lg_maxdiff_gemm,
            (lg_on_gpu_h2 - lg_g).abs().max().item())
        # Host topk for the pair gate (reuse the prewarm's result when
        # the caller already computed it — no double math).
        if e_h is None or v_h is None:
            e_h, v_h = self._gpu_router_host_topk(i, h2_host)
        if e_h != elist:
            self._gpu_router_ties += 1
            self._gpu_router_fallback = True
            self._gpu_router_host_list = (e_h, v_h)
            return
        worst = 0.0
        for a, b in zip(v_h, vlist):
            denom = max(abs(b), 1e-12)
            rel = abs(a - b) / denom
            if rel > worst:
                worst = rel
        self._pair_maxdiff = max(self._pair_maxdiff, worst)
        # End-to-end (h2 -> GEMM -> renorm) pair gate: 1e-5 relative
        # (oracle-8 fix 2; 1e-6 is below the fp32 K=2048 noise floor).
        if worst > 1e-5:
            self._gpu_router_fails += 1
            if self._gpu_router_check:
                # I3 (oracle-10): the pair gate is FUNCTIONAL in CHECK=1
                # — a >1e-5 breach is a hard RuntimeError, not a silent
                # counter (pair_maxdiff <= 1e-5 asserted).  CHECK=0 stays
                # measured-only (counters, no crosscheck call).
                raise RuntimeError(
                    f"layer {i}: GPU-router pair gate breached: "
                    f"pair_maxdiff {self._pair_maxdiff:.3e} > 1e-5 "
                    f"(ids {elist}, host vals "
                    f"{[round(x, 6) for x in v_h]}, gpu vals "
                    f"{[round(x, 6) for x in vlist]})")

    def _record_moe_gpu(
        self, i: int, ext: Any, elist: list[int] | None,
        vlist: list[float] | None, *, upload_ids_gs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-mode MoE record (fused arena path; the GPU topk already
        wrote _arena_ids/_arena_gs in the open batch).

        elist/vlist: canonical expert ids + renorm values from the topk
        readback.  The FIRST (in-batch) record passes None (readback not
        materialized yet) — no pin loop, no host upload: the dispatches
        read the topk-written ids/gs directly.  A pin/tie re-record
        passes the canonical list + upload_ids_gs=True so the host
        re-uploads corrected ids/gs + arena content (the rare
        correctness path; the old "inert" host upload is gone from the
        common path).

        Returns (acc_t, sd_t, gsc_t)."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        gate_mat = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        up_mat = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        down_mat = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        lc = self._moe_layer_c[i]
        gu_rows, gu_cols, dn_rows, dn_cols = lc
        gate_rows = cfg.hidden_size // 4
        sg = self.pinned[_translate(f"{base}.mlp.shared_expert.gate_proj")]
        su = self.pinned[_translate(f"{base}.mlp.shared_expert.up_proj")]
        gsc = self.pinned[_translate(f"{base}.mlp.shared_expert_gate")]
        sdp = self.pinned[_translate(
            f"{base}.mlp.shared_expert.down_proj")]
        _mp = self._arena_map[i]
        if elist is not None:
            # Pin loop (fixpoint, same as the fused branch of
            # _record_moe_in): re-assign missing routed experts and
            # re-upload their arena content (LRU-evicted slots are
            # reused; uploads read the host pack cache, HIT-happy).
            for _pin in range(9):
                _missing = [e for e in elist if _mp.get(e) is None]
                if not _missing:
                    break
                for e in _missing:
                    sidx = self._arena_assign(i, e)
                    _fk = (id(gate_mat), id(up_mat), e)
                    _dk = (id(down_mat), e)
                    _gp, _gu, _ = gate_mat[e]
                    _up, _uu, _ = up_mat[e]
                    _fp, _fs, _fh = self._host_pack_get(
                        _fk, lambda: self._build_fused_pack(
                            _gp, _up, _gu, _uu))
                    self.ext.batch_upload(
                        self._arena_wf[i], _fp,
                        sidx * self._arena_slot_wf_b)
                    self.ext.batch_upload(
                        self._arena_sf[i], _fs.contiguous(),
                        sidx * self._arena_slot_sf_b)
                    _dp, _ds, _ = down_mat[e]
                    _dq, _ds32, _dh = self._host_pack_get(
                        _dk, lambda: self._build_down_pack(_dp, _ds))
                    self.ext.batch_upload(
                        self._arena_wd[i], _dq.contiguous(),
                        sidx * self._arena_slot_wd_b)
                    self.ext.batch_upload(
                        self._arena_sd[i], _ds32.contiguous(),
                        sidx * self._arena_slot_sd_b)
                    self._bump("exp_up", 0.0)
            _routed_missing = [e for e in elist if _mp.get(e) is None]
            if _routed_missing:
                raise RuntimeError(
                    f"layer {i}: routed expert(s) {_routed_missing} have no "
                    f"arena slot after the pin loop (arena_n="
                    f"{self._arena_n}, routed={len(elist)}); refusing "
                    f"silent slot-0 fallback")
            if self._arena_n - 1 < len(elist):
                raise RuntimeError(
                    f"layer {i}: arena has {self._arena_n - 1} usable slots "
                    f"(slot 0 reserved) but "
                    f"{len(elist)} routed experts need one each")
        if upload_ids_gs:
            # Host-corrected ids/gs (pin/tie re-record).  Same fp64->fp32
            # RTNE conversion rule as the C++ (float) cast.
            gu_sc = [float(gate_mat[_e][2]) for _e in elist]
            dn_sc = [float(down_mat[_e][2]) * vlist[_ei]
                     for _ei, _e in enumerate(elist)]
            _ids = torch.tensor(
                [_mp[_e] for _e in elist], dtype=torch.int32)
            _gs = torch.tensor(gu_sc + dn_sc, dtype=torch.float32)
            ext.batch_upload(self._arena_ids[i], _ids, 0)
            ext.batch_upload(self._arena_gs[i], _gs, 0)
        acc_t = torch.empty(cfg.hidden_size)
        sd_t = torch.empty(cfg.hidden_size)
        gsc_t = torch.empty(gsc.rows)
        ext.batch_moe_fused_record(
            self.h_x_fused, self.h_y_acc,
            self.h_sg, self.h_su, self.h_gsc,
            self.h_x3, self.h_sdown,
            self.h_silu_all,
            self._arena_wf[i], self._arena_sf[i],
            self._arena_wd[i], self._arena_sd[i],
            self._arena_ids[i], self._arena_gs[i],
            gu_rows, gu_cols, self._arena_rows_fused,
            dn_rows, dn_cols, self._arena_rows_down,
            gu_rows,  # silu_N: X base stride per expert in silu_all
            sg.handle, sg.rows, sg.cols,
            su.handle, su.rows, su.cols,
            gsc.handle, gsc.rows, gsc.cols,
            sdp.handle, sdp.rows, sdp.cols,
            gate_rows,
            acc_t, sd_t, gsc_t,
        )
        return acc_t, sd_t, gsc_t

    # One layer = TWO command-buffer submissions (input side, output side)
    # instead of ~28 submit+fence roundtrips. Every GPU stage of a layer
    # (dense in-projs, 8 routed experts, shared expert, out_proj) records
    # into a single batch; the host only runs the strict-recurrence math
    # (conv1d/GDN state, attention, router, norms) between submit A and B.
    def decode_step_batched(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """Per-token forward via the batched resident path (task #12).

        Mathematically identical to the base ``decode_step`` (the host math
        is copied verbatim); the GPU dispatch pattern changes: all expert
        calls of a layer share one command buffer, the down products
        accumulate resident-to-resident (per-expert gscale x router value),
        and one readback fetches the whole layer instead of ~40.
        """
        cfg = self.config
        ext = self.ext
        # oracle-11 fix 1: safety-net drain of any queue-2 upload state left
        # by an interrupted token.  An exception between the prewarm emit+
        # flush and the consumer batch_submit([cookie]) (or inside the emit
        # itself) would otherwise leave a half-recorded upload batch that a
        # LATER token's prewarm emit would APPEND into, plus unconsumed
        # in-flight staging/semaphores.  Drain ends + submits + waits every
        # half-recorded/in-flight queue-2 batch and recycles its resources;
        # on the happy path (no queue-2 state, flush consumed by its MoE
        # submit) this is two C++ boolean checks, ~0 cost.  Runs before any
        # prewarm emit of this token, so a faulted batch can never grow.
        ext.batch_upload_async_drain()
        _stop = getattr(self, "_bs_stop", None)
        # Task #13: GPU-resident GDN state syncs host->device once when
        # dirty (after reset/prefill/sequential decode), not per token.
        self._sync_gdn_gpu()
        for i, layer_type in enumerate(cfg.layer_types):
            if _stop is not None and i >= _stop and _stop >= 0:
                break
            base = f"language_model.model.layers.{i}"
            h = self._rms(
                x, self._dense(f"{base}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            # Speculative pre-pack (b', within-token): the pre-attention
            # input h(L,T) is available here, BEFORE the layer's attention
            # batch. The lookahead enqueues CPU pack cells now; the worker
            # packs during batchA/batch1 + host work + batch2, and the
            # pre-batch3 drain of this SAME layer consumes them.
            spec_on = self._spec_buf is not None
            if spec_on:
                self._spec_lookahead(i, h)
            # --- batch A: merged linear-attention block (task #13) ----
            # in-projs -> GDN conv -> GDN recur -> out_proj -> x1 = x+ao
            # all resident; single submit, single x1 readback. Full-
            # attention keeps the old batch1/host/batch2 (KV cache + SDPA
            # stay on host).
            if layer_type == "linear_attention":
                ext.batch_begin()
                ext.batch_upload(self.h_x_fused, h.contiguous().float())
                ext.batch_upload(self.h_x_resid, x.contiguous().float())
                import time as _time
                t0 = _time.perf_counter()
                self._record_gdn_in(i, ext, download=False)
                cfg_g = self.config
                ext.batch_gdn_conv_io(
                    self.h_qkv, self.conv_buf_gpu[i], self.gdn_taps[i],
                    self.h_mixed, self._gdn_C)
                ext.batch_gdn_recur_io(
                    self.h_mixed, self.h_z, self.h_a, self.h_b,
                    self.gdn_alog[i], self.gdn_dtb[i], self.gdn_normw[i],
                    self.gdn_state_gpu[i], self.h_x2,
                    self._gdn_C, self._gdn_key_dim,
                    cfg_g.linear_num_value_heads,
                    cfg_g.linear_key_head_dim,
                    cfg_g.linear_value_head_dim, self._gdn_rep,
                    self._gdn_scale, 1e-6, cfg_g.rms_norm_eps)
                out_stem = f"{base}.linear_attn.out_proj"
                opt = self.pinned[_translate(out_stem)]
                ext.batch_bf16_io(
                    self.h_x2, opt.handle, opt.handle, self.h_ao,
                    1, opt.rows, opt.cols, False,
                )
                ext.batch_vecadd_io(
                    self.h_x_resid, self.h_ao, self.h_x1,
                    cfg.hidden_size)
                x1_t = torch.empty(cfg.hidden_size)
                ext.batch_download(self.h_x1, x1_t)
                self._bump("batchA_record", _time.perf_counter() - t0)
                ext.batch_submit()
                x1 = x1_t
                # h2 is computed on GPU inside batch 3 (RMS over the
                # resident x1 this batch wrote).  The host COPY always
                # runs on the router branch too: the arena prewarm (fix
                # 3) predicts this token's routed set from it (same math
                # as the GPU topk, bit-parity on non-tie rows) and CHECK
                # mode cross-validates it.  x1 is already host-side here
                # (batch A read it back), so no extra per-layer readback.
                h2 = self._rms(
                    x1,
                    self._dense(
                        f"{base}.post_attention_layernorm.weight"),
                    cfg.rms_norm_eps,
                )
            else:
                # --- batch 1: attention input side (dense in-projs) ---
                ext.batch_begin()
                ext.batch_upload(self.h_x_fused, h.contiguous().float())
                import time as _time
                t0 = _time.perf_counter()
                # Defaults keep the type checker happy; only the taken
                # branch's tensors are actually read below.
                qg_t = k_t = v_t = torch.empty(0)
                qg_t, k_t, v_t = self._record_attn_in(i, ext)
                self._bump("batchA_record", _time.perf_counter() - t0)
                ext.batch_submit()

                # --- host: attention recurrence (needs the in-proj
                # downloads; ``pre_out`` is the PRE-out_proj vector) ---
                pre_out = self._full_attn_host(i, pos, qg_t, k_t, v_t)

                # --- batch 2: attention out_proj, residual add, h2 ----
                ext.batch_begin()
                out_stem = f"{base}.self_attn.o_proj"
                opt = self.pinned[_translate(out_stem)]
                ext.batch_upload(self.h_x2, pre_out.contiguous().float())
                ao_t = torch.empty(opt.rows)
                ext.batch_bf16_io(
                    self.h_x2, opt.handle, opt.handle, self.h_ao,
                    1, opt.rows, opt.cols, False,
                )
                ext.batch_download(self.h_ao, ao_t)
                ext.batch_submit()
                # The MoE input MUST come from the post-attention residual
                # ``x + attention`` (same as sequential decode_step).
                x1 = x + ao_t
                # Same as the GDN branch: host h2 always on the router
                # branch (arena prewarm + CHECK cross-validation); x1 is
                # host-side here, so no extra per-layer readback.
                h2 = self._rms(
                    x1,
                    self._dense(
                        f"{base}.post_attention_layernorm.weight"),
                    cfg.rms_norm_eps,
                )

            # --- batch 3: MoE input side from the TRUE h2 -------------
            # Drain this layer's lookahead slot (enqueued pre-batch above)
            # BEFORE recording, so worker packs are ready for
            # _host_pack_get's consult during the MoE.
            if spec_on:
                self._spec_drain(i)
            _async_cookie = -1
            if self._gpu_router:
                # Host shadow router for the arena prewarm (fix 3): runs
                # BEFORE batch_begin so the prewarm's map mutations are
                # visible to this batch's sync_map/topk.
                elist_p, vlist_p = self._gpu_router_host_topk(i, h2)
                self._gpu_router_prewarm(i, elist_p)
                # Task #29 (async H2D overlap): emit the prewarm CONTENT
                # as queue-2 copies OUTSIDE the open batch and flush them
                # (returns the semaphore cookie; -1 when no content is
                # pending).  The DMA overlaps this host record window as
                # well as the barrier/fill record below instead of
                # executing serially inside batch-3; batch 3 consumes the
                # cookie via batch_submit([cookie]) after recording
                # batch_cross_barrier at batch_begin (visibility).
                _async_cookie = self._gpu_router_prewarm_async_uploads(i)
            ext.batch_begin()
            if _async_cookie >= 0:
                # Queue-2 -> main-queue memory barrier (task #29): the
                # semaphore orders the copy before the batch; this makes
                # the queue-2 TRANSFER writes visible to the batch's
                # compute dispatches.  Recorded at the consumer batch's
                # start; counted separately from batch_barrier so the
                # per-token hazard-barrier counter stays comparable to
                # the pre-task baseline.
                ext.batch_cross_barrier()
            _m0 = self.stat_calls.get("cache_miss", 0) if spec_on else 0
            if self._gpu_router:
                # ---- GPU-resident RMS -> router -> topk -> MoE --------
                # (oracle-3 R1-R5 + oracle-8 fix 3).  h2 is computed on
                # GPU and stays resident (no per-layer host upload); the
                # router GEMM runs the fp32 shader on pinned fp32 weights
                # (never batch_bf16_io); the topk-256x8 kernel writes the
                # arena ids/gs, so the fused MoE record reads GPU-routed
                # expert ids + folded scales in the same batch.
                #
                # ARENA PREWARM (fix 3, before batch_begin): the host
                # shadow router on host h2 (identical math, bit-parity on
                # non-tie rows) assigns the PREDICTED routed set to arena
                # slots BEFORE the map upload, so the GPU topk resolves
                # them to real slots this same batch; their CONTENT goes
                # through the queue-2 async upload above (task #29) and
                # lands via batch_cross_barrier + batch_submit([cookie]).
                # Pins (full re-record submits) now fire only on host-vs-
                # GPU topk divergence (exact-tie flips) — the steady-
                # state pin rate is ~0 instead of a per-token re-record
                # storm.
                self._gpu_router_sync_map(i)
                if layer_type != "linear_attention":
                    # full-attn: x1 = x + ao is host-side; upload it for
                    # the RMS.  GDN layers already have x1 resident in
                    # h_x1 (batch A wrote it).
                    ext.batch_upload(
                        self.h_x1, x1.contiguous().float())
                ext.batch_rms1p_io(
                    self.h_x1, self._norm_w_res[i], self.h_x_fused,
                    cfg.hidden_size, cfg.rms_norm_eps, 1)
                ext.batch_fill(self.h_y_acc, 0.0, cfg.hidden_size)
                # Router GEMM: h2 x W_r^T -> logits[E], fp32 (R1: NO
                # BF16 cast of the router weights).
                ext.batch_fp32_io(
                    self.h_x_fused, self._router_w_res[i],
                    self._router_w_res[i], self.h_logits,
                    1, cfg.num_experts, cfg.hidden_size, False)
                # batch_download is float-typed; the topk_ids u32 come
                # back as raw float bits -> view(int32).
                ids_t = torch.empty(cfg.num_experts_per_tok)
                vals_t = torch.empty(cfg.num_experts_per_tok)
                ext.batch_topk256_io(
                    self.h_logits, self._exp_scale_res[i],
                    self._arena_map_res[i],
                    self.h_topk_ids, self.h_topk_vals,
                    self._arena_ids[i], self._arena_gs[i],
                    cfg.num_experts, cfg.num_experts_per_tok, 1)
                ext.batch_download(self.h_topk_ids, ids_t)
                ext.batch_download(self.h_topk_vals, vals_t)
                # Sentinel-contract readback (oracle-8 fix 1): fetch the
                # arena_ids the fused record reads so the host can assert
                # the raw NO_SLOT sentinel never leaked into it.
                ai_t = torch.empty(cfg.num_experts_per_tok)
                ext.batch_download(self._arena_ids[i], ai_t)
                # In-batch arena CONTENT warm-up (fix 3) is GONE on the
                # router path: prewarm content now goes through queue-2
                # batch_upload_async (task #29) BEFORE batch_begin, and
                # batch_cross_barrier + batch_submit([cookie]) guarantee
                # the fused dispatches below see it.  The sentinel
                # readback + sentinel-contract assert still run here.
                if self._gpu_router_check:
                    h2g_t = torch.empty(cfg.hidden_size)
                    ext.batch_download(self.h_x_fused, h2g_t)
                    lg_t = torch.empty(cfg.num_experts)
                    ext.batch_download(self.h_logits, lg_t)
                    self._last_h2g = h2g_t
                    self._last_lg = lg_t
                # MoE record in the SAME batch: dispatches read the
                # topk-written ids/gs + the prewarmed content.
                acc_t, sd_t, gsc_t = self._record_moe_gpu(
                    i, ext, None, None, upload_ids_gs=False)
                ext.batch_submit(
                    [_async_cookie] if _async_cookie >= 0 else [])
                elist = ids_t.view(torch.int32).tolist()
                vlist = vals_t.tolist()
                # Assertion (oracle-8 fix 1): the arena_ids readback must
                # never contain the raw NO_SLOT sentinel — the kernel
                # maps it to the reserved no-content slot 0, so any value
                # outside [0, arena_n) (including 0x7FFFFFFF or the old
                # 0xFFFFFFFF/-1) is a sentinel-contract break and an
                # out-of-bounds arena index for the fused dispatches.
                ai = ai_t.view(torch.int32).tolist()
                # I2 (oracle-10): RuntimeError, not assert — -O cannot
                # disarm the sentinel/range contract.  The arena_ids
                # readback must never contain the raw NO_SLOT sentinel:
                # the kernel maps it to the reserved no-content slot 0,
                # so any value outside [0, arena_n) (including
                # 0x7FFFFFFF or the old 0xFFFFFFFF/-1) is a
                # sentinel-contract break and an out-of-bounds arena
                # index for the fused dispatches.
                if not all(0 <= _a < self._arena_n for _a in ai):
                    raise RuntimeError(
                        f"layer {i}: arena_ids readback {ai} contains an "
                        f"out-of-range slot; raw NO_SLOT sentinel "
                        f"{hex(_ARENA_NO_SLOT)} must never appear (kernel "
                        f"maps it to reserved slot 0)")
                if self._gpu_router_check:
                    self._gpu_router_crosscheck(
                        i, h2, h2g_t, lg_t, elist, vlist,
                        elist_p, vlist_p)
                # Pin/tie fallback: re-record with host-corrected ids/gs
                # + arena content in a fresh batch (rare: LRU-evicted
                # routed experts, or router-tie divergence).
                missing = [e for e in elist
                           if self._arena_map[i].get(e) is None]
                if missing or self._gpu_router_fallback:
                    if missing:
                        self.stat_calls["gpu_router_pin"] = (
                            self.stat_calls.get("gpu_router_pin", 0) + 1)
                    if self._gpu_router_fallback:
                        # hard-fall-to-host on ties (oracle-3 R5)
                        elist, vlist = self._gpu_router_host_list
                        self._gpu_router_fallback = False
                    ext.batch_begin()
                    ext.batch_fill(self.h_y_acc, 0.0, cfg.hidden_size)
                    acc_t, sd_t, gsc_t = self._record_moe_gpu(
                        i, ext, elist, vlist, upload_ids_gs=True)
                    ext.batch_submit()
                    self.stat_calls["gpu_router_rerecord"] = (
                        self.stat_calls.get("gpu_router_rerecord", 0) + 1)
            else:
                # ---- original host h2/router/topk path (unchanged) ----
                ext.batch_upload(
                    self.h_x_fused, h2.contiguous().float())
                ext.batch_fill(self.h_y_acc, 0.0, cfg.hidden_size)
                acc_t, sd_t, gsc_t = self._record_moe_in(i, h2, ext)
                ext.batch_submit()
            if spec_on:
                # Half-B skip rule: a layer only speculates for the NEXT
                # token when THIS token actually missed (pack work
                # happened -> likely again). Re-armed per layer, per token.
                self._spec_armed[i] = (
                    self.stat_calls.get("cache_miss", 0) > _m0)
            gs = torch.sigmoid(gsc_t)
            # Shared down folds silu(gate)*up inside batch 3 (swiglu on
            # GPU); sd_t is the SHARED down output, not the routed acc.
            x = x1 + acc_t + gs * sd_t
            if os.environ.get("FTV_DEBUG_X"):
                import pathlib as _p
                try:
                    with open(os.environ["FTV_DEBUG_X"], "a") as _f:
                        _f.write(
                            f"bat {i} {layer_type} {x.abs().sum().item():.9e} "
                            f"fin={torch.isfinite(x).all().item()}\n"
                        )
                except OSError:
                    pass
        x = self._rms(
            x, self._dense("language_model.model.norm.weight"),
            cfg.rms_norm_eps,
        )
        lm = self.pinned[_translate("language_model.lm_head")]
        ext.batch_begin()
        ext.batch_upload(self.h_x_fused, x.contiguous().float())
        logits_t = torch.empty(lm.rows)
        ext.batch_bf16_io(
            self.h_x_fused, lm.handle, lm.handle, self.h_y,
            1, lm.rows, lm.cols, False,
        )
        ext.batch_download(self.h_y, logits_t)
        ext.batch_submit()
        return logits_t

    def _record_gdn_in(
        self, i: int, ext: Any, download: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Record the 4 dense linear-attention in-projections + downloads
        into the open batch; returns (qkv, z, a, b) host tensors.
        With download=False (task #13 GPU-resident GDN) the outputs stay
        resident in h_qkv/h_z/h_a/h_b for the GDN kernels and empty
        tensors are returned."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        qkv_s = self.pinned[_translate(f"{base}.linear_attn.in_proj_qkv")]
        z_s = self.pinned[_translate(f"{base}.linear_attn.in_proj_z")]
        a_s = self.pinned[_translate(f"{base}.linear_attn.in_proj_a")]
        b_s = self.pinned[_translate(f"{base}.linear_attn.in_proj_b")]
        qkv_t = torch.empty(qkv_s.rows)
        z_t = torch.empty(z_s.rows)
        a_t = torch.empty(a_s.rows)
        b_t = torch.empty(b_s.rows)
        ext.batch_bf16_io(self.h_x_fused, qkv_s.handle, qkv_s.handle,
                          self.h_qkv, 1, qkv_s.rows, qkv_s.cols, False)
        ext.batch_bf16_io(self.h_x_fused, z_s.handle, z_s.handle,
                          self.h_z, 1, z_s.rows, z_s.cols, False)
        ext.batch_bf16_io(self.h_x_fused, a_s.handle, a_s.handle,
                          self.h_a, 1, a_s.rows, a_s.cols, False)
        ext.batch_bf16_io(self.h_x_fused, b_s.handle, b_s.handle,
                          self.h_b, 1, b_s.rows, b_s.cols, False)
        if not download:
            empty = torch.empty(0)
            return empty, empty, empty, empty
        ext.batch_download(self.h_qkv, qkv_t)
        ext.batch_download(self.h_z, z_t)
        ext.batch_download(self.h_a, a_t)
        ext.batch_download(self.h_b, b_t)
        return qkv_t, z_t, a_t, b_t

    def _record_attn_in(
        self, i: int, ext: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Record q_proj/k_proj/v_proj + downloads (full attention)."""
        base = f"language_model.model.layers.{i}"
        q_s = self.pinned[_translate(f"{base}.self_attn.q_proj")]
        k_s = self.pinned[_translate(f"{base}.self_attn.k_proj")]
        v_s = self.pinned[_translate(f"{base}.self_attn.v_proj")]
        qg_t = torch.empty(q_s.rows)
        k_t = torch.empty(k_s.rows)
        v_t = torch.empty(v_s.rows)
        ext.batch_bf16_io(self.h_x_fused, q_s.handle, q_s.handle,
                          self.h_qg, 1, q_s.rows, q_s.cols, False)
        ext.batch_bf16_io(self.h_x_fused, k_s.handle, k_s.handle,
                          self.h_k, 1, k_s.rows, k_s.cols, False)
        ext.batch_bf16_io(self.h_x_fused, v_s.handle, v_s.handle,
                          self.h_v, 1, v_s.rows, v_s.cols, False)
        ext.batch_download(self.h_qg, qg_t)
        ext.batch_download(self.h_k, k_t)
        ext.batch_download(self.h_v, v_t)
        return qg_t, k_t, v_t

    def _record_moe_in(
        self, i: int, h: torch.Tensor, ext: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Record the routed-expert + shared-expert input side into the
        open batch and return (acc, sd, gsc) host tensors.

        The 8 expert down products accumulate resident-to-resident with the
        router value folded into the per-expert gscale:
        acc += silu(gate_e)*up_e @ Wd_e^T * (down_gscale_e * value_e).
        The shared-expert path folds silu(gate)*up on the GPU (swiglu)
        into h_x3 and runs the shared down in the SAME batch, so the host
        only reads back acc, the shared down output and the gating
        scalar (task #12 opt: kills the former batch-4 submit+upload)."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        gate_mat = self.expert_mats[f"{base}.mlp.switch_mlp.gate_proj"]
        up_mat = self.expert_mats[f"{base}.mlp.switch_mlp.up_proj"]
        down_mat = self.expert_mats[f"{base}.mlp.switch_mlp.down_proj"]
        router_w = self._dense(f"{base}.mlp.gate.weight")
        logits = F.linear(h.float(), router_w.float()).float()
        probs = torch.softmax(logits, dim=-1)
        values, index = torch.topk(
            probs, cfg.num_experts_per_tok, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)
        vlist = values.tolist() if values.dim() == 1 else values.tolist()[0]
        elist = index.tolist() if index.dim() == 1 else index.tolist()[0]
        # Expert-slot canonicalization (task #19): bind the top-8 experts to
        # canonical positions by sorting (expert, value) pairs by expert id
        # ascending. The desc-set cache key is (weight-buffers[e] x
        # silu-slot[e]) pairings; without the sort, each token's routing
        # order permutes which expert lands in which silu slot, so the same
        # expert flips across all 8 slot keys token-to-token and every
        # gateup/acc desc churns (~350-400 of the 435 misses/tok are this
        # permutation, not real new weights). After the sort a given expert
        # always occupies a fixed slot, so identical expert sets hit the
        # cache. Accumulation into h_y_acc is order-independent fp32 add (a
        # reorder-only ~1e-7 shift); the router/topk ran BEFORE this sort,
        # so softmax choice is untouched, and dn_sc follows vlist[e], which
        # is sorted in the same permutation as elist (keeps the pair).
        # SET-RELATIVE CAVEAT (verdict): expert E's canonical slot here is
        # its rank within the CURRENT top-8 set, NOT a global expert->slot
        # table. Repeated identical sets therefore stabilize (the measured
        # 435 -> 275/tok, -37%), but any token whose SET changes re-ranks
        # all 8 in the new set and re-binds them -- so -37% is a LOWER
        # bound; a true global expert->slot mapping would extract more
        # (that needs next-token expert prefetch, the next task; logged,
        # not implemented now).
        _pairs = sorted(zip(elist, vlist))
        elist = [_e for _e, _v in _pairs]
        vlist = [_v for _e, _v in _pairs]
        # Per-expert statics (shapes + gscales) never change: cache the
        # float() conversions + shape reads once per bundle (task #12
        # opt: kills 16 try/except + shape reads per layer per token).
        statics = self._moe_statics
        gu_w: list[int] = []
        gu_s: list[int] = []
        gu_sc: list[float] = []
        dn_w: list[int] = []
        dn_s: list[int] = []
        dn_sc: list[float] = []
        gu_rows = gu_cols = dn_rows = dn_cols = 0
        gu_slot0: list[int] = []
        dn_slot0: list[int] = []
        for e, expert in enumerate(elist):
            skey = (id(gate_mat), id(up_mat), id(down_mat), expert)
            st = statics.get(skey)
            if st is None:
                g_packed0, _g_s0, g_gs0 = gate_mat[expert]
                d_packed0, _d_s0, d_gs0 = down_mat[expert]
                try:
                    fgs = float(g_gs0)
                    dgs = float(d_gs0)
                except (TypeError, ValueError, RuntimeError):
                    raise RuntimeError(
                        f"bad MoE gscale for expert {expert}") from None
                st = (g_packed0.shape[-2], g_packed0.shape[-1] * 2,
                      d_packed0.shape[-2], d_packed0.shape[-1] * 2,
                      fgs, dgs)
                statics[skey] = st
            (r_gu, c_gu, r_dn, c_dn, fgs, dgs) = st
            gu_rows, gu_cols, dn_rows, dn_cols = r_gu, c_gu, r_dn, c_dn
            h_w, h_s, _w, _wb, _sb = self._fused_gateup_handles(
                gate_mat, up_mat, expert, batch=True)
            gu_w.append(h_w)
            gu_s.append(h_s)
            gu_sc.append(fgs)
            gu_slot0.append(self._fused_slot0(gate_mat, expert))
            h_wd, h_sd = self._down_handles_batch(
                down_mat, expert, batch=True)
            dn_w.append(h_wd)
            dn_s.append(h_sd)
            dn_sc.append(dgs * vlist[e])
            dn_slot0.append(self._down_slot0(down_mat, expert))
        sg = self.pinned[_translate(f"{base}.mlp.shared_expert.gate_proj")]
        su = self.pinned[_translate(f"{base}.mlp.shared_expert.up_proj")]
        gsc = self.pinned[_translate(f"{base}.mlp.shared_expert_gate")]
        sdp = self.pinned[_translate(
            f"{base}.mlp.shared_expert.down_proj")]
        acc_t = torch.empty(cfg.hidden_size)
        sd_t = torch.empty(cfg.hidden_size)
        gsc_t = torch.empty(gsc.rows)
        gate_rows = self.config.hidden_size // 4
        if self._arena_enabled and self._arena_ids:
            # Fused MUL_MAT_ID-style MoE layer (arena path): the 16
            # per-expert dispatches collapse to 2 — one fused 8-expert
            # gateup+silu (writes silu_all), one fused 8-expert acc
            # (reads silu_all, accumulates h_y_acc in canonical order).
            # Both index the per-layer arena via the resident u32[8]
            # ids buffer (slot per canonical expert, same assignment
            # the uploads above used) and the float[16] gs buffer.
            #
            # PIN: a mid-record global-cache eviction can pop a ROUTED
            # expert's arena-map entry, and a later routed expert then
            # reuses its slot (content clobbered).  The legacy
            # 16-dispatch path tolerates this — each dispatch reads its
            # slot at its own recorded position — but the fused
            # dispatches read ALL 8 slots at once, so every routed
            # expert must own a live slot with content-correct data at
            # record time.  Re-assign missing routed experts (LRU-evict
            # non-routed entries) and re-upload their fused+down content
            # from the host pack cache (HIT-happy; idempotent builds).
            # Loop to fixpoint: a re-assignment evicts at most one
            # entry, which may itself be routed; each pass strictly
            # increases routed membership, so ≤8 passes always settle.
            _mp = self._arena_map[i]
            for _pin in range(9):
                _missing = [e for e in elist if _mp.get(e) is None]
                if not _missing:
                    break
                for e in _missing:
                    sidx = self._arena_assign(i, e)
                    _fk = (id(gate_mat), id(up_mat), e)
                    _dk = (id(down_mat), e)
                    _gp, _gu, _ = gate_mat[e]
                    _up, _uu, _ = up_mat[e]
                    _fp, _fs, _fh = self._host_pack_get(
                        _fk, lambda: self._build_fused_pack(
                            _gp, _up, _gu, _uu))
                    self.ext.batch_upload(
                        self._arena_wf[i], _fp,
                        sidx * self._arena_slot_wf_b)
                    self.ext.batch_upload(
                        self._arena_sf[i], _fs.contiguous(),
                        sidx * self._arena_slot_sf_b)
                    _dp, _ds, _ = down_mat[e]
                    _dq, _ds32, _dh = self._host_pack_get(
                        _dk, lambda: self._build_down_pack(_dp, _ds))
                    self.ext.batch_upload(
                        self._arena_wd[i], _dq.contiguous(),
                        sidx * self._arena_slot_wd_b)
                    self.ext.batch_upload(
                        self._arena_sd[i], _ds32.contiguous(),
                        sidx * self._arena_slot_sd_b)
                    self._bump("exp_up", 0.0)
            # No silent slot-0 fallback: the fused dispatches read ALL
            # routed experts' slots at once, so a routed expert missing
            # from the arena map at record time would decode against a
            # stale/foreign slot (pre-pin code used `_mp.get(e, 0)` and
            # silently fell back to slot 0 — a wrong-weights decode with
            # no error).  The pin loop above guarantees membership;
            # verify loudly instead of defaulting.
            _routed_missing = [e for e in elist if _mp.get(e) is None]
            if _routed_missing:
                raise RuntimeError(
                    f"layer {i}: routed expert(s) {_routed_missing} have no "
                    f"arena slot after the pin loop (arena_n="
                    f"{self._arena_n}, routed={len(elist)}); refusing silent "
                    f"slot-0 fallback")
            if self._arena_n - 1 < len(elist):
                raise RuntimeError(
                    f"layer {i}: arena has {self._arena_n - 1} usable slots "
                    f"(slot 0 reserved) but "
                    f"{len(elist)} routed experts need one each")
            _ids = torch.tensor(
                [_mp[e] for e in elist],
                dtype=torch.int32)
            # gs = [fgs_0..7, dgs_0*v_0 .. dgs_7*v_7]: the SAME fp64
            # values the legacy path folds into the per-expert scale
            # push constants, converted to fp32 with the same RTNE rule
            # as the C++ (float)double cast, so the fused dot's s =
            # e4m3*gs is bit-identical per row.
            _gs = torch.tensor(gu_sc + dn_sc, dtype=torch.float32)
            ext.batch_upload(self._arena_ids[i], _ids, 0)
            ext.batch_upload(self._arena_gs[i], _gs, 0)
            ext.batch_moe_fused_record(
                self.h_x_fused, self.h_y_acc,
                self.h_sg, self.h_su, self.h_gsc,
                self.h_x3, self.h_sdown,
                self.h_silu_all,
                self._arena_wf[i], self._arena_sf[i],
                self._arena_wd[i], self._arena_sd[i],
                self._arena_ids[i], self._arena_gs[i],
                gu_rows, gu_cols, self._arena_rows_fused,
                dn_rows, dn_cols, self._arena_rows_down,
                gu_rows,  # silu_N: X base stride per expert in silu_all
                sg.handle, sg.rows, sg.cols,
                su.handle, su.rows, su.cols,
                gsc.handle, gsc.rows, gsc.cols,
                sdp.handle, sdp.rows, sdp.cols,
                gate_rows,
                acc_t, sd_t, gsc_t,
            )
        else:
            # Legacy 16-dispatch path (arena disabled / unknown layer):
            # 8 gateup + 8 acc + shared gate/up/gate + swiglu fold +
            # shared down + 3 downloads, one crossing.
            #
            # MID-RECORD SLOT-REUSE HAZARD (why the fused branch pins):
            # a mid-record _cache_put eviction can pop a ROUTED expert's
            # arena-map entry, and a later routed expert's _arena_assign
            # then reuses that slot (content clobbered).  This path
            # "tolerates" it — each of the 16 dispatches already captured
            # its own slot0 push constant at its own record position —
            # but the fused path reads ALL 8 slots at once from the
            # resident ids buffer, so it must pin every routed expert
            # (re-assign + re-upload) before recording.  See the fused
            # branch above.
            ext.batch_moe_record(
                self.h_x_fused, self.h_y_acc,
                self.h_sg, self.h_su, self.h_gsc,
                self.h_x3, self.h_sdown,
                self.h_silu_slots,
                gu_w, gu_s, gu_sc, gu_rows, gu_cols,
                dn_w, dn_s, dn_sc, dn_rows, dn_cols,
                sg.handle, sg.rows, sg.cols,
                su.handle, su.rows, su.cols,
                gsc.handle, gsc.rows, gsc.cols,
                sdp.handle, sdp.rows, sdp.cols,
                gate_rows,
                gu_slot0, dn_slot0,
                acc_t, sd_t, gsc_t,
            )
        return acc_t, sd_t, gsc_t

    def _gdn_host(
        self, i: int,
        qkv: torch.Tensor, z: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    ) -> torch.Tensor:
        """Host recurrence of one linear-attention layer; returns the
        gated vector that feeds the (batched) out_proj."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        conv_w = self._dense(f"{base}.linear_attn.conv1d.weight")
        if conv_w.dim() == 3 and conv_w.shape[1] == 1:
            taps = conv_w[:, 0, :].float()
        else:
            taps = conv_w.float().reshape(conv_w.shape[0], -1)
        buf = self.conv_buf[i]
        mixed = (taps[:, :3] * buf).sum(-1) + taps[:, 3] * qkv
        buf[:, 0] = buf[:, 1]
        buf[:, 1] = buf[:, 2]
        buf[:, 2] = qkv
        mixed = F.silu(mixed)
        nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        key_dim = nk * hk
        query, key, value = torch.split(
            mixed, [key_dim, key_dim, nv * hv], dim=-1)
        query = query.reshape(nk, hk).repeat_interleave(nv // nk, dim=0)
        key = key.reshape(nk, hk).repeat_interleave(nv // nk, dim=0)
        value = value.reshape(nv, hv)
        a_log = self._dense(f"{base}.linear_attn.A_log")
        dt_bias = self._dense(f"{base}.linear_attn.dt_bias")
        beta = torch.sigmoid(b)
        g = -torch.exp(a_log) * F.softplus(a + dt_bias)
        scale = 1.0 / math.sqrt(hk)
        qt = l2norm(query) * scale
        kt = l2norm(key)
        state = self.gdn_state[i]
        state.mul_(torch.exp(g)[:, None, None])
        kv_mem = (state * kt[:, :, None]).sum(dim=1)
        delta = (value - kv_mem) * beta[:, None]
        state.add_(kt[:, :, None] * delta[:, None, :])
        core = (state * qt[:, :, None]).sum(dim=1)
        norm_w = self._dense(f"{base}.linear_attn.norm.weight")
        var = core.pow(2).mean(-1, keepdim=True)
        normed = core * torch.rsqrt(var + cfg.rms_norm_eps)
        return (normed * norm_w * F.silu(z.reshape(nv, hv))).reshape(nv * hv)

    def _full_attn_host(
        self, i: int, pos: int,
        qg: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    ) -> torch.Tensor:
        """Host attention of one full-attention layer (KV append + SDPA);
        returns the gated context vector feeding batched o_proj."""
        cfg = self.config
        base = f"language_model.model.layers.{i}"
        hq, hkv, hd = (
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
        )
        query, gate = qg.view(hq, 2 * hd).chunk(2, dim=-1)
        key = key.view(hkv, hd)
        value = value.view(hkv, hd)
        q_norm = self._dense(f"{base}.self_attn.q_norm.weight")
        k_norm = self._dense(f"{base}.self_attn.k_norm.weight")
        query = rms_norm_head(query, q_norm, cfg.rms_norm_eps).float()
        key = rms_norm_head(key, k_norm, cfg.rms_norm_eps).float()
        cos = self.rope_cos[pos].unsqueeze(0)
        sin = self.rope_sin[pos].unsqueeze(0)
        query = apply_partial_rotary(query, cos, sin, self.rotary_dim)
        key = apply_partial_rotary(key, cos, sin, self.rotary_dim)
        if self.k_cache[i].dtype == torch.bfloat16:
            self.k_cache[i][pos].copy_(key.to(torch.bfloat16))
            self.v_cache[i][pos].copy_(value.to(torch.bfloat16))
            keys = self.k_cache[i][: pos + 1].float()
            values = self.v_cache[i][: pos + 1].float()
        else:
            self.k_cache[i][pos].copy_(key)
            self.v_cache[i][pos].copy_(value)
            keys = self.k_cache[i][: pos + 1]
            values = self.v_cache[i][: pos + 1]
        repeat = hq // hkv
        if repeat > 1:
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)
        q = query.unsqueeze(1)
        k = keys.transpose(0, 1)
        v = values.transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)).squeeze(1) / math.sqrt(hd)
        probs = torch.softmax(scores.float(), dim=-1).unsqueeze(1)
        context = (probs @ v).transpose(0, 1).reshape(hq * hd)
        context = context * torch.sigmoid(gate.reshape(hq * hd))
        return context

