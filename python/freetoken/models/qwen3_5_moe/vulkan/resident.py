"""Hybrid GPU/CPU streaming decode for Qwen3.5-MoE (Cyber-Tiel 35B).

Strategy (fits 8GB VRAM against a 20GB checkpoint):
- GPU (proven ``linear_ao_affine_resident_io`` kernel): every large
  quantized matvec — attention projections, shared experts, per-token
  active experts, lm_head.
- CPU (torch, fp32): router softmax/top-k, GatedDeltaNet recurrence and
  conv state, RMSNorms, silu gating, full-attention SDPA with RAM KV
  cache, embed row dequant, greedy sampling.
- VRAM: dense-path weights + lm_head pinned (~1GB); the 8 active
  experts per layer are uploaded, used, and freed per token.

Single-token ``decode_step`` also serves as prefill: looping it over the
prompt builds GDN state and the KV caches exactly as generation does.
"""
from __future__ import annotations

import math
import os
import pathlib

import torch
import torch.nn.functional as F

from .attention import (
    apply_partial_rotary,
    compute_text_rope,
    rms_norm_head,
)
from .config import Qwen35MoeConfig, load_qwen35_config
from .linear_attn import l2norm
from .loader import (
    AffineQuantMatrix,
    ShardedSafetensors,
    _load_quant_matrix,
    dequantize_affine,
)
from ._runtime import get_ext

PREFIX = "language_model"


class _Pinned:
    """One quantized matrix pinned in VRAM (packed + BF16 scales/biases)."""

    __slots__ = ("matrix", "h_packed", "h_scales", "h_biases", "rows", "cols")

    def __init__(self, ext, matrix: AffineQuantMatrix):
        self.matrix = matrix
        self.h_packed = ext.upload_resident(matrix.packed.contiguous())
        self.h_scales = ext.upload_resident(matrix.scales.contiguous())
        self.h_biases = ext.upload_resident(matrix.biases.contiguous())
        self.rows = matrix.rows
        self.cols = matrix.cols


class ResidentQwen35:
    """Streaming decoder; see module docstring for the placement strategy."""

    def __init__(
        self,
        model_dir: str | pathlib.Path,
        ext=None,
        max_len: int = 4096,
        # NOTE: fp32 is the default because bf16 KV is numerically unsafe
        # on this architecture: measured 0.0014 (pos 0) growing to 0.0066
        # (pos 3) abs error per full-attn layer vs 3e-06 fp32, compounding
        # over 10 full layers into word salad. Keep bf16 only as a
        # capacity option for very long contexts, with eyes open.
        kv_dtype: str = "fp32",
    ) -> None:
        torch.set_num_threads(8)
        # ext=None resolves the pybind module lazily via _runtime, so the
        # class stays importable when Vulkan is absent; passing ext keeps
        # the fork's explicit-injection contract. Normalize the LOCAL first:
        # __init__ keeps calling ext.* (allocate_resident below), so a None
        # local would crash the implicit-resolution path the tests skip.
        if ext is None:
            ext = get_ext()
        self.ext = ext
        if kv_dtype not in ("fp32", "bf16"):
            raise ValueError(f"kv_dtype must be fp32|bf16, got {kv_dtype!r}")
        self.kv_dtype = kv_dtype
        try:
            self.config: Qwen35MoeConfig = load_qwen35_config(model_dir)
            cfg = self.config
            self.tensors = ShardedSafetensors(model_dir)
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"cannot open Qwen35 model at {model_dir}"
            ) from exc
        self.max_len = max_len

        # Scratch fp32 transfer buffers. x input width peaks at the
        # attention output widths (GDN out / full-attn context =
        # 4096), NOT hidden (2048); y output width peaks at vocab.
        max_in = max(
            cfg.hidden_size,
            cfg.linear_num_value_heads * cfg.linear_value_head_dim,
            cfg.num_attention_heads * cfg.head_dim,
        )
        self.h_x = ext.allocate_resident(max_in * 4)
        self.h_y = ext.allocate_resident(cfg.vocab_size * 4)

        self.pinned: dict[str, _Pinned] = {}
        self.expert_mats: dict[str, AffineQuantMatrix] = {}
        self._pin_all()

        # Embed matrix stays on CPU (mmap); rows are dequantized on demand.
        stem = f"{PREFIX}.model.embed_tokens"
        bits, group = cfg.quant_for(stem)
        self.embed = _load_quant_matrix(self.tensors, stem, bits, group)

        # GatedDeltaNet per-layer state (RAM, fp32).
        key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        conv_dim = 2 * key_dim + value_dim
        self.gdn_state: dict[int, torch.Tensor] = {}
        self.conv_buf: dict[int, torch.Tensor] = {}
        for i, t in enumerate(cfg.layer_types):
            if t == "linear_attention":
                self.gdn_state[i] = torch.zeros(
                    cfg.linear_num_value_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_value_head_dim,
                )
                self.conv_buf[i] = torch.zeros(conv_dim, 3)

        # Full-attention KV caches: preallocated RAM [max_len, Kv, D] per
        # full layer (120k-ready: no per-step torch.stack; slice [:pos+1]
        # instead). 10 layers * 120k * 2(KV) * 2(heads) * 256 elems =
        # 1.229G = ~4.58GiB fp32 / ~2.29GiB bf16. fp32 is the default:
        # bf16 KV injects ~3% relative error per full-attn layer here
        # (measured, growing with position) and 120k runs should prefer
        # q8_0/q4_0 next, not raw bf16 truncation. Model max_pos is
        # 262144: refuse beyond.
        if max_len > 262144:
            raise ValueError(
                f"max_len {max_len} exceeds model max_pos 262144"
            )
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

        # Text RoPE table (M-RoPE rows are identical for text positions).
        pos = torch.arange(max_len, dtype=torch.float32)
        cos, sin = compute_text_rope(
            pos, cfg.rope_theta, cfg.head_dim, cfg.partial_rotary_factor
        )
        self.rope_cos = cos  # [max_len, rotary_dim]
        self.rope_sin = sin
        # Already validated inside compute_text_rope; read back, don't recompute.
        self.rotary_dim = self.rope_cos.shape[-1]
        self.reset()

    # -- weight staging -------------------------------------------------
    def _pin_quant(self, stem: str, extra=()) -> _Pinned:
        bits, group = self.config.quant_for(stem)
        matrix = _load_quant_matrix(self.tensors, stem, bits, group, extra)
        slot = _Pinned(self.ext, matrix)
        self.pinned[stem] = slot
        return slot

    def _pin_all(self) -> None:
        cfg = self.config
        experts = (cfg.num_experts,)
        for i, layer_type in enumerate(cfg.layer_types):
            base = f"{PREFIX}.model.layers.{i}"
            if layer_type == "linear_attention":
                for suffix in (
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.in_proj_a",
                    "linear_attn.in_proj_b",
                    "linear_attn.out_proj",
                ):
                    self._pin_quant(f"{base}.{suffix}")
            else:
                for suffix in (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ):
                    self._pin_quant(f"{base}.{suffix}")
            for suffix in (
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj",
                # NOTE: shared_expert_gate is quantized ([1, H] int8),
                # not dense like the router — it needs the affine path.
                "mlp.shared_expert_gate",
            ):
                self._pin_quant(f"{base}.{suffix}")
            # Experts stay on CPU; only the active 8/layer are uploaded.
            for suffix in (
                "mlp.switch_mlp.gate_proj",
                "mlp.switch_mlp.up_proj",
                "mlp.switch_mlp.down_proj",
            ):
                stem = f"{base}.{suffix}"
                bits, group = cfg.quant_for(stem)
                matrix = _load_quant_matrix(
                    self.tensors, stem, bits, group, experts
                )
                self.expert_mats[stem] = matrix
        self._pin_quant(f"{PREFIX}.lm_head")

    def reset(self) -> None:
        """Clear recurrent state and KV caches for a new sequence."""
        for tensor in self.gdn_state.values():
            tensor.zero_()
        for tensor in self.conv_buf.values():
            tensor.zero_()
        for tensor in self.k_cache.values():
            tensor.zero_()
        for tensor in self.v_cache.values():
            tensor.zero_()
        self._seq_len = 0

    # -- GPU matvec ------------------------------------------------------
    def _affine(
        self,
        x: torch.Tensor,
        packed_h: int,
        scales_h: int,
        biases_h: int,
        n_out: int,
        k_in: int,
        bits: int,
        group: int,
    ) -> torch.Tensor:
        ext = self.ext
        ext.update_resident(self.h_x, x.contiguous().float())
        ext.linear_ao_affine_resident_io(
            self.h_x, packed_h, scales_h, biases_h, self.h_y,
            1, n_out, k_in, bits, group,
        )
        return ext.download_resident(self.h_y, [n_out])

    def _pinned_affine(self, x: torch.Tensor, stem: str) -> torch.Tensor:
        slot = self.pinned[stem]
        matrix = slot.matrix
        return self._affine(
            x, slot.h_packed, slot.h_scales, slot.h_biases,
            slot.rows, slot.cols, matrix.bits, matrix.group_size,
        )

    def _affine_batch(
        self,
        x: torch.Tensor,
        packed_h: int,
        scales_h: int,
        biases_h: int,
        n_out: int,
        k_in: int,
        bits: int,
        group: int,
    ) -> torch.Tensor:
        """Batched (Tc>1) affine matvec for chunked prefill.

        Uses temp resident buffers sized Tc*N (freed on return) instead of
        the single-token h_x/h_y scratch, so chunk size is bounded only by
        VRAM. The ao_affine kernel dispatches (N/128)xT — proven T>1-safe
        by its push-constant/grid layout (see ext_module.cpp).
        """
        tc = x.shape[0]
        ext = self.ext
        h_x = ext.allocate_resident(tc * k_in * 4)
        h_y = ext.allocate_resident(tc * n_out * 4)
        try:
            ext.update_resident(h_x, x.contiguous().float())
            ext.linear_ao_affine_resident_io(
                h_x, packed_h, scales_h, biases_h, h_y,
                tc, n_out, k_in, bits, group,
            )
            return ext.download_resident(h_y, [tc, n_out])
        finally:
            ext.free_resident(h_x)
            ext.free_resident(h_y)

    def _pinned_affine_batch(self, x: torch.Tensor, stem: str) -> torch.Tensor:
        slot = self.pinned[stem]
        matrix = slot.matrix
        return self._affine_batch(
            x, slot.h_packed, slot.h_scales, slot.h_biases,
            slot.rows, slot.cols, matrix.bits, matrix.group_size,
        )

    # -- CPU glue ---------------------------------------------------------
    @staticmethod
    def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        # Qwen3.5 norms are zero-centered: y = norm(x) * (1 + w).
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * (1.0 + w)

    def _dense(self, name: str) -> torch.Tensor:
        tensor = self.tensors.get(name)
        if tensor.dtype == torch.bfloat16:
            return tensor.float()
        return tensor

    def embed_row(self, token_id: int) -> torch.Tensor:
        """Dequantize one embed row on CPU (µs; no VRAM needed)."""
        matrix = self.embed
        packed = matrix.packed[token_id : token_id + 1, :].contiguous()
        scales = matrix.scales[token_id : token_id + 1, :].contiguous()
        biases = matrix.biases[token_id : token_id + 1, :].contiguous()
        return dequantize_affine(
            packed, scales, biases, matrix.bits, matrix.group_size,
        ).reshape(-1)

    # -- block steps -------------------------------------------------------
    def _gdn_step(self, i: int, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        base = f"{PREFIX}.model.layers.{i}"
        qkv = self._pinned_affine(x, f"{base}.linear_attn.in_proj_qkv")
        z = self._pinned_affine(x, f"{base}.linear_attn.in_proj_z")
        a = self._pinned_affine(x, f"{base}.linear_attn.in_proj_a")
        b = self._pinned_affine(x, f"{base}.linear_attn.in_proj_b")

        conv_w = self._dense(f"{base}.linear_attn.conv1d.weight")
        if conv_w.dim() == 3 and conv_w.shape[1] == 1:
            taps = conv_w[:, 0, :].float()  # [C, K] (loader permuted [C,K,1])
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
            mixed, [key_dim, key_dim, nv * hv], dim=-1
        )
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
        core = (state * qt[:, :, None]).sum(dim=1)  # [V, D]

        norm_w = self._dense(f"{base}.linear_attn.norm.weight")
        var = core.pow(2).mean(-1, keepdim=True)
        normed = core * torch.rsqrt(var + cfg.rms_norm_eps)
        # RMSNormGated uses a plain ones-init weight (NOT zero-centered).
        gated = (
            normed * norm_w * F.silu(z.reshape(nv, hv))
        ).reshape(nv * hv)
        return self._pinned_affine(gated, f"{base}.linear_attn.out_proj")

    def _full_attn_step(self, i: int, x: torch.Tensor, pos: int) -> torch.Tensor:
        cfg = self.config
        base = f"{PREFIX}.model.layers.{i}"
        hq, hkv, hd = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        qg = self._pinned_affine(x, f"{base}.self_attn.q_proj")
        query, gate = qg.view(hq, 2 * hd).chunk(2, dim=-1)
        key = self._pinned_affine(x, f"{base}.self_attn.k_proj").view(hkv, hd)
        value = self._pinned_affine(x, f"{base}.self_attn.v_proj").view(hkv, hd)

        q_norm = self._dense(f"{base}.self_attn.q_norm.weight")
        k_norm = self._dense(f"{base}.self_attn.k_norm.weight")
        query = rms_norm_head(query, q_norm, cfg.rms_norm_eps).float()
        key = rms_norm_head(key, k_norm, cfg.rms_norm_eps).float()

        cos = self.rope_cos[pos].unsqueeze(0)
        sin = self.rope_sin[pos].unsqueeze(0)
        query = apply_partial_rotary(query, cos, sin, self.rotary_dim)
        key = apply_partial_rotary(key, cos, sin, self.rotary_dim)

        # KV cache may be bf16 (120k default) — store cast, compute fp32.
        if self.k_cache[i].dtype == torch.bfloat16:
            self.k_cache[i][pos].copy_(key.to(torch.bfloat16))
            self.v_cache[i][pos].copy_(value.to(torch.bfloat16))
            keys = self.k_cache[i][: pos + 1].float()
            values = self.v_cache[i][: pos + 1].float()
        else:
            self.k_cache[i][pos].copy_(key)
            self.v_cache[i][pos].copy_(value)
            keys = self.k_cache[i][: pos + 1]  # [T, Hkv, D], no stack alloc
            values = self.v_cache[i][: pos + 1]
        repeat = hq // hkv
        if repeat > 1:
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)
        # query is already heads-first [Hq, D]; just add the time axis.
        q = query.unsqueeze(1)  # [Hq, 1, D]
        k = keys.transpose(0, 1)  # [Hq, T, D]
        v = values.transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)).squeeze(1) / math.sqrt(hd)
        probs = torch.softmax(scores.float(), dim=-1).unsqueeze(1)
        context = (probs @ v).transpose(0, 1).reshape(hq * hd)
        context = context * torch.sigmoid(gate.reshape(hq * hd))
        return self._pinned_affine(context, f"{base}.self_attn.o_proj")

    def _moe_step(self, i: int, h: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        base = f"{PREFIX}.model.layers.{i}"
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

    def _expert_affine(
        self, x: torch.Tensor, matrix: AffineQuantMatrix, expert: int
    ) -> torch.Tensor:
        """One expert matvec: upload the expert slice, run, free."""
        rows = matrix.rows
        packed = matrix.packed[expert * rows : (expert + 1) * rows].contiguous()
        scales = matrix.scales[expert * rows : (expert + 1) * rows].contiguous()
        biases = matrix.biases[expert * rows : (expert + 1) * rows].contiguous()
        h_p = self.ext.upload_resident(packed)
        h_s = self.ext.upload_resident(scales)
        h_b = self.ext.upload_resident(biases)
        try:
            return self._affine(
                x, h_p, h_s, h_b, rows,
                matrix.cols, matrix.bits, matrix.group_size,
            )
        finally:
            self.ext.free_resident(h_p)
            self.ext.free_resident(h_s)
            self.ext.free_resident(h_b)

    # -- full decode --------------------------------------------------------
    def decode_step(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """One token forward; ``x`` is the fp32 embed row, ``pos`` its index."""
        cfg = self.config
        _stop = getattr(self, "_bs_stop", None)
        for i, layer_type in enumerate(cfg.layer_types):
            if _stop is not None and i >= _stop and _stop >= 0:
                # final norm + lm_head still apply below
                break
            base = f"{PREFIX}.model.layers.{i}"
            h = self._rms(
                x, self._dense(f"{base}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            if layer_type == "linear_attention":
                x = x + self._gdn_step(i, h)
            else:
                x = x + self._full_attn_step(i, h, pos)
            h2 = self._rms(
                x, self._dense(f"{base}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = x + self._moe_step(i, h2)
            if os.environ.get("FTV_DEBUG_X"):
                import pathlib as _p
                try:
                    with open(os.environ["FTV_DEBUG_X"], "a") as _f:
                        _f.write(
                            f"seq {i} {layer_type} {x.abs().sum().item():.9e} "
                            f"fin={torch.isfinite(x).all().item()}\n"
                        )
                except OSError:
                    pass
        # Final zero-centered RMSNorm (model.norm) before the head —
        # skipping it feeds unnormalized activations to lm_head (salad).
        x = self._rms(
            x, self._dense(f"{PREFIX}.model.norm.weight"),
            cfg.rms_norm_eps,
        )
        # Host GDN state advanced (linear layers); subclasses with GPU
        # mirrors (NVFP4 batched path) re-sync on next use. getattr keeps
        # non-NVFP4 subclasses attribute-free.
        try:
            self._gdn_gpu_dirty = True
        except AttributeError:
            pass
        return self._pinned_affine(x, f"{PREFIX}.lm_head")

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        rng: torch.Generator | None,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        """Single-token sampler: greedy at temperature<=0, else temp/top-k/top-p.

        Publisher sampling for this checkpoint is temp 0.6 / top_k 20 /
        top_p 0.95; greedy can lock onto a pathological top-1 on quirky
        quants, so sampling is a first-class path, not an afterthought.
        """
        if temperature <= 0.0:
            try:
                return int(torch.argmax(logits, dim=-1).item())
            except RuntimeError as exc:
                raise RuntimeError("greedy sampling failed") from exc
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        if top_k > 0:
            top_vals, top_idx = torch.topk(probs, min(top_k, probs.numel()))
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, top_idx, top_vals)
            probs = mask / mask.sum().clamp_min(1e-20)
        if top_p < 1.0:
            order = torch.argsort(probs, descending=True)
            ranked = probs[order]
            keep = ranked.cumsum(-1) <= top_p
            keep[0] = True
            mask = torch.zeros_like(probs)
            mask[order[keep]] = ranked[keep]
            probs = mask / mask.sum().clamp_min(1e-20)
        try:
            return int(torch.multinomial(probs, 1, generator=rng).item())
        except RuntimeError as exc:
            raise RuntimeError("multinomial sampling failed") from exc

    def generate(
        self, prompt_ids: list[int], max_new: int = 64, eos: int | None = None,
        prefill_chunk: int = 1024, log_progress: bool = False,
        temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
        seed: int = 1234,
    ) -> list[int]:
        """Prefill by looping decode_step, then generate (greedy by default).

        120k note: prefill is still one decode_step per token (exact GDN
        recurrence + MoE); ``prefill_chunk`` only controls progress logs so
        a 120k prefill doesn't look hung. True batched chunked-prefill
        (T>1 GPU matvecs) is future work; see full_attn_forward_chunk.
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
        logits = torch.zeros(self.config.vocab_size)
        for token_id in prompt_ids:
            logits = self.decode_step(self.embed_row(token_id), pos)
            pos += 1
            self._seq_len = pos
            if log_progress and (pos % prefill_chunk == 0):
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
            logits = self.decode_step(self.embed_row(next_id), pos)
            pos += 1
        return out
