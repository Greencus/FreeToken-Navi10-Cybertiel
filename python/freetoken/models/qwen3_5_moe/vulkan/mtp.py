"""MTP (nextn) speculative decoding for the Vulkan Qwen3.5-MoE decoder.

A ``qwen35moe`` GGUF may ship one trained MTP block (``blk.40``: a
full-attention layer with its own experts plus the four ``nextn`` tensors
``eh_proj`` / ``enorm`` / ``hnorm`` / ``shared_head_norm``). Upstream
FreeToken drops that block (``models/qwen3_5_moe/gguf.py`` subtracts
``nextn_predict_layers``); llama.cpp's server implements speculative drafting
with it (``common/speculative.cpp::common_speculative_impl_draft_mtp`` plus
the verify/accept loop in ``tools/server/server-context.cpp``). This module
ports that behaviour onto the resident Vulkan engine.

Per llama.cpp, one generation round is:

1. **draft** - feed ``(token, h)`` through the MTP block alone, repeatedly:
   the pending token at position ``p`` with the previous hidden row, then
   each drafted token with the *draft's own* ``shared_head_norm`` output.
   A step costs one full-attention layer + one MoE layer, not a whole model
   pass. A step is kept only while its top-1 probability (renormalized over
   the draft sampler's top-10) is at least ``p_min``.
2. **verify** - run the 40 trunk layers over the ``k+1`` candidate positions
   (the pending token plus the ``k`` drafts) as ONE batched pass.
3. **accept** - ``common_sampler_sample_and_accept_n``: greedy sample each
   verify row, stop at the first draft mismatch, always take one bonus
   sample when every draft matched. ``n_accepted`` is the number of tokens
   kept, so the next position is ``pos + n_accepted``.
4. **catch-up** - re-decode the same positions through the MTP block with the
   *target's* hidden rows shifted right by one, seeded with ``pending_h``
   (llama.cpp's ``process()``), so every MTP KV cell carries a target-side
   h input - the distribution the block was trained on.
5. **rollback** - a partial accept restores the trunk state from a host
   snapshot taken before the verify and re-runs the accepted prefix as one
   batched pass. Causality makes that re-run row-for-row identical to the
   verify rows it replaces; a full accept skips the copy entirely. KV cells
   beyond the new position are never read (queries only attend ``<=`` their
   own position), so stale draft cells are harmless.

The MTP block's KV lives in this module's own tensors and is installed under
key ``40`` only for the duration of a block call, so the trunk's per-layer
walk (``cfg.layer_types``, 40 entries) never sees it and the trunk caches
never mix with it.

Greedy only (``temperature <= 0``): the accept rule above is llama.cpp's
server path (the modified-rejection-sampling variant exists only in its
``examples/speculative``), and the identity gate is defined against the
non-speculative greedy decode.

The trunk weights are reused through the existing pinned/expert machinery;
this module only adds the MTP block's own weights (registered at
``enable_mtp`` time), its KV caches, and the driver loop. ``generate_mtp`` is
the entry point; ``enable_mtp`` prepares a resident model for it.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any

import torch

from freetoken.models.gguf.tensors import float_tensor

from .gguf_provider import GgufWeightProvider
from .gguf_resident import _GgufPinned
from .nvfp4_resident import _translate

# The single MTP block this file's model ships: blk.40 (block_count 41,
# nextn_predict_layers 1). Its index is intentionally outside the trunk's
# layer_types range.
MTP_LAYER = 40

# Tensors the GGUF must carry for the MTP block to be usable. Checked before
# anything is pinned so a checkpoint without a nextn block fails with one
# clear error instead of a KeyError deep in a kernel call.
_REQUIRED_TENSORS = (
    "blk.40.attn_norm.weight",
    "blk.40.post_attention_norm.weight",
    "blk.40.attn_q.weight",
    "blk.40.attn_k.weight",
    "blk.40.attn_v.weight",
    "blk.40.attn_output.weight",
    "blk.40.attn_q_norm.weight",
    "blk.40.attn_k_norm.weight",
    "blk.40.ffn_gate_inp.weight",
    "blk.40.ffn_gate_inp_shexp.weight",
    "blk.40.ffn_gate_shexp.weight",
    "blk.40.ffn_up_shexp.weight",
    "blk.40.ffn_down_shexp.weight",
    "blk.40.ffn_gate_exps.weight",
    "blk.40.ffn_up_exps.weight",
    "blk.40.ffn_down_exps.weight",
    "blk.40.nextn.eh_proj.weight",
    "blk.40.nextn.enorm.weight",
    "blk.40.nextn.hnorm.weight",
    "blk.40.nextn.shared_head_norm.weight",
)

# Dense stems of blk.40 the trunk's own helpers can pin/look up as-is (the
# provider's _fetch parses ".layers.N" generically, so layer 40 needs no
# new mapping - only its name has to be admitted to the provider's set).
_DENSE_STEMS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
    "mlp.shared_expert_gate",
)

# Layer-40 names the trunk's own step/chunk helpers fetch through _dense
# (never pinned; _fetch maps them like any other layer's).
_DENSE_TAILS = ("self_attn.q_norm.weight", "self_attn.k_norm.weight")


def _rms_raw(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Plain RMSNorm: the nextn norms store raw weights, not zero-centered."""
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w.float()


class MtpHead:
    """One MTP block plus the draft / verify / catch-up state around it."""

    def __init__(
        self,
        model: Any,
        n_max: int = 3,
        n_min: int = 0,
        p_min: float = 0.0,
        top_k: int = 10,
    ) -> None:
        tensors = model.tensors
        if not isinstance(tensors, GgufWeightProvider):
            raise ValueError(
                "MTP needs the GGUF weight provider (blk.40 nextn tensors); "
                f"this model was opened from {type(tensors).__name__}"
            )
        missing = [n for n in _REQUIRED_TENSORS if n not in tensors.index]
        if missing:
            raise ValueError(
                "this GGUF has no usable MTP (nextn) block, missing "
                f"{missing[0]} (and {len(missing) - 1} more)"
            )
        cfg = model.config
        if n_max < 0 or n_min < 0 or n_min > n_max or not 0.0 <= p_min <= 1.0:
            raise ValueError(
                f"bad MTP knobs n_max={n_max} n_min={n_min} p_min={p_min}"
            )
        self.model = model
        self.n_max = int(n_max)
        self.n_min = int(n_min)
        self.p_min = float(p_min)
        self.top_k = int(top_k)
        self.eps = cfg.rms_norm_eps

        # Run verify / draft / catch-up on the batched multi-row walk
        # (decode_rows_batched) when the batched decode contract holds -
        # the T>1 K-quant chunk lane stays as the reference/fallback and
        # is forced with FTV_MTP_BATCHED=0.
        self._batched = (
            getattr(model, "_gguf_batched", False)
            and getattr(model, "_packed_kq", False)
            and getattr(model, "_exp_cache_budget", 0) > 0
            and not getattr(model, "_gpu_router", True)
            and os.environ.get("FTV_MTP_BATCHED", "1").strip() != "0"
        )

        # Rollback-snapshot optimization, opt-in (FTV_MTP_SKIP_KV_SNAPSHOT=1):
        # snapshot/restore skip the trunk KV host copies. Sound because every
        # KV read is preceded by a write of the same cell in the same row
        # (_full_attn_host writes k_cache[i][pos] before reading [:pos+1]),
        # so the rejected rows' cells past pos+n_acc are never read before a
        # later round overwrites them. Default 0 = the full-copy semantics
        # this head shipped with (identical tokens either way; the gate
        # asserts that).
        self._skip_kv_snap = (
            os.environ.get("FTV_MTP_SKIP_KV_SNAPSHOT", "0").strip() != "0"
        )

        # Admit blk.40's dense names to the provider, then pin them with the
        # GGUF decoder's own helpers. _fetch already knows every leaf below,
        # so no provider change is needed - only the name set grows. The
        # provider speaks translated keys, the decoder's own helpers take the
        # affine-style stems and translate internally.
        base = f"model.language_model.layers.{MTP_LAYER}"
        stem_base = f"language_model.model.layers.{MTP_LAYER}"
        tensors._names.update(
            f"{base}.{suffix}.weight" for suffix in _DENSE_STEMS
        )
        # (set.update would iterate a bare string into characters, so the
        # fixed names go in one iterable.)
        tensors._names.update(
            (
                f"{base}.input_layernorm.weight",
                f"{base}.post_attention_layernorm.weight",
                f"{base}.mlp.gate.weight",
                *(f"{base}.{tail}" for tail in _DENSE_TAILS),
            )
        )
        for suffix in _DENSE_STEMS:
            model._pin_gguf(f"{stem_base}.{suffix}")

        # nextn-specific weights: eh_proj pins like any dense matrix, the
        # three nextn norms stay host tensors (they are applied in torch).
        self.eh_stem = f"language_model.model.layers.{MTP_LAYER}.mtp.eh_proj"
        packed_eh = None
        if getattr(model, "_dense_q80", False):
            packed_eh = tensors.packed_matrix(
                tensors._t(f"blk.{MTP_LAYER}.nextn.eh_proj")
            )
        if packed_eh is not None:
            slab, _k = packed_eh
            model.pinned[_translate(self.eh_stem)] = _GgufPinned(
                model.ext, slab, packed=True
            )
        else:
            eh = tensors._matrix(
                tensors._t(f"blk.{MTP_LAYER}.nextn.eh_proj"), torch.bfloat16
            )
            model.pinned[_translate(self.eh_stem)] = _GgufPinned(model.ext, eh)
        self.enorm = _nextn_norm(tensors, "enorm")
        self.hnorm = _nextn_norm(tensors, "hnorm")
        self.shared_head_norm = _nextn_norm(tensors, "shared_head_norm")

        # Routed experts of the MTP block join the decoder's VRAM LRU through
        # the same ExpertBank contract. With the GGUF K-quant arena on
        # (_kq_arena), also register layer 40 in _mat_layer: the base
        # helpers that resolve a bank to its layer (_fused_slot0,
        # _down_slot0, hotset walk) must see 40, not None. The registry
        # stays inert on this path (_arena_enabled is forced off for GGUF
        # and _arena_keys is empty), but an accidental lookup has to
        # answer with the right layer instead of silently taking the
        # "legacy path reads rows [0, N)" fallback.
        for proj in ("gate", "up", "down"):
            stem = f"{stem_base}.mlp.switch_mlp.{proj}_proj"
            bank = tensors.expert_bank(MTP_LAYER, proj)
            model.expert_mats[stem] = bank
            if getattr(model, "_kq_arena", False):
                model._mat_layer[id(bank)] = MTP_LAYER

        # The MTP block's own KV cache, same layout/dtype as the trunk's.
        kv_torch_dtype = (
            torch.float32 if model.kv_dtype == "fp32" else torch.bfloat16
        )
        self.k_cache = torch.zeros(
            model.max_len, cfg.num_key_value_heads, cfg.head_dim,
            dtype=kv_torch_dtype,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.pending_h = torch.zeros(cfg.hidden_size)

        # Trunk-state snapshot for the partial-accept rollback (~95 MB: the
        # GDN states, conv history, and the 10 full-attn KV caches). Allocated
        # once so a round's rollback does not churn the host allocator.
        # With FTV_MTP_SKIP_KV_SNAPSHOT=1 the KV half is neither allocated
        # nor copied (see the flag's comment above).
        self._snap = {
            "gdn": {i: torch.empty_like(t) for i, t in model.gdn_state.items()},
            "conv": {i: torch.empty_like(t) for i, t in model.conv_buf.items()},
        }
        if not self._skip_kv_snap:
            self._snap["k"] = {
                i: torch.empty_like(t) for i, t in model.k_cache.items()}
            self._snap["v"] = {
                i: torch.empty_like(t) for i, t in model.v_cache.items()}
        self.stats: dict[str, float] = {}
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        """Fresh sequence: MTP KV and the carried hidden row go to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.pending_h.zero_()
        self.stats = {
            "rounds": 0.0,
            "drafted": 0.0,
            "accepted": 0.0,
            "rollbacks": 0.0,
            "ms_prefill": 0.0,
            "ms_draft": 0.0,
            "ms_verify": 0.0,
            "ms_catchup": 0.0,
            "ms_rollback": 0.0,
            "ms_snapshot": 0.0,
        }

    @contextmanager
    def _mtp_kv(self):
        """Expose this head's KV as layer 40 for one block call."""
        model = self.model
        had_k = MTP_LAYER in model.k_cache
        model.k_cache[MTP_LAYER] = self.k_cache
        model.v_cache[MTP_LAYER] = self.v_cache
        try:
            yield
        finally:
            if not had_k:
                del model.k_cache[MTP_LAYER]
                del model.v_cache[MTP_LAYER]

    # -- MTP block (blk.40) ---------------------------------------------
    def _nextn_input(self, emb: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """eh_proj(norm(embed) ++ norm(h_prev)), the block's input row(s)."""
        e = _rms_raw(emb, self.enorm, self.eps)
        hn = _rms_raw(h_prev, self.hnorm, self.eps)
        return self.model._pinned_affine(
            torch.cat([e, hn], dim=-1), self.eh_stem
        )

    def _nextn_input_chunk(
        self, emb: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        return self.model._pinned_affine_batch(
            torch.cat([_rms_raw(emb, self.enorm, self.eps),
                       _rms_raw(h_prev, self.hnorm, self.eps)], dim=-1),
            self.eh_stem,
        )

    def _block_step(
        self, pos: int, x0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One block pass at ``pos`` (T=1): the batched walk when
        ``_batched``, else the sequential kernels.

        Returns ``(logits [vocab], h_nextn [H])``; ``h_nextn`` is the
        ``shared_head_norm`` output, the draft chain's next h input.
        """
        model = self.model
        cfg = model.config
        base = f"language_model.model.layers.{MTP_LAYER}"
        with self._mtp_kv():
            if self._batched:
                x = model.decode_rows_batched(
                    x0.unsqueeze(0), pos,
                    layers=[(MTP_LAYER, "full_attention")],
                    final_norm=False,
                )
                hn = _rms_raw(x[0], self.shared_head_norm, cfg.rms_norm_eps)
                return model._pinned_affine(hn, "language_model.lm_head"), hn
            h = model._rms(
                x0, model._dense(f"{base}.input_layernorm.weight"), cfg.rms_norm_eps
            )
            x = x0 + model._full_attn_step(MTP_LAYER, h, pos)
            h2 = model._rms(
                x,
                model._dense(f"{base}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = x + model._moe_step(MTP_LAYER, h2)
        hn = _rms_raw(x, self.shared_head_norm, cfg.rms_norm_eps)
        return model._pinned_affine(hn, "language_model.lm_head"), hn

    def _block_chunk(
        self, pos: int, x0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The same block pass over ``x0``'s rows (batched walk when
        ``_batched``, else the T>1 chunk kernels)."""
        model = self.model
        cfg = model.config
        base = f"language_model.model.layers.{MTP_LAYER}"
        tc = x0.shape[0]
        with self._mtp_kv():
            if self._batched:
                x = model.decode_rows_batched(
                    x0, pos,
                    layers=[(MTP_LAYER, "full_attention")],
                    final_norm=False,
                )
                hn = _rms_raw(x, self.shared_head_norm, cfg.rms_norm_eps)
                logits = model._pinned_affine_batch(
                    hn, "language_model.lm_head")
                assert logits.shape == (tc, model.config.vocab_size)
                return logits, hn
            h = model._rms(
                x0, model._dense(f"{base}.input_layernorm.weight"), cfg.rms_norm_eps
            )
            x = x0 + model._full_attn_chunk(MTP_LAYER, h, pos)
            h2 = model._rms(
                x,
                model._dense(f"{base}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = x + model._moe_chunk(MTP_LAYER, h2)
        hn = _rms_raw(x, self.shared_head_norm, cfg.rms_norm_eps)
        logits = model._pinned_affine_batch(hn, "language_model.lm_head")
        assert logits.shape == (tc, model.config.vocab_size)
        return logits, hn

    # -- trunk -----------------------------------------------------------
    def verify(self, pos: int, token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """One batched trunk pass over ``token_ids`` at ``pos``.

        Returns ``(logits [T, vocab], h_nextn [T, H])`` - the post-final-norm
        hidden rows, which is what the MTP catch-up feeds back.
        """
        model = self.model
        cfg = model.config
        x = torch.stack([model.embed_row(t) for t in token_ids])
        if self._batched:
            # The batched walk: one command buffer per layer phase for all
            # T rows; it leaves _gdn_gpu_dirty False (the GPU GDN mirrors
            # just became authoritative), unlike the chunk lane below.
            x = model.decode_rows_batched(x, pos)
            logits = model._pinned_affine_batch(x, "language_model.lm_head")
            return logits, x
        for i, layer_type in enumerate(cfg.layer_types):
            base = f"language_model.model.layers.{i}"
            h = model._rms(
                x, model._dense(f"{base}.input_layernorm.weight"), cfg.rms_norm_eps
            )
            if layer_type == "linear_attention":
                x = x + model._gdn_chunk(i, h)
            else:
                x = x + model._full_attn_chunk(i, h, pos)
            h2 = model._rms(
                x,
                model._dense(f"{base}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = x + model._moe_chunk(i, h2)
        x = model._rms(
            x, model._dense("language_model.model.norm.weight"), cfg.rms_norm_eps
        )
        model._gdn_gpu_dirty = True
        logits = model._pinned_affine_batch(x, "language_model.lm_head")
        return logits, x

    def snapshot(self) -> None:
        """Copy the trunk's recurrent + KV state (rollback point).

        With host state authoritative (chunk/sequential lane ran last)
        the host rows are copied as-is; with the GPU mirrors authoritative
        (batched walk ran last) they are pulled down first - the host rows
        may be stale then, and ``restore`` always writes host -> dirty.
        The KV caches are host tensors in both lanes (attention runs on
        host), so they are copied directly either way - unless
        ``FTV_MTP_SKIP_KV_SNAPSHOT=1`` elides the KV half (see the flag's
        comment in ``__init__``: every KV cell is written before it is
        read, so the copies only preserve unread garbage).
        """
        model = self.model
        if model._gdn_gpu_dirty:
            for i, t in model.gdn_state.items():
                self._snap["gdn"][i].copy_(t)
            for i, t in model.conv_buf.items():
                self._snap["conv"][i].copy_(t)
        else:
            ext = model.ext
            ext.batch_begin()
            for i in model.gdn_state_gpu:
                ext.batch_download(
                    model.gdn_state_gpu[i], self._snap["gdn"][i])
            for i in model.conv_buf_gpu:
                ext.batch_download(
                    model.conv_buf_gpu[i], self._snap["conv"][i])
            ext.batch_submit()
        if not self._skip_kv_snap:
            for i, t in model.k_cache.items():
                self._snap["k"][i].copy_(t)
            for i, t in model.v_cache.items():
                self._snap["v"][i].copy_(t)

    def restore(self) -> None:
        """Put the trunk back to the last snapshot."""
        model = self.model
        for i, t in self._snap["gdn"].items():
            model.gdn_state[i].copy_(t)
        for i, t in self._snap["conv"].items():
            model.conv_buf[i].copy_(t)
        if not self._skip_kv_snap:
            for i, t in self._snap["k"].items():
                model.k_cache[i].copy_(t)
            for i, t in self._snap["v"].items():
                model.v_cache[i].copy_(t)
        model._gdn_gpu_dirty = True

    # -- draft -----------------------------------------------------------
    def draft(
        self, pos: int, token_id: int, h_prev: torch.Tensor, n_max: int, p_min: float
    ) -> list[int]:
        """``n_max`` drafted tokens via the MTP block alone.

        llama.cpp's draft(): step 0 pairs the pending token with the target's
        previous hidden row; later steps pair each drafted token with the
        draft's own ``shared_head_norm`` output, and stop early once the top-1
        probability (renormalized over the top-10 the draft sampler sees)
        falls below ``p_min``.
        """
        model = self.model
        drafts: list[int] = []
        token = token_id
        h = h_prev
        for step in range(n_max):
            at = pos + step
            if at >= model.max_len:
                break
            x0 = self._nextn_input(model.embed_row(token), h)
            logits, hn = self._block_step(at, x0)
            top_vals, top_idx = torch.topk(logits, min(self.top_k, logits.numel()))
            p1 = float(torch.softmax(top_vals.float(), dim=-1)[0])
            if p1 < p_min:
                break
            token = int(top_idx[0])
            drafts.append(token)
            h = hn
        return drafts

    def catchup(
        self, pos: int, token_ids: list[int], h_prevs: torch.Tensor
    ) -> None:
        """llama.cpp's process(): re-decode these positions with target h.

        ``h_prevs`` is one row per position: the carried ``pending_h`` first,
        then the verify batch's hidden rows shifted right by one. The block's
        KV cells for these positions are overwritten with those inputs, which
        is what the draft's next round attends to.
        """
        emb = torch.stack([self.model.embed_row(t) for t in token_ids])
        x0 = self._nextn_input_chunk(emb, h_prevs)
        self._block_chunk(pos, x0)


def accept_greedy(logits_rows: torch.Tensor, drafts: list[int]) -> list[int]:
    """llama.cpp's ``common_sampler_sample_and_accept_n``, greedy sampling.

    Sample each verify row, push it, stop at the first draft mismatch; when
    every draft matched, push one more (bonus) sample. The returned tokens
    are what the round commits, so the round advances ``len(result)``
    positions and keeps ``len(result) - 1`` drafted tokens.
    """
    result: list[int] = []
    for i, draft in enumerate(drafts):
        token = int(logits_rows[i].argmax().item())
        result.append(token)
        if token != draft:
            return result
    result.append(int(logits_rows[len(drafts)].argmax().item()))
    return result


def _nextn_norm(tensors: GgufWeightProvider, leaf: str) -> torch.Tensor:
    """One nextn norm weight, raw (they are not zero-centered)."""
    info = tensors.index[f"blk.{MTP_LAYER}.nextn.{leaf}.weight"]
    return float_tensor(info).float().contiguous()


def enable_mtp(
    model: Any, n_max: int = 3, n_min: int = 0, p_min: float = 0.0, top_k: int = 10
) -> MtpHead:
    """Prepare ``model`` for MTP drafting; idempotent, returns the head.

    Pins the MTP block's dense weights in VRAM, registers its expert banks
    with the decoder's LRU, and allocates its KV caches. Costs ~80 MB of
    pinned VRAM plus whatever the block's experts add to the LRU.
    """
    head = getattr(model, "mtp", None)
    if head is None:
        head = MtpHead(model, n_max=n_max, n_min=n_min, p_min=p_min, top_k=top_k)
        model.mtp = head
    else:
        head.n_max, head.n_min, head.p_min, head.top_k = (
            int(n_max), int(n_min), float(p_min), int(top_k)
        )
    return head


def prefill_with_h(
    model: Any, ids: list[int], chunk_size: int = 128, log_progress: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trunk prefill that also returns every prompt position's hidden row.

    ``ResidentNVFP4.prefill_chunked`` is the single source of truth for the
    prefill math (MTP identity would silently break if this re-implemented
    it), so it runs unchanged; the final norm's output - the post-output-norm
    hidden the MTP block consumes - is collected by a temporary hook on the
    decoder's ``_rms``. That hook matches on the norm weight's identity, and
    the final norm is the only RMSNorm call whose weight is the trunk's.
    """
    norm_w = model._dense("language_model.model.norm.weight")
    captured: list[torch.Tensor] = []
    orig = model._rms

    def _hook(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        y = orig(x, w, eps)
        if w is norm_w and x.dim() == 2:
            captured.append(y)
        return y

    model._rms = _hook  # instance attribute shadows the class staticmethod
    try:
        logits = model.prefill_chunked(
            ids, chunk_size=chunk_size, log_progress=log_progress
        )
    finally:
        del model._rms
    if not captured:
        raise RuntimeError("prefill_with_h: final-norm hook never fired")
    return logits, torch.cat(captured, dim=0)


def generate_mtp(
    model: Any,
    prompt_ids: list[int],
    max_new: int = 64,
    eos: int | None = None,
    prefill_chunk: int = 128,
    log_progress: bool = False,
    n_max: int = 3,
    n_min: int = 0,
    p_min: float = 0.0,
    top_k: int = 10,
) -> list[int]:
    """MTP speculative decode; same greedy tokens as ``model.generate``.

    Greedy only: the accept rule compares drafted tokens against the
    target's greedy samples, so a non-greedy target would need the
    modified rejection sampling this port (like llama.cpp's server path)
    does not implement.

    ``n_max`` drafts per round, ``n_min`` the fewest drafts worth keeping,
    ``p_min`` the draft top-1 probability gate (llama.cpp's server defaults
    are 3 / 0 / 0.0; the fork recommends p_min 0.25 for a 35B-A3B). The
    per-round timings and accept counters land on ``model.mtp.stats``.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if max_new < 0:
        raise ValueError(f"max_new must be >= 0, got {max_new}")
    head = enable_mtp(model, n_max=n_max, n_min=n_min, p_min=p_min, top_k=top_k)

    model.reset()
    head.reset()
    t0 = time.perf_counter()
    logits, h_rows = prefill_with_h(
        model, prompt_ids, chunk_size=prefill_chunk, log_progress=log_progress
    )
    head.stats["ms_prefill"] = (time.perf_counter() - t0) * 1e3
    pos = model._seq_len
    pending_h = h_rows[-1].clone()
    # llama.cpp runs process() over the whole prompt during prefill, so the
    # draft ctx's own KV holds every prompt position (h shifted right, zeros
    # before the prompt - its pending_h starts zeroed). Without this the
    # first round's drafts attend to an empty cache and always reject.
    t0 = time.perf_counter()
    catch = torch.cat(
        [torch.zeros(1, h_rows.shape[1], dtype=h_rows.dtype), h_rows[:-1]], dim=0
    )
    for start in range(0, len(prompt_ids), max(1, prefill_chunk)):
        head.catchup(
            start,
            prompt_ids[start : start + prefill_chunk],
            catch[start : start + prefill_chunk],
        )
    head.stats["ms_prefill"] += (time.perf_counter() - t0) * 1e3
    # The prompt's own next-token sample: already emitted, not yet decoded.
    # Every later token comes out of a verify row, exactly as llama.cpp
    # emits slot.sampled before its first speculative round.
    pending_id = int(logits.argmax().item())

    out: list[int] = []
    if max_new > 0:
        out.append(pending_id)
        if eos is not None and pending_id == eos:
            return out
    while len(out) < max_new:
        if pos >= model.max_len:
            break
        room = model.max_len - pos - 1
        if room < 1:
            # Only one position left: no room to draft, single-row trunk pass.
            logits_row, h_row = head.verify(pos, [pending_id])
            pending_h = h_row[0]
            pos += 1
            model._seq_len = pos
            pending_id = int(logits_row[0].argmax().item())
            out.append(pending_id)
            if eos is not None and pending_id == eos:
                break
            continue

        n_draft = min(head.n_max, room)
        t0 = time.perf_counter()
        drafts = head.draft(pos, pending_id, pending_h, n_draft, head.p_min)
        head.stats["ms_draft"] += (time.perf_counter() - t0) * 1e3
        if len(drafts) < head.n_min:
            drafts = []
        verify_ids = [pending_id, *drafts]
        m = len(verify_ids)

        t0 = time.perf_counter()
        head.snapshot()
        head.stats["ms_snapshot"] += (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        logits_rows, h_rows_v = head.verify(pos, verify_ids)
        head.stats["ms_verify"] += (time.perf_counter() - t0) * 1e3
        if not torch.isfinite(logits_rows).all():
            raise RuntimeError(
                f"non-finite MTP verify logits at position {pos}"
            )

        # common_sampler_sample_and_accept_n, greedy.
        emitted = accept_greedy(logits_rows, drafts)
        n_acc = len(emitted)  # accepted drafts + the one bonus token

        # process(): the MTP block re-decodes these positions with the target
        # hidden rows shifted right, seeded with the carried pending_h.
        t0 = time.perf_counter()
        catch_h = torch.cat([pending_h.unsqueeze(0), h_rows_v[:-1]], dim=0)
        head.catchup(pos, verify_ids, catch_h)
        head.stats["ms_catchup"] += (time.perf_counter() - t0) * 1e3

        if n_acc < m:
            t0 = time.perf_counter()
            head.restore()
            # Always re-run the kept prefix, including the single seed row:
            # the verify already computed it, but the restore just threw its
            # KV cells and its GDN state updates away with the rejected rows.
            # The prefix rows are causally independent of the rows that
            # followed them, so this re-run is row-for-row what the verify
            # computed - only the positions past n_acc stay undone.
            head.verify(pos, verify_ids[:n_acc])
            head.stats["ms_rollback"] += (time.perf_counter() - t0) * 1e3
            head.stats["rollbacks"] += 1.0

        pos += n_acc
        model._seq_len = pos
        pending_h = h_rows_v[n_acc - 1].clone()
        pending_id = emitted[-1]
        head.stats["rounds"] += 1.0
        head.stats["drafted"] += float(len(drafts))
        head.stats["accepted"] += float(max(0, n_acc - 1))

        done = False
        for token in emitted:
            if len(out) >= max_new:
                break
            out.append(token)
            if eos is not None and token == eos:
                done = True
                break
        if done or len(out) >= max_new:
            break
    return out


__all__ = [
    "MtpHead",
    "MTP_LAYER",
    "accept_greedy",
    "enable_mtp",
    "generate_mtp",
    "prefill_with_h",
]
