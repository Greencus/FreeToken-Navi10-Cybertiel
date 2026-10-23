"""GGUF weight provider for the Vulkan Qwen3.5-MoE decoder (``ResidentGGUF``).

The Vulkan backend has no ggml kernels, so a ``qwen35moe`` GGUF is served by
dequantizing its tensors on the host (mmap, never a whole-file copy) into
ordinary torch tensors and handing them to the resident BF16/FP16/FP32 linear
kernels. This module is the weight-source layer: it duck-types
``ShardedSafetensors`` (``.get`` / ``.contains``) so the existing decoder
(``ResidentNVFP4``'s ``_dense`` / ``_pinned_affine`` / GDN param init) works
unchanged, answering the *translated* Ressl-style keys the engine asks for
(``model.language_model.layers.N.*``, ``lm_head.weight``).

Transforms are exactly what upstream's read-only loader
``freetoken/models/qwen3_5_moe/gguf.py::iter_gguf_weights`` applies (zero-
centred norms get the ``-1.0`` back-off here, GDN value-head tiles are
untiled, ``A_log`` is stored as ``-exp(A)``); the runtime permutation of
``ssm_out`` stays in the engine (input-side), matching the upstream loader.

Routed experts stay packed in the file: :meth:`GgufWeightProvider.expert_bf16`
dequantizes ONE expert slab on demand (Q4_K/Q5_K -> BF16) through the shared
host LRU ``DequantCache``; the VRAM LRU lives in the decoder.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import is_quantized, row_bytes
from freetoken.models.gguf.dequant import (
    DequantCache,
    dequant_expert,
    dequant_matrix,
    torch_shape,
)
from freetoken.models.gguf.packed import kquant_qtype, packed_expert, packed_view_tensor
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index, load_gguf_metadata
from freetoken.models.gguf.tensors import float_tensor
from freetoken.third_party.gguf import GGMLQuantizationType
from freetoken.third_party.gguf.quants import dequantize as _np_dequantize

from .config import Qwen35MoeConfig

# Packed-byte size above which 2-D matrices are dequantized in row chunks
# (bounds the fp32 transient; lm_head/token_embd are ~515 MB packed each).
_CHUNK_BYTES = 64 * 1024 * 1024
_CHUNK_ROWS = 32768

# FTV_DENSE_Q80: translated engine leaf -> (file leaf, gather kind).
# "row" = identity (zero-copy mmap view); "qkv"/"z" = the provider's GDN
# transform is a pure row permutation -> byte-row gather; "out" = the
# provider's out_proj is an hv-tile column permutation, block-aligned ->
# per-row block-group regather.  Everything else stays a BF16 pin.
_PACKABLE_Q80 = {
    "self_attn.q_proj.weight": ("attn_q", "row"),
    "self_attn.o_proj.weight": ("attn_output", "row"),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp", "row"),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp", "row"),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp", "row"),
    "linear_attn.in_proj_qkv.weight": ("attn_qkv", "qkv"),
    "linear_attn.in_proj_z.weight": ("attn_gate", "z"),
    "linear_attn.out_proj.weight": ("ssm_out", "out"),
}


def _packed_verify_enabled() -> bool:
    return os.environ.get("FTV_DENSE_Q80_VERIFY", "0").strip() != "0"

_FLOAT_DTYPES = {
    int(GGMLQuantizationType.F32): torch.float32,
    int(GGMLQuantizationType.F16): torch.float16,
    int(GGMLQuantizationType.BF16): torch.bfloat16,
}


def gguf_config(model_path: str | pathlib.Path) -> Qwen35MoeConfig:
    """The decoder config for a ``qwen35moe`` GGUF, from its metadata.

    Mirrors upstream ``parse_gguf_config`` field-for-field (block_count
    minus the MTP nextn layers, interval/bitmapped layer typing, ssm dims).
    """
    path = str(model_path)
    md = load_gguf_metadata(path)
    arch = md.get("general.architecture")
    if arch != "qwen35moe":
        raise ValueError(f"expected qwen35moe GGUF architecture, got {arch!r}")

    def kv(key: str):
        return md[f"qwen35moe.{key}"]

    head_dim = int(kv("attention.key_length"))
    n_v = int(kv("ssm.time_step_rank"))
    num_layers = int(kv("block_count")) - int(md.get("qwen35moe.nextn_predict_layers", 0))
    recurrent = md.get("qwen35moe.attention.recurrent_layers")
    if recurrent is not None:
        layer_types = [
            "linear_attention" if r else "full_attention"
            for r in list(recurrent)[:num_layers]
        ]
    else:
        interval = int(kv("full_attention_interval"))
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]
    tokens = md.get("tokenizer.ggml.tokens") or ()
    return Qwen35MoeConfig(
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=head_dim,
        partial_rotary_factor=int(kv("rope.dimension_count")) / head_dim,
        rope_theta=float(kv("rope.freq_base")),
        mrope_section=[int(s) for s in kv("rope.dimension_sections")][:3],
        num_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        moe_intermediate_size=int(kv("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(kv("expert_shared_feed_forward_length")),
        vocab_size=len(tokens),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        linear_num_key_heads=int(kv("ssm.group_count")),
        linear_num_value_heads=n_v,
        linear_key_head_dim=int(kv("ssm.state_size")),
        linear_value_head_dim=int(kv("ssm.inner_size")) // n_v,
        linear_conv_kernel_dim=int(kv("ssm.conv_kernel")),
    ).validate()


def untile(t: torch.Tensor, num_k_heads: int, head_dim: int) -> torch.Tensor:
    """Tiled value heads (k-head minor) back to grouped (k-head major).

    Mirror of ``qwen3_5_moe/gguf.py::_untile``; an involution on the row
    axis (transpose(0,1) twice is identity), which the tests assert.
    """
    n = t.shape[0]
    if n % (num_k_heads * head_dim):
        raise ValueError(
            f"untile: rows {n} not divisible by {num_k_heads}*{head_dim}"
        )
    r = n // (num_k_heads * head_dim)
    idx = torch.arange(n).view(r, num_k_heads, head_dim).transpose(0, 1).reshape(n)
    return t.index_select(0, idx)


class _Shape:
    """Zero-storage shape proxy for the arena's per-expert shape-peek
    (``expert_mats[stem][0]`` unpacked as a 3-tuple with a ``.shape``)."""

    __slots__ = ("shape",)

    def __init__(self, rows: int, cols: int) -> None:
        self.shape = (rows, cols)


class ExpertBank(list):
    """256 per-expert proxy tuples ``(shape_proxy, None, 1.0)``.

    List-compatible with the base decoder's expert-mats contract; the
    ``layer``/``proj`` identity lets the decoder dequantize the right slab
    of the 3-D GGUF bank on a VRAM-cache miss. ``qtype``/``row_bytes``/
    ``packed_bytes`` describe the IN-SHADER packed lane (K-quant shaders
    unpack the GGUF bytes verbatim); ``qtype is None`` means that bank has
    no shader unpacker and must keep dequantizing on the host.
    """

    def __init__(
        self,
        layer: int,
        proj: str,
        rows: int,
        cols: int,
        count: int,
        *,
        qtype: str | None = None,
        row_bytes: int = 0,
    ) -> None:
        super().__init__((_Shape(rows, cols), None, 1.0) for _ in range(count))
        self.layer = layer
        self.proj = proj
        self.rows = rows
        self.cols = cols
        self.qtype = qtype
        self.row_bytes = row_bytes
        self.packed_bytes = rows * row_bytes


class _EmbedRows:
    """Lazy token-embedding rows: one Q8_0 row dequantized per ``[id]``."""

    def __init__(self, info: GgufTensorInfo) -> None:
        self._info = info
        self.shape = torch_shape(info)
        self.dtype = torch.float32

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, token_id: int) -> torch.Tensor:
        info = self._info
        rows = self.shape[0]
        if not -rows <= token_id < rows:
            raise IndexError(f"token {token_id} outside [0, {rows})")
        if token_id < 0:
            token_id += rows
        k = int(info.ne[0])
        if is_quantized(info.ggml_type):
            rb = row_bytes(info.ggml_type, k)
            buf = np.frombuffer(
                info.data(), dtype=np.uint8, count=rb, offset=token_id * rb
            ).reshape(1, rb)
            floats = _np_dequantize(buf, GGMLQuantizationType(info.ggml_type))
        else:
            np_dt = torch.empty(0, dtype=_FLOAT_DTYPES[info.ggml_type]).numpy().dtype
            floats = np.frombuffer(
                info.data(), dtype=np_dt, count=k, offset=token_id * k * np_dt.itemsize
            ).reshape(1, k)
        return torch.from_numpy(np.ascontiguousarray(floats)).view(-1).float()


class GgufWeightProvider:
    """Weight source for ``ResidentGGUF``: config + tensors from one GGUF.

    Duck-types ``ShardedSafetensors``: ``.get(name)`` / ``.contains(name)``
    with the engine's translated names (see module doc). Everything is
    mmap-backed; dense results are cached (fp32/ bf16/ fp16 torch tensors),
    experts go through the shared host ``DequantCache``.
    """

    def __init__(self, model_path: str | pathlib.Path) -> None:
        self.model_path = str(model_path)
        self.config = gguf_config(self.model_path)
        self.index: dict[str, GgufTensorInfo] = gguf_tensor_index(self.model_path)
        self._dcache = DequantCache.from_env()
        self._cache: dict[str, torch.Tensor] = {}
        cfg = self.config
        nk, hv = cfg.linear_num_key_heads, cfg.linear_value_head_dim
        # q+k row count before the v block: key dim (upstream v_start),
        # not the value dim - equal on this model but the load path is
        # keyed on the state layout, not the value layout.
        self._gdn_v_start = 2 * nk * cfg.linear_key_head_dim
        self._names: set[str] = set()
        for i, layer_type in enumerate(cfg.layer_types):
            base = f"model.language_model.layers.{i}"
            self._names.update({
                f"{base}.input_layernorm.weight",
                f"{base}.post_attention_layernorm.weight",
                f"{base}.mlp.gate.weight",
                f"{base}.mlp.shared_expert.gate_proj.weight",
                f"{base}.mlp.shared_expert.up_proj.weight",
                f"{base}.mlp.shared_expert.down_proj.weight",
                f"{base}.mlp.shared_expert_gate.weight",
            })
            if layer_type == "linear_attention":
                self._names.update({
                    f"{base}.linear_attn.in_proj_qkv.weight",
                    f"{base}.linear_attn.in_proj_z.weight",
                    f"{base}.linear_attn.in_proj_a.weight",
                    f"{base}.linear_attn.in_proj_b.weight",
                    f"{base}.linear_attn.out_proj.weight",
                    f"{base}.linear_attn.conv1d.weight",
                    f"{base}.linear_attn.A_log",
                    f"{base}.linear_attn.dt_bias",
                    f"{base}.linear_attn.norm.weight",
                })
            else:
                self._names.update({
                    f"{base}.self_attn.q_proj.weight",
                    f"{base}.self_attn.k_proj.weight",
                    f"{base}.self_attn.v_proj.weight",
                    f"{base}.self_attn.o_proj.weight",
                    f"{base}.self_attn.q_norm.weight",
                    f"{base}.self_attn.k_norm.weight",
                })
        self._names.update({
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        })

    # -- ShardedSafetensors duck type ------------------------------------
    def contains(self, name: str) -> bool:
        return name in self._names

    def get(self, name: str) -> Any:
        if name not in self._names:
            raise KeyError(f"GGUF provider has no tensor {name!r}")
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        out = self._fetch(name)
        if name != "model.language_model.embed_tokens.weight":
            self._cache[name] = out
        return out

    # -- fetch / transforms ----------------------------------------------
    def _t(self, leaf: str) -> GgufTensorInfo:
        for key in (f"{leaf}.weight", leaf):
            info = self.index.get(key)
            if info is not None:
                return info
        raise KeyError(f"GGUF file has no tensor {leaf!r}")

    def _fetch(self, name: str) -> Any:
        if name == "model.language_model.embed_tokens.weight":
            return _EmbedRows(self._t("token_embd"))
        if name == "model.language_model.norm.weight":
            return float_tensor(self._t("output_norm")) - 1.0
        if name == "lm_head.weight":
            return self._matrix(self._t("output"), torch.bfloat16)
        marker = ".layers."
        li = name.index(marker) + len(marker)
        dot = name.index(".", li)
        layer = int(name[li:dot])
        leaf = name[dot + 1:]  # "linear_attn.A_log", "self_attn.q_proj.weight", ...
        blk = f"blk.{layer}"
        cfg = self.config
        nk, hv = cfg.linear_num_key_heads, cfg.linear_value_head_dim
        v_start = self._gdn_v_start
        if leaf == "input_layernorm.weight":
            return float_tensor(self._t(f"{blk}.attn_norm")) - 1.0
        if leaf == "post_attention_layernorm.weight":
            return float_tensor(self._t(f"{blk}.post_attention_norm")) - 1.0
        if leaf == "mlp.gate.weight":
            return float_tensor(self._t(f"{blk}.ffn_gate_inp")).float()
        if leaf == "mlp.shared_expert.gate_proj.weight":
            return self._matrix(self._t(f"{blk}.ffn_gate_shexp"), torch.bfloat16)
        if leaf == "mlp.shared_expert.up_proj.weight":
            return self._matrix(self._t(f"{blk}.ffn_up_shexp"), torch.bfloat16)
        if leaf == "mlp.shared_expert.down_proj.weight":
            return self._matrix(self._t(f"{blk}.ffn_down_shexp"), torch.bfloat16)
        if leaf == "mlp.shared_expert_gate.weight":
            return (
                float_tensor(self._t(f"{blk}.ffn_gate_inp_shexp")).float().reshape(1, -1)
            )
        if leaf == "linear_attn.in_proj_qkv.weight":
            rows = self._matrix(self._t(f"{blk}.attn_qkv"), torch.bfloat16)
            return torch.cat([rows[:v_start], untile(rows[v_start:], nk, hv)])
        if leaf == "linear_attn.in_proj_z.weight":
            return untile(
                self._matrix(self._t(f"{blk}.attn_gate"), torch.bfloat16), nk, hv
            )
        if leaf == "linear_attn.in_proj_a.weight":
            return untile(float_tensor(self._t(f"{blk}.ssm_alpha")).float(), nk, 1)
        if leaf == "linear_attn.in_proj_b.weight":
            return untile(float_tensor(self._t(f"{blk}.ssm_beta")).float(), nk, 1)
        if leaf == "linear_attn.out_proj.weight":
            # llama.cpp stores ssm_out with tiled value-head columns; upstream
            # permutes its INPUT at runtime (`out_heads_tiled` in gdn.py), but
            # the Vulkan engine has no such hook, so bake the permutation into
            # the weight: grouped head j = i_k*r + i_r -> tiled slot i_r*nk + i_k.
            w = self._matrix(self._t(f"{blk}.ssm_out"), torch.bfloat16)
            nv = cfg.linear_num_value_heads
            r = nv // nk
            head = torch.arange(nv)
            tiled = (head % r) * nk + (head // r)
            cols = (tiled[:, None] * hv + torch.arange(hv)).reshape(-1)
            return w.index_select(1, cols)
        if leaf == "linear_attn.conv1d.weight":
            conv = float_tensor(self._t(f"{blk}.ssm_conv1d")).float()
            conv = torch.cat([conv[:v_start], untile(conv[v_start:], nk, hv)])
            return conv.unsqueeze(1)  # (C, 1, K); the engine slices [:, 0, :]
        if leaf == "linear_attn.A_log":
            return untile(
                torch.log(-float_tensor(self._t(f"{blk}.ssm_a")).float()), nk, 1
            )
        if leaf == "linear_attn.dt_bias":
            return untile(float_tensor(self._t(f"{blk}.ssm_dt.bias")).float(), nk, 1)
        if leaf == "linear_attn.norm.weight":
            # RMSNormGated weight is NOT zero-centred (upstream: no -1.0).
            return float_tensor(self._t(f"{blk}.ssm_norm")).float()
        if leaf == "self_attn.q_proj.weight":
            return self._matrix(self._t(f"{blk}.attn_q"), torch.bfloat16)
        if leaf == "self_attn.k_proj.weight":
            return self._matrix(self._t(f"{blk}.attn_k"), torch.float16)
        if leaf == "self_attn.v_proj.weight":
            return self._matrix(self._t(f"{blk}.attn_v"), torch.float16)
        if leaf == "self_attn.o_proj.weight":
            return self._matrix(self._t(f"{blk}.attn_output"), torch.bfloat16)
        if leaf == "self_attn.q_norm.weight":
            return float_tensor(self._t(f"{blk}.attn_q_norm")) - 1.0
        if leaf == "self_attn.k_norm.weight":
            return float_tensor(self._t(f"{blk}.attn_k_norm")) - 1.0
        raise KeyError(f"no GGUF mapping for {name!r}")

    def _matrix(self, info: GgufTensorInfo, dtype: torch.dtype) -> torch.Tensor:
        """Whole 2-D tensor in torch order at ``dtype``; row-chunked when the
        packed bytes exceed ``_CHUNK_BYTES`` (bounds the fp32 transient)."""
        if info.nbytes <= _CHUNK_BYTES:
            return dequant_matrix(info, dtype=dtype)
        k = int(info.ne[0])
        rows_total = int(np.prod(info.ne[1:]))
        if is_quantized(info.ggml_type) and k % 256:
            # K-quant superblocks span 256 elements; row-chunking is only
            # safe when a row holds whole superblocks. Never hit here
            # (chunked path is Q8_0/F16 embeddings/heads), but be explicit.
            return dequant_matrix(info, dtype=dtype)
        out = torch.empty(rows_total, k, dtype=dtype)
        for r0 in range(0, rows_total, _CHUNK_ROWS):
            r1 = min(rows_total, r0 + _CHUNK_ROWS)
            out[r0:r1] = _dequant_rows(info, r0, r1).to(dtype)
        return out

    # -- FTV_DENSE_Q80: the file's OWN Q8_0 bytes ------------------------
    def packed_matrix(
        self, info: GgufTensorInfo
    ) -> tuple[torch.Tensor, int] | None:
        """The file's raw Q8_0 slab as ``(rows, row_bytes)`` uint8 + K.

        Zero-copy view of the mmap for identity stems (the GEMV then
        reads the file-exact weights - MORE faithful than the dequant ->
        BF16-rounded copy - at 1.0625 B/weight instead of 2).  Returns
        None when the tensor is not Q8_0 or K is not a whole number of
        32-value blocks: the caller falls back to the BF16 pin.
        """
        if int(info.ggml_type) != int(GGMLQuantizationType.Q8_0):
            return None
        k = int(info.ne[0])
        if k % 32:
            return None
        rows = int(np.prod(info.ne[1:]))
        rb = (k // 32) * 34
        slab = np.frombuffer(info.data(), dtype=np.uint8, count=rows * rb)
        return packed_view_tensor(slab.reshape(rows, rb)), k

    def get_packed(self, name: str) -> tuple[torch.Tensor, int] | None:
        """Packed Q8_0 bytes for a translated engine name, or None.

        None (-> plain BF16 pin) for tensors that are not Q8_0 in the
        file (k/v are F16, norms/GDN params F32) and for the two GDN
        stems whose provider transform is not identity: those are
        gathered byte-exactly on the host (row/block moves only, no
        dequant) so the pinned rows still match what the engine expects.
        ``FTV_DENSE_Q80_VERIFY=1`` cross-checks the gathered rows against
        the dequantized BF16 reference (gates; costs one dequant).
        """
        if name == "lm_head.weight":
            leaf, kind = "output", "row"
            blk = None
        elif ".layers." in name:
            marker = ".layers."
            li = name.index(marker) + len(marker)
            dot = name.index(".", li)
            blk = f"blk.{int(name[li:dot])}"
            leaf_map = _PACKABLE_Q80.get(name[dot + 1:])
            if leaf_map is None:
                return None
            leaf, kind = leaf_map
        else:
            return None
        info = self.index.get(f"{blk}.{leaf}.weight" if blk else f"{leaf}.weight")
        if info is None:
            return None
        packed = self.packed_matrix(info)
        if packed is None:
            return None
        slab, k = packed
        rows = int(slab.shape[0])
        nk, hv = self.config.linear_num_key_heads, self.config.linear_value_head_dim

        if kind == "row":
            out = slab
        elif kind in ("qkv", "z"):
            # provider: torch.cat([rows[:v_start], untile(rows[v_start:])])
            # / untile(rows) - a pure ROW permutation, applied to the byte
            # rows as an index gather (per-row bytes never change).
            start = self._gdn_v_start if kind == "qkv" else 0
            n = rows - start
            r = n // (nk * hv)
            # torch .view(r,nk,hv).transpose(0,1) == numpy .transpose(1,0,2)
            # (numpy wants the full permutation, not a 2-dim swap).
            idx = np.arange(n).reshape(r, nk, hv).transpose(1, 0, 2).reshape(-1)
            perm = np.concatenate([np.arange(start), start + idx])
            out = torch.from_numpy(np.take(slab.numpy(), perm, axis=0))
        elif kind == "out":
            # provider: w.index_select(1, cols) - a COLUMN permutation by
            # hv-sized tiles, each tile a whole number of Q8_0 blocks, so
            # regrouping 4-block units inside every byte row is exact.
            cfg = self.config
            nv = cfg.linear_num_value_heads
            r = nv // nk
            head = np.arange(nv)
            tiled = (head % r) * nk + (head // r)
            groups = (k // 32) // (hv // 32)          # hv-sized groups/row
            assert groups == nv, (groups, nv, hv, k)
            rb = (k // 32) * 34
            gsz = (hv // 32) * 34
            view = slab.reshape(rows, groups, gsz)
            out = torch.from_numpy(
                np.ascontiguousarray(view.numpy()[:, tiled, :]).reshape(rows, rb)
            )
        else:  # pragma: no cover - _PACKABLE_Q80 is closed
            return None

        if _packed_verify_enabled():
            self._verify_packed(name, out, k)
        return out, k

    def _verify_packed(self, name: str, packed: torch.Tensor, k: int) -> None:
        """Cross-check gathered Q8_0 rows against the BF16 pin reference."""
        rows = packed.shape[0]
        rb = packed.shape[1]
        blocks = rb // 34
        v = packed.reshape(rows, blocks, 34).numpy()
        # numpy .view(dtype) only rewrites the LAST dim - reshape after.
        d = v[..., 0:2].copy().view(np.uint16).view(np.float16).reshape(rows, blocks)
        q = v[..., 2:34].copy().view(np.int8).reshape(rows, blocks, 32)
        deq = (
            torch.from_numpy(
                (q.astype(np.float32) * d[..., None].astype(np.float32))
            )
            .reshape(rows, k)
            .to(torch.bfloat16)
        )
        # _fetch, not get: the reference must not land in _cache - that
        # would pin ~4 GB of host BF16 for the process (the packed path
        # exists to not materialize those copies at all).
        want = self._fetch(name)
        if want is None or not torch.equal(deq, want.to(torch.bfloat16)):
            diff = (deq.float() - want.float()).abs().max().item()
            raise AssertionError(
                f"FTV_DENSE_Q80_VERIFY: {name} packed rows differ from the "
                f"dequant reference (maxdiff {diff:.3e})"
            )

    # -- experts ----------------------------------------------------------
    def expert_bank(self, layer: int, proj: str) -> ExpertBank:
        """Proxy bank for ``blk.{layer}.ffn_{proj}_exps`` (shape + packed lane)."""
        info = self._expert_info(layer, proj)
        if len(info.ne) != 3:
            raise ValueError(f"{info.name}: ne={info.ne} is not a 3-D expert bank")
        k, rows, experts = info.ne
        # K-quant shader selector: Q4_K/Q5_K/Q8_0 banks unpack in-shader,
        # anything else (or K not a multiple of 256) stays on the host path.
        qt = kquant_qtype(info.ggml_type) if is_quantized(info.ggml_type) else None
        if qt is not None and k % 256:
            qt = None  # K-quant superblocks span 256 elements; no shader lane
        rb = row_bytes(info.ggml_type, k) if qt else 0
        return ExpertBank(
            layer, proj, int(rows), int(k), int(experts),
            qtype=qt, row_bytes=rb,
        )

    def expert_packed(
        self, layer: int, proj: str, expert: int
    ) -> np.ndarray:
        """One expert slab's GGUF bytes verbatim, ``(rows, row_bytes)`` uint8.

        Zero-copy view into the mmap (same slab ``expert_bf16`` dequantizes);
        the caller uploads it as-is for the in-shader unpacker, so the host
        never materializes a float copy of the model.
        """
        return packed_expert(
            self.index, f"blk.{layer}.ffn_{proj}_exps.weight", expert
        )

    def expert_bf16(self, layer: int, proj: str, expert: int) -> torch.Tensor:
        """One expert slab dequantized to BF16 ``(rows, K)``, host-LRU cached."""
        info = self._expert_info(layer, proj)
        key = (info.root or info.path, info.name, expert, "bf16")
        if self._dcache is not None:
            return self._dcache.get_or_compute(
                key, lambda: dequant_expert(self.index, info, expert, dtype=torch.bfloat16)
            )
        return dequant_expert(self.index, info, expert, dtype=torch.bfloat16)

    def _expert_info(self, layer: int, proj: str) -> GgufTensorInfo:
        info = self.index.get(f"blk.{layer}.ffn_{proj}_exps.weight")
        if info is None:
            raise KeyError(f"blk.{layer} has no ffn_{proj}_exps bank")
        return info


def _dequant_rows(info: GgufTensorInfo, r0: int, r1: int) -> torch.Tensor:
    """Rows ``[r0, r1)`` of one 2-D GGUF tensor, dequantized to torch order."""
    k = int(info.ne[0])
    if is_quantized(info.ggml_type):
        rb = row_bytes(info.ggml_type, k)
        buf = np.frombuffer(
            info.data(), dtype=np.uint8, count=(r1 - r0) * rb, offset=r0 * rb
        ).reshape(r1 - r0, rb)
        floats = _np_dequantize(buf, GGMLQuantizationType(info.ggml_type))
    else:
        np_dt = torch.empty(0, dtype=_FLOAT_DTYPES[info.ggml_type]).numpy().dtype
        floats = np.frombuffer(
            info.data(), dtype=np_dt, count=(r1 - r0) * k,
            offset=r0 * k * np_dt.itemsize,
        ).reshape(r1 - r0, k)
    return torch.from_numpy(np.ascontiguousarray(floats)).view(r1 - r0, k).float()


__all__ = [
    "ExpertBank",
    "GgufWeightProvider",
    "gguf_config",
    "untile",
]
