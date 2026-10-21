"""Packed K-quant expert provider (freetoken/models/gguf/packed.py) gates.

The Vulkan shaders eat GGUF bytes verbatim; this module is what hands them
the bytes: one expert slab of (rows, row_bytes) uint8 straight out of the
mmap, with the orientation the GEMV kernels expect. The invariants that
matter are therefore byte-level, and they are checked against the existing
dequant path (which llama.cpp's own C dequantizers match bit-for-bit) so a
layout change can never slip through unnoticed.

No Vulkan device needed - these are CPU/mmap tests and always run.

Env: FREETOKEN_TEST_MODEL_GGUF = the CyberTiel GGUF (mmap only, never a
dequantized copy) enables the real-checkpoint assertions.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from freetoken.layers.quantization.ggml_types import is_quantized, row_bytes
from freetoken.models.gguf.packed import (
    PackedExpert,
    kquant_qtype,
    packed_expert,
    packed_expert_bytes,
    packed_expert_info,
    packed_gate_up_expert,
    packed_tensor,
)
from freetoken.models.gguf.reader import gguf_tensor_index
from freetoken.third_party.gguf import GGMLQuantizationType
from freetoken.third_party.gguf.quants import dequantize as np_dequant

GGUF = os.environ.get("FREETOKEN_TEST_MODEL_GGUF")
requires_gguf = pytest.mark.skipif(
    not GGUF or not os.path.isfile(GGUF),
    reason="FREETOKEN_TEST_MODEL_GGUF not set",
)

# the CyberTiel file's expert banks: Q5_K on blk.0-14, Q4_K on blk.15-40
Q5_LAYER, Q4_LAYER = 0, 15


def test_kquant_qtype_mapping():
    assert kquant_qtype(int(GGMLQuantizationType.Q4_K)) == "q4_k"
    assert kquant_qtype(int(GGMLQuantizationType.Q5_K)) == "q5_k"
    assert kquant_qtype(int(GGMLQuantizationType.Q8_0)) == "q8_0"
    # no in-shader unpacker for the rest: the provider must refuse, not
    # silently hand the shader something it cannot read
    assert kquant_qtype(int(GGMLQuantizationType.Q6_K)) is None
    assert kquant_qtype(int(GGMLQuantizationType.Q4_0)) is None
    assert kquant_qtype(int(GGMLQuantizationType.F16)) is None


@requires_gguf
@pytest.mark.parametrize("layer,qtype", [(Q5_LAYER, "q5_k"), (Q4_LAYER, "q4_k")])
def test_packed_expert_matches_dequant_input(layer, qtype):
    """packed_expert must be exactly the bytes dequant_expert consumes."""
    from freetoken.models.gguf.dequant import _expert_slab, dequant_expert

    index = gguf_tensor_index(GGUF)
    name = f"blk.{layer}.ffn_gate_exps.weight"
    info = index[name]
    meta = packed_expert_info(index, name, 7)
    assert meta.qtype == qtype, meta
    assert meta.rows == 512 and meta.cols == 2048, meta
    assert meta.aligned and meta.row_bytes % 4 == 0, meta
    assert meta.nbytes == meta.rows * meta.row_bytes

    slab = packed_expert(index, name, 7)
    assert slab.dtype == np.uint8
    assert slab.shape == (meta.rows, meta.row_bytes), slab.shape
    assert slab.nbytes == packed_expert_bytes(index, name, 7)

    # packed_expert and the host dequant path read the same slab, so the two
    # routes can never disagree about where an expert's bytes live
    assert meta.row_bytes == row_bytes(info.ggml_type, meta.cols)
    assert np.array_equal(slab, _expert_slab(info, 7))
    assert np.array_equal(
        np_dequant(slab, GGMLQuantizationType(info.ggml_type)),
        dequant_expert(index, name, 7, dtype=torch.float32).numpy())

    # row 0 of the slab is the dequant path's first block-group: n_blocks of
    # d/dmin + scales for a cols-wide Q*_K row
    n_blocks = meta.cols // 256
    assert meta.row_bytes == n_blocks * (144 if qtype == "q4_k" else 176)


@requires_gguf
def test_experts_are_distinct_and_neighbouring():
    """Two experts must be disjoint byte ranges of the bank (orientation bug
    would otherwise alias expert 0 onto expert 1)."""
    index = gguf_tensor_index(GGUF)
    name = "blk.0.ffn_gate_exps.weight"
    a = packed_expert(index, name, 0)
    b = packed_expert(index, name, 1)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)

    # expert 7 must start exactly 7 slabs into the bank
    base = np.frombuffer(index[name].data(), np.uint8)
    expert = packed_expert(index, name, 7)
    stride = expert.nbytes
    assert np.array_equal(base[7 * stride:8 * stride], expert.reshape(-1))


@requires_gguf
def test_packed_gate_up_stacks_gate_then_up():
    from freetoken.models.gguf.dequant import dequant_expert

    index = gguf_tensor_index(GGUF)
    gu = packed_gate_up_expert(index, Q5_LAYER, 3, model_path=GGUF)
    g = packed_expert(index, f"blk.{Q5_LAYER}.ffn_gate_exps.weight", 3)
    u = packed_expert(index, f"blk.{Q5_LAYER}.ffn_up_exps.weight", 3)
    assert gu.shape == (g.shape[0] + u.shape[0], g.shape[1]), gu.shape
    assert np.array_equal(gu[:g.shape[0]], g)
    assert np.array_equal(gu[g.shape[0]:], u)
    assert not np.array_equal(g, u)


@requires_gguf
def test_packed_tensor_is_upload_ready():
    index = gguf_tensor_index(GGUF)
    slab = packed_expert(index, "blk.15.ffn_gate_exps.weight", 11)
    t = packed_tensor(slab)
    assert t.dtype == torch_uint8() and t.is_contiguous()
    assert t.numel() == slab.size
    assert t.numpy().tobytes() == slab.tobytes()
    # writable: upload paths and arena packing must be able to mutate it
    t[0, 0] = t[0, 0]
    assert t.requires_grad is False


def torch_uint8():
    import torch
    return torch.uint8


@requires_gguf
def test_unsupported_type_raises():
    from freetoken.models.gguf.dequant import expert_bytes_packed  # noqa: F401

    index = gguf_tensor_index(GGUF)
    name = "token_embd.weight"
    with pytest.raises(ValueError, match="not a 3-D expert bank"):
        packed_expert_info(index, name, 0)


@requires_gguf
def test_packed_expert_out_of_range():
    index = gguf_tensor_index(GGUF)
    with pytest.raises(IndexError):
        packed_expert(index, f"blk.{Q4_LAYER}.ffn_gate_exps.weight", 256)


@requires_gguf
def test_non_quantized_bank_is_rejected():
    index = gguf_tensor_index(GGUF)
    name = "blk.0.ln_1.weight"
    if name not in index:
        pytest.skip(f"{name} not in this file")
    assert not is_quantized(index[name].ggml_type)
    with pytest.raises(ValueError, match="not quantized"):
        packed_expert_info(index, name, 0)


@requires_gguf
def test_every_expert_bank_row_bytes_aligns_to_4():
    """Every bank the shader will ever read must be uint32-indexable."""
    index = gguf_tensor_index(GGUF)
    checked = 0
    for name, info in index.items():
        if not is_quantized(info.ggml_type) or len(info.ne) != 3:
            continue
        qt = kquant_qtype(info.ggml_type)
        if qt is None:
            continue
        rb = row_bytes(info.ggml_type, info.ne[0])
        assert rb % 4 == 0, (name, qt, rb)
        checked += 1
    assert checked >= 3, f"only {checked} expert banks checked"  # gate/up/down
