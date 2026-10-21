"""Dequant round-trip and structural checks for models/gguf/dequant.py.

Q8_0 is exact (values are integer multiples of the stored fp16 scale), so it gets a
byte-exact comparison against an independent scalar unpacker; Q4_K/Q5_K get block
layout + scale-reconstruction checks. The synthetic tensors live in tiny real GGUF
files (written by the test, read back through the actual mmap reader); pointing
FREETOKEN_TEST_MODEL_GGUF at the real file adds per-expert slicing checks on real
Q5_K / Q4_K / Q8_0 slabs.
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import (
    DequantCache,
    dequant_expert,
    dequant_matrix,
    expert_bytes_packed,
    expert_count,
    torch_shape,
)
from freetoken.third_party.gguf import GGML_QUANT_SIZES, GGMLQuantizationType as T

GGUF = os.environ.get("FREETOKEN_TEST_MODEL_GGUF")
requires_file = pytest.mark.skipif(not GGUF or not os.path.isfile(GGUF), reason="FREETOKEN_TEST_MODEL_GGUF not set")


def write_gguf(path, tensors: dict[str, tuple[bytes, tuple[int, ...], int]]) -> None:
    """A minimal GGUF v3: magic/version/counts, tensor infos, 32-aligned data section.

    ``tensors`` maps name -> (packed bytes, ggml ``ne``, ggml type id).
    """
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), 0)
    infos = bytearray()
    blobs: list[tuple[int, bytes]] = []
    offset = 0
    for name, (payload, ne, qtype) in tensors.items():
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb
        infos += struct.pack("<I", len(ne))
        infos += struct.pack("<" + "Q" * len(ne), *ne)
        infos += struct.pack("<IQ", qtype, offset)
        pad = (-offset) % 32
        offset += pad
        blobs.append((offset, payload))
        offset += len(payload)
    start = len(header) + len(infos)
    start = (-start) % 32 + start  # data section 32-aligned, as general.alignment defaults
    out = bytearray(header + bytes(infos) + b"\0" * (start - len(header) - len(infos)))
    for off, payload in blobs:
        out += b"\0" * (off + start - len(out)) + payload
    with open(path, "wb") as f:
        f.write(out)


def _index_of(tmp_path, tensors):
    from freetoken.models.gguf.reader import gguf_tensor_index

    p = str(tmp_path / "synth.gguf")
    write_gguf(p, tensors)
    return p, gguf_tensor_index(p)


# ---------------------------------------------------------------- Q8_0: exact

def _pack_q8_0(rows: np.ndarray) -> bytes:
    """Reference Q8_0 packer: d = amax/127 per block of 32, qs = round(x/d)."""
    out = bytearray()
    for row in rows.astype(np.float32):
        for start in range(0, row.size, 32):
            blk = row[start : start + 32]
            d = float(np.max(np.abs(blk))) / 127.0
            qs = np.clip(np.round(blk / d), -127, 127).astype(np.int8)
            out += struct.pack("<e", np.float16(d)) + qs.tobytes()
    return bytes(out)


def _unpack_q8_0_scalar(packed: np.ndarray, k: int) -> np.ndarray:
    """Independent scalar Q8_0 unpacker (no quants.py): d from 2 bytes, qs int8."""
    rows = []
    for row in packed.reshape(-1, packed.shape[-1]):
        vals = np.empty((row.size // 34, 32), dtype=np.float32)
        for b in range(row.size // 34):
            (d,) = struct.unpack_from("<e", row, b * 34)
            qs = np.frombuffer(row, dtype=np.int8, count=32, offset=b * 34 + 2)
            vals[b] = np.float32(d) * qs
        rows.append(vals.reshape(k))
    return np.stack(rows)


def test_q8_0_roundtrip_exact_against_scalar_unpacker(tmp_path):
    rng = np.random.default_rng(0)
    rows = rng.normal(0, 0.05, (7, 256)).astype(np.float32)
    packed = np.frombuffer(_pack_q8_0(rows), dtype=np.uint8).reshape(7, 272)
    _, index = _index_of(tmp_path, {"synth.q8": (packed.tobytes(), (256, 7), int(T.Q8_0))})
    got = dequant_matrix(index["synth.q8"], dtype=torch.float32).numpy()
    want = _unpack_q8_0_scalar(packed, 256)
    assert np.array_equal(got, want), f"max abs diff {np.abs(got - want).max()}"


def test_q8_0_real_tensor_exact():
    if not GGUF:
        pytest.skip("FREETOKEN_TEST_MODEL_GGUF not set")
    from freetoken.models.gguf.reader import gguf_tensor_index

    index = gguf_tensor_index(GGUF)
    info = index["blk.0.ffn_gate_shexp.weight"]  # Q8_0, 512x2048
    assert info.ggml_type == int(T.Q8_0)
    got = dequant_matrix(info, dtype=torch.float32)
    assert got.shape == (512, 2048)
    packed = np.frombuffer(info.data(), dtype=np.uint8).reshape(512, 2048 // 32 * 34)
    # a sample of rows against the independent scalar unpacker (all 512 would work too)
    rows = np.array([0, 1, 137, 511])
    want = _unpack_q8_0_scalar(packed[rows], 2048)
    assert np.array_equal(got[rows].numpy(), want)


# ------------------------------------------------------- Q4_K / Q5_K structure

def test_q4_k_block_layout_and_scale_reconstruction(tmp_path):
    rng = np.random.default_rng(1)
    n_blocks = 5
    scale_bytes = rng.integers(0, 64, (n_blocks, 12), dtype=np.uint8)
    qs_bytes = rng.integers(0, 256, (n_blocks, 128), dtype=np.uint8)
    ds = rng.uniform(0.001, 0.02, n_blocks).astype(np.float16)
    dmins = rng.uniform(0.0, 0.005, n_blocks).astype(np.float16)
    packed = bytearray()
    for b in range(n_blocks):
        packed += ds[b].tobytes() + dmins[b].tobytes() + scale_bytes[b].tobytes() + qs_bytes[b].tobytes()
    assert len(packed) == n_blocks * 144  # Q4_K type size 144
    _, index = _index_of(tmp_path, {"synth.q4k": (bytes(packed), (256, n_blocks), int(T.Q4_K))})
    out = dequant_matrix(index["synth.q4k"], dtype=torch.float32).numpy()
    assert out.shape == (n_blocks, 256)
    assert np.isfinite(out).all()
    # every value is d*sc*q - dmin*m with q in [0,15]: bound by the block's max scale path
    from freetoken.third_party.gguf.quants import Q4_K as _Q4K

    sc, m = _Q4K.get_scale_min(scale_bytes)
    bound = ds.astype(np.float32)[:, None] * sc.max(axis=1)[:, None] * 15 + dmins.astype(np.float32)[:, None] * m.max(axis=1)[:, None]
    assert np.all(np.abs(out) <= bound + 1e-6)
    # at most 8 distinct scales x 16 codes (+ min shifts) per block -> bounded lattice
    per_block_unique = [len(np.unique(np.round(out[b], decimals=6))) for b in range(n_blocks)]
    assert all(u <= 8 * 16 + 8 for u in per_block_unique)


def test_quant_block_sizes():
    assert GGML_QUANT_SIZES[T.Q5_K] == (256, 176)
    assert GGML_QUANT_SIZES[T.Q4_K] == (256, 144)
    assert GGML_QUANT_SIZES[T.Q8_0] == (32, 34)


@pytest.mark.parametrize("qtype", [T.Q4_K, T.Q5_K, T.Q8_0])
def test_expert_slicing_matches_whole_dequant(tmp_path, qtype):
    from freetoken.models.gguf.dequant import _expert_slab

    block, size = GGML_QUANT_SIZES[qtype]
    experts, rows, k = 4, 3, 256
    row_b = k // block * size
    rng = np.random.default_rng(int(qtype))
    buf = bytearray()
    for _e in range(experts):
        for _r in range(rows):
            if qtype == T.Q8_0:
                buf += _pack_q8_0(rng.normal(0, 0.05, (1, k)))
            else:
                # valid K-quant superblock: small fp16 d/dmin, 6-bit scales, random codes
                d = np.float16(rng.uniform(0.001, 0.02))
                dmin = np.float16(rng.uniform(0.0, 0.005))
                scales = rng.integers(0, 64, 12, dtype=np.uint8).tobytes()
                qs = rng.integers(0, 256, size - 16, dtype=np.uint8).tobytes()
                buf += d.tobytes() + dmin.tobytes() + scales + qs
    assert len(buf) == experts * rows * row_b
    _, index = _index_of(tmp_path, {"synth.exps": (bytes(buf), (k, rows, experts), int(qtype))})
    info = index["synth.exps"]
    whole = dequant_matrix(info, dtype=torch.float32)
    assert whole.shape == (expert_count(info), rows, k)
    assert torch_shape(info) == (experts, rows, k)
    for e in range(expert_count(info)):
        sl = dequant_expert(index, "synth.exps", e, dtype=torch.float32)
        assert sl.shape == (rows, k)
        assert torch.equal(sl, whole[e]), f"expert {e} slice != whole[e] for {qtype}"
        assert _expert_slab(info, e).nbytes == expert_bytes_packed(info)


def test_expert_index_error_and_rank_check(tmp_path):
    _, index = _index_of(tmp_path, {"b2": (os.urandom(2 * 32 * 34), (256, 2), int(T.Q8_0))})
    info = index["b2"]  # 2-D, not a bank
    with pytest.raises(ValueError):
        dequant_expert(index, "b2", 0)
    with pytest.raises(ValueError):
        expert_count(info)


# ---------------------------------------------------------------- cache

def test_cache_lru_eviction_and_budget():
    cache = DequantCache(budget_mb=1)  # 1 MiB
    a = torch.zeros(1024, 32)          # 128 KiB each
    calls = []

    def make(tag):
        calls.append(tag)
        return a

    k1 = cache.get_or_compute("a", lambda: make("a"))
    assert cache.get_or_compute("a", lambda: make("a2")) is k1
    assert calls == ["a"]                                   # hit, no recompute
    for i in range(30):                                     # 30 * 128 KiB > 1 MiB -> evictions
        cache.get_or_compute(i, lambda i=i: make(i))
    assert cache.used_bytes <= 1024 * 1024
    assert len(cache) < 31
    cache.clear()
    assert cache.used_bytes == 0 and len(cache) == 0


def test_cache_from_env(monkeypatch):
    monkeypatch.setenv("FTV_GGUF_DEQUANT_CACHE_MB", "0")
    assert DequantCache.from_env() is None
    monkeypatch.setenv("FTV_GGUF_DEQUANT_CACHE_MB", "64")
    c = DequantCache.from_env()
    assert c is not None and c.budget_bytes == 64 * 2**20
    monkeypatch.delenv("FTV_GGUF_DEQUANT_CACHE_MB")
    c = DequantCache.from_env()
    assert c is not None and c.budget_bytes == 256 * 2**20
    monkeypatch.setenv("FTV_GGUF_DEQUANT_CACHE_MB", "not-a-number")
    with pytest.raises(ValueError):
        DequantCache.from_env()


# ------------------------------------------------- real-file expert statistics

@requires_file
def test_real_expert_dequant_stats_match_scale_fields():
    """Q5_K blk.0 and Q4_K blk.20 slabs: |x| <= d*max(sc)*31 + dmin*max(m)*63 per row."""
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.third_party.gguf.quants import Q4_K as _Q4K

    index = gguf_tensor_index(GGUF)
    # gate_exps ne=(2048,512,256) -> per-expert (rows=512, K=2048); down ne=(512,2048,256) -> (2048, 512)
    for name, want_type, want_shape in (
        ("blk.0.ffn_gate_exps.weight", T.Q5_K, (512, 2048)),
        ("blk.20.ffn_down_exps.weight", T.Q4_K, (2048, 512)),
    ):
        info = index[name]
        assert info.ggml_type == int(want_type), name
        slab = dequant_expert(index, name, 7, dtype=torch.float32).numpy()
        rows, k = want_shape
        assert slab.shape == want_shape
        assert np.isfinite(slab).all()
        row_b = expert_bytes_packed(info) // rows
        raw = np.frombuffer(info.data(), dtype=np.uint8, count=rows * row_b, offset=7 * expert_bytes_packed(info)).reshape(rows, row_b)
        d = raw[:, 0:2].view(np.float16).astype(np.float32)[:, 0]
        dmin = raw[:, 2:4].view(np.float16).astype(np.float32)[:, 0]
        sc, m = _Q4K.get_scale_min(raw[:, 4:16])
        bound = d[:, None] * sc.max(axis=1)[:, None] * 31 + dmin[:, None] * m.max(axis=1)[:, None]
        assert np.all(np.abs(slab) <= bound + 1e-6)
        # calibrated checkpoint weights are small: magnitude sanity
        assert np.abs(slab).mean() < 1.0


@requires_file
def test_real_q8_0_dense_roundtrip():
    from freetoken.models.gguf.reader import gguf_tensor_index

    index = gguf_tensor_index(GGUF)
    info = index["blk.0.attn_qkv.weight"]
    assert info.ggml_type == int(T.Q8_0)
    rows = dequant_matrix(info, dtype=torch.float32)
    assert rows.shape == (8192, 2048)
    packed = np.frombuffer(info.data(), dtype=np.uint8).reshape(8192, 2048 // 32 * 34)
    sample = np.array([0, 4096, 8191])
    assert np.array_equal(rows[sample].numpy(), _unpack_q8_0_scalar(packed[sample], 2048))
