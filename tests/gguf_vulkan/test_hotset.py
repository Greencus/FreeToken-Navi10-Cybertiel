"""Static hot-set expert table gates (freetoken/models/gguf/hotset.py).

CPU/numpy only - no Vulkan device, no model load (byte-weighted checks
need FREETOKEN_TEST_MODEL_GGUF but stay metadata-only). The table module
is what the Vulkan decoder's hot-set residency pins from, so its
invariants are: deterministic ranking, compact npz round trip, honest
trace replay, and a shipped table that matches the CyberTiel shape.

Env:
    FREETOKEN_TEST_MODEL_GGUF  enables the GGUF-backed byte-size checks
    FTV_HOTSET_TEST_TRACES     optional dir with router_ids_{repeated,
                               unique,merged}.npy for real-trace replay
                               (default /tmp/opencode/perf_ceiling)
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from freetoken.models.gguf.hotset import (
    DEFAULT_TOP_K,
    TABLE_FORMAT_VERSION,
    HotsetTable,
    build_hotset,
    default_table_path,
    layer_expert_bytes_from_gguf,
    load_hotset,
    replay_hit_rate,
    save_hotset,
)

GGUF = os.environ.get("FREETOKEN_TEST_MODEL_GGUF")
requires_gguf = pytest.mark.skipif(
    not GGUF or not os.path.isfile(GGUF),
    reason="FREETOKEN_TEST_MODEL_GGUF not set",
)

TRACE_DIR = pathlib.Path(
    os.environ.get("FTV_HOTSET_TEST_TRACES", "/tmp/opencode/perf_ceiling"))


def _synthetic_trace(num_layers=4, tokens=50, topk=8, experts=32, seed=0):
    """Trace with a planted hot set: experts 0..3 carry 90% of traffic."""
    rng = np.random.default_rng(seed)
    hot = np.array([0, 1, 2, 3])
    cold = np.arange(4, experts)
    pick = rng.choice(hot, size=(num_layers, tokens, topk), p=[0.25] * 4)
    mask = rng.random((num_layers, tokens, topk)) < 0.10
    pick = np.where(mask, rng.choice(cold, size=pick.shape), pick)
    return pick.astype(np.int16)


def test_build_ranks_by_frequency():
    trace = _synthetic_trace()
    table = build_hotset(trace, top_k=4, source="synthetic")
    assert table.num_layers == 4 and table.top_k == 4
    assert table.num_experts == 32
    # Planted hot experts must rank first on every layer, by count.
    for layer in range(4):
        assert set(table.expert_ids[layer, :4].tolist()) == {0, 1, 2, 3}
        counts = table.counts[layer]
        assert counts[0] >= counts[1] >= counts[2] >= counts[3]
    # counts are the real selection counts of the ranked ids.
    flat0 = trace[0].reshape(-1)
    for slot, e in enumerate(table.expert_ids[0]):
        assert table.counts[0, slot] == int((flat0 == e).sum())


def test_build_is_deterministic():
    trace = _synthetic_trace(seed=1)
    a = build_hotset(trace, top_k=6)
    b = build_hotset(trace, top_k=6)
    assert np.array_equal(a.expert_ids, b.expert_ids)
    assert np.array_equal(a.counts, b.counts)


def test_npz_round_trip(tmp_path):
    trace = _synthetic_trace(seed=2)
    table = build_hotset(trace, top_k=8, source="round-trip",
                         layer_expert_bytes=[1000] * 4)
    path = str(tmp_path / "t.npz")
    save_hotset(table, path)
    back = load_hotset(path)
    assert np.array_equal(back.expert_ids, table.expert_ids)
    assert np.array_equal(back.counts, table.counts)
    assert back.num_experts == table.num_experts
    assert back.trace_tokens == table.trace_tokens
    assert back.trace_topk == table.trace_topk
    assert back.source == "round-trip"
    assert np.array_equal(back.layer_expert_bytes, table.layer_expert_bytes)
    assert back.trace_hit_rate == pytest.approx(table.trace_hit_rate)
    # compact: a top-24 x 256 table must stay in the KiB range
    big = build_hotset(
        np.random.default_rng(3).integers(0, 256, size=(40, 300, 8)),
        top_k=24)
    big_path = str(tmp_path / "big.npz")
    save_hotset(big, big_path)
    assert os.path.getsize(big_path) < 64 * 1024


def test_replay_static_covers_planted_hotset():
    trace = _synthetic_trace(tokens=200, seed=4)
    table = build_hotset(trace, top_k=4)
    rates = replay_hit_rate(table, trace)
    # static top-4 over the planted 4-hot-expert distribution (10% of
    # slots are uniform-cold): ~90% of requests hit
    assert rates["overall"] > 0.85
    assert len(rates["per_layer"]) == table.num_layers
    assert rates["requests"] == 4 * 200 * 8  # L x tokens x trace topk


def test_replay_byte_weighted_prefers_big_layers():
    trace = _synthetic_trace(seed=5)
    table = build_hotset(trace, top_k=4)
    uniform = replay_hit_rate(table, trace)
    skewed = replay_hit_rate(table, trace, layer_expert_bytes=[100, 1, 1, 1])
    # weighting layer 0 heavily must move the byte rate TOWARD layer 0's
    # own hit rate (never away from it)
    per = uniform["per_layer"]
    assert (skewed["byte_weighted"] - per[0]) * (uniform["overall"] - per[0]) >= 0
    assert abs(skewed["byte_weighted"] - per[0]) <= abs(
        uniform["overall"] - per[0]) + 1e-12


def test_replay_rejects_layer_mismatch():
    table = build_hotset(_synthetic_trace(), top_k=4)
    with pytest.raises(ValueError):
        replay_hit_rate(table, _synthetic_trace(num_layers=3))


def test_shipped_table_shape():
    path = default_table_path()
    assert os.path.isfile(path), f"shipped table missing: {path}"
    table = load_hotset(path)
    assert table.num_layers == 40
    assert table.num_experts == 256
    assert 1 <= table.top_k <= 256
    assert table.expert_ids.dtype == np.int16
    assert table.counts.shape == table.expert_ids.shape
    # ranked by descending count on every layer
    for layer in range(40):
        counts = table.counts[layer]
        assert (counts[:-1] >= counts[1:]).all()
        assert (table.expert_ids[layer] >= 0).all()
        assert (table.expert_ids[layer] < 256).all()


@requires_gguf
def test_layer_expert_bytes_metadata_only():
    lb = layer_expert_bytes_from_gguf(GGUF, 40)
    assert lb.shape == (40,)
    # Q5_K blocks (0-14) are larger than Q4_K (15-39) on the ICE mix
    assert lb[0] > lb[39] > 0
    assert lb.mean() < 4 * 1024 * 1024  # a few MiB per expert, never BF16


@pytest.mark.parametrize("name", ["merged", "repeated", "unique"])
def test_replay_real_traces(name):
    """Replay the shipped table on the real proxy-router traces when present."""
    trace_path = TRACE_DIR / f"router_ids_{name}.npy"
    if not trace_path.is_file():
        pytest.skip(f"no trace at {trace_path}")
    table = load_hotset(default_table_path())
    trace = np.load(trace_path)
    if trace.shape[0] != table.num_layers:
        pytest.skip(f"trace layers {trace.shape[0]} != table {table.num_layers}")
    layer_bytes = None
    if GGUF and os.path.isfile(GGUF):
        layer_bytes = layer_expert_bytes_from_gguf(GGUF, table.num_layers)
    rates = replay_hit_rate(table, trace, layer_expert_bytes=layer_bytes)
    assert 0.0 < rates["overall"] <= 1.0
    assert 0.0 < rates["byte_weighted"] <= 1.0
    print(f"\nreplay {name}: overall={rates['overall'] * 100:.1f}% "
          f"byte={rates['byte_weighted'] * 100:.1f}%")


def test_format_version_constant():
    assert TABLE_FORMAT_VERSION == 1
    assert DEFAULT_TOP_K == 24
    assert isinstance(HotsetTable, type)


@requires_gguf
@pytest.mark.parametrize("layer,expert", [(0, 7), (15, 3), (39, 255)])
def test_fused_gateup_layout_matches_per_matrix(layer, expert):
    """The batched-decode fused bundle must be byte-identical to the
    per-matrix packed slabs the sequential path uploads: gate rows first,
    up rows after, zero re-quantization in between. This pins the layout
    contract the hot-set pin (`_pin_fused_gateup`) relies on, so the
    pinned buffer cannot drift from what the engine would have streamed.
    """
    from freetoken.models.gguf.packed import (
        gguf_tensor_index,
        packed_expert,
        packed_gate_up_expert,
    )

    index = gguf_tensor_index(GGUF)
    fused = packed_gate_up_expert(index, layer, expert)
    gate = packed_expert(index, f"blk.{layer}.ffn_gate_exps.weight", expert)
    up = packed_expert(index, f"blk.{layer}.ffn_up_exps.weight", expert)
    assert fused.shape[0] == gate.shape[0] + up.shape[0]
    assert fused.dtype == gate.dtype == up.dtype == np.uint8
    # byte-for-byte, halves separately (cheap full-array checks at ~1 MiB)
    assert np.array_equal(fused[: gate.shape[0]], gate)
    assert np.array_equal(fused[gate.shape[0]:], up)
    print(f"\nfused layout layer={layer} expert={expert}: "
          f"{fused.shape[0]} rows == gate {gate.shape[0]} + up {up.shape[0]}")
