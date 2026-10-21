"""GgufWeightProvider + gguf_config gates for the Vulkan CyberTiel path.

The provider is the weight-source layer ResidentGGUF runs on, so these
tests pin the three things that silently corrupt a decode if they drift:

1. ``gguf_config`` matches the file's ground truth (40 layers after the MTP
   block, full-attn every 4th layer, 256 experts top-8, GDN dims).
2. Name coverage: every translated name the decoder asks ``get()`` for is
   answered; MTP (blk.40) and dense-mlp names are NOT.
3. Transforms match upstream ``iter_gguf_weights`` (the read-only reference
   loader) for every float-sourced tensor - norms (-1.0), GDN untile on
   qkv-v/z/a/b, conv, A_log = log(-A), dt_bias - plus the ssm_out tiled
   column permutation, checked against the gdn.py reshape semantics by a
   different algebra (argsort of the head-index reshape vs the provider's
   head arithmetic).

Env: FREETOKEN_TEST_MODEL_GGUF = the CyberTiel GGUF (the same var
tests/vulkan model gates use). Metadata checks need the file (mmap header
only); the untile/ExpertBank unit tests always run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import dequant_matrix
from freetoken.models.gguf.reader import load_gguf_metadata
from freetoken.models.qwen3_5_moe.vulkan import gguf_provider as gp
from freetoken.models.qwen3_5_moe.vulkan.gguf_provider import (
    ExpertBank,
    GgufWeightProvider,
    gguf_config,
    untile,
)
from freetoken.third_party.gguf import GGMLQuantizationType, dequantize as np_dequantize

GGUF = os.environ.get("FREETOKEN_TEST_MODEL_GGUF")
requires_file = pytest.mark.skipif(
    not GGUF or not os.path.isfile(GGUF),
    reason="FREETOKEN_TEST_MODEL_GGUF not set",
)

FULL_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}
PROV_PREFIX = "model.language_model"


# -- always-run unit tests --------------------------------------------------
def test_untile_exact_permutation_small():
    # nk=2, head_dim=2, r=2: grouped (r,k,d) -> tiled (k,r,d) row indices.
    t = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    out = untile(t, 2, 2)
    assert out[:, 0].tolist() == [0, 1, 4, 5, 2, 3, 6, 7]


def test_untile_preserves_row_multiset():
    torch.manual_seed(0)
    t = torch.randn(16 * 2 * 8, 3)
    out = untile(t, 16, 8)
    assert out.shape == t.shape
    # Exact row-multiset equality (sums would differ in ulps: fp32 addition
    # is order-sensitive and untile reorders rows).
    io = torch.argsort(out[:, 0])
    it = torch.argsort(t[:, 0])
    assert torch.equal(out[io], t[it])


def test_expert_bank_contract():
    bank = ExpertBank(5, "gate", 512, 2048, 256)
    assert len(bank) == 256
    assert (bank.layer, bank.proj, bank.rows, bank.cols) == (5, "gate", 512, 2048)
    proxy, scales, gscale = bank[17]
    assert proxy.shape == (512, 2048)
    assert scales is None and gscale == 1.0


# -- file-gated gates -------------------------------------------------------
@requires_file
def test_gguf_config_matches_ground_truth():
    cfg = gguf_config(GGUF)
    assert cfg.num_hidden_layers == 40  # block_count 41 - 1 MTP
    assert len(cfg.layer_types) == 40
    for i in range(40):
        expect = "full_attention" if i in FULL_ATTN_LAYERS else "linear_attention"
        assert cfg.layer_types[i] == expect, f"layer {i}"
    assert cfg.hidden_size == 2048
    assert cfg.vocab_size == 248320
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 256
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 512
    assert cfg.shared_expert_intermediate_size == 512
    assert cfg.linear_num_key_heads == 16
    assert cfg.linear_num_value_heads == 32
    assert cfg.linear_key_head_dim == 128
    assert cfg.linear_value_head_dim == 128
    assert cfg.linear_conv_kernel_dim == 4
    assert abs(cfg.rms_norm_eps - 9.99999997475e-07) < 1e-12
    assert cfg.rope_theta == 1e7
    assert cfg.partial_rotary_factor == 64 / 256
    assert cfg.mrope_section == [11, 11, 10]


@requires_file
def test_name_coverage_matches_decoder_requests():
    prov = GgufWeightProvider(GGUF)
    cfg = prov.config
    asked: set[str] = set()
    for i in range(cfg.num_hidden_layers):
        base = f"{PROV_PREFIX}.layers.{i}"
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert_gate.weight",
        ):
            asked.add(f"{base}.{suffix}")
        if cfg.layer_types[i] == "linear_attention":
            for suffix in (
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.out_proj.weight",
                "linear_attn.conv1d.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                "linear_attn.norm.weight",
            ):
                asked.add(f"{base}.{suffix}")
        else:
            for suffix in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ):
                asked.add(f"{base}.{suffix}")
    asked.update({
        f"{PROV_PREFIX}.embed_tokens.weight",
        f"{PROV_PREFIX}.norm.weight",
        "lm_head.weight",
    })
    missing = sorted(n for n in asked if not prov.contains(n))
    assert not missing, f"provider cannot answer {missing[:5]} (+{max(0, len(missing)-5)})"
    # MTP block and dense-mlp names must NOT be served (41st block / no dense FFN).
    assert not prov.contains(f"{PROV_PREFIX}.layers.40.input_layernorm.weight")
    assert not prov.contains(f"{PROV_PREFIX}.layers.3.mlp.gate_proj.weight")
    # A full-attention layer has no GDN leaves and vice versa.
    assert not prov.contains(f"{PROV_PREFIX}.layers.0.self_attn.q_proj.weight")
    assert not prov.contains(f"{PROV_PREFIX}.layers.3.linear_attn.A_log")


@requires_file
def test_float_sourced_transforms_match_upstream_loader():
    """Provider vs upstream iter_gguf_weights: exact on fp32 sources, bf16
    on the fused float in-projections (upstream stores state as bf16)."""
    from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights

    upstream: dict[str, torch.Tensor] = {}
    for name, t in iter_gguf_weights(
        GGUF, torch.device("cpu"),
        include_moe_experts=False, include_non_moe=True, include_vision=False,
    ):
        upstream[name] = t
    prov = GgufWeightProvider(GGUF)

    def eq_fp32(provider_name: str, upstream_name: str) -> None:
        mine = prov.get(provider_name)
        theirs = upstream[upstream_name]
        assert mine.dtype == torch.float32, provider_name
        assert mine.shape == theirs.shape, provider_name
        assert torch.equal(mine, theirs.float()), provider_name

    def eq_bf16(provider_name: str, theirs: torch.Tensor) -> None:
        mine = prov.get(provider_name)
        assert mine.shape == theirs.shape, provider_name
        assert torch.equal(mine.to(torch.bfloat16), theirs), provider_name

    for i in (0, 3, 39):
        base = f"model.layers.{i}"
        pbase = f"{PROV_PREFIX}.layers.{i}"
        eq_fp32(f"{pbase}.input_layernorm.weight", f"{base}.input_layernorm.weight")
        eq_fp32(f"{pbase}.post_attention_layernorm.weight",
                f"{base}.post_attention_layernorm.weight")
        eq_fp32(f"{pbase}.mlp.gate.weight", f"{base}.mlp.gate.weight")
        eq_fp32(f"{pbase}.mlp.shared_expert_gate.weight",
                f"{base}.mlp.shared_expert_gate.weight")
        if i in FULL_ATTN_LAYERS:
            eq_fp32(f"{pbase}.self_attn.q_norm.weight", f"{base}.self_attn.q_norm.weight")
            eq_fp32(f"{pbase}.self_attn.k_norm.weight", f"{base}.self_attn.k_norm.weight")
        else:
            eq_fp32(f"{pbase}.linear_attn.conv1d.weight", f"{base}.linear_attn.conv1d.weight")
            eq_fp32(f"{pbase}.linear_attn.A_log", f"{base}.linear_attn.A_log")
            eq_fp32(f"{pbase}.linear_attn.dt_bias", f"{base}.linear_attn.dt_bias")
            eq_fp32(f"{pbase}.linear_attn.norm.weight", f"{base}.linear_attn.norm.weight")
            ba = upstream[f"{base}.linear_attn.in_proj_ba.weight"]  # [beta; alpha], bf16
            eq_bf16(f"{pbase}.linear_attn.in_proj_b.weight", ba[: ba.shape[0] // 2])
            eq_bf16(f"{pbase}.linear_attn.in_proj_a.weight", ba[ba.shape[0] // 2:])
            ours = torch.cat([
                prov.get(f"{pbase}.linear_attn.in_proj_b.weight"),
                prov.get(f"{pbase}.linear_attn.in_proj_a.weight"),
            ])
            assert torch.equal(ours.to(torch.bfloat16), ba)
    eq_fp32(f"{PROV_PREFIX}.norm.weight", "model.norm.weight")


@requires_file
def test_ssm_out_column_permutation_matches_gdn_reshape_semantics():
    """Provider's baked-in ssm_out column permutation == gdn.py's input
    permutation (out_heads_tiled), verified via the reshape/argsort path
    rather than the provider's head-index arithmetic."""
    prov = GgufWeightProvider(GGUF)
    cfg = prov.config
    nk, hv, nv = cfg.linear_num_key_heads, cfg.linear_value_head_dim, cfg.linear_num_value_heads
    r = nv // nk
    i = 0  # first GDN layer
    pbase = f"{PROV_PREFIX}.layers.{i}"
    grouped = prov.get(f"{pbase}.linear_attn.out_proj.weight")  # baked permutation
    raw = dequant_matrix(prov._t(f"blk.{i}.ssm_out"), dtype=torch.bfloat16)
    assert grouped.shape == raw.shape == (cfg.hidden_size, nv * hv)
    # gdn.py: out.view(total, nk, r, hv).transpose(1, 2) sends grouped head
    # j=(i_k, i_r) to tiled slot (i_r, i_k). perm[s] = grouped head index at
    # tiled slot s, from the reshape expression alone.
    perm = torch.arange(nv).view(nk, r).transpose(0, 1).reshape(-1)
    slot_of_head = torch.argsort(perm)  # grouped head j -> tiled slot
    cols = (slot_of_head[:, None] * hv + torch.arange(hv)).reshape(-1)
    assert torch.equal(grouped, raw.index_select(1, cols))
    # And the identity must NOT hold on the raw (unpermuted) matrix.
    assert not torch.equal(grouped, raw)


@requires_file
def test_kv_stay_fp16_native_and_shapes():
    prov = GgufWeightProvider(GGUF)
    cfg = prov.config
    pbase = f"{PROV_PREFIX}.layers.3"
    kv_rows = cfg.num_key_value_heads * cfg.head_dim  # 512
    for leaf in ("self_attn.k_proj", "self_attn.v_proj"):
        mine = prov.get(f"{pbase}.{leaf}.weight")
        assert mine.dtype == torch.float16
        assert mine.shape == (kv_rows, cfg.hidden_size)
        assert torch.isfinite(mine.float()).all()
    q = prov.get(f"{pbase}.self_attn.q_proj.weight")
    assert q.dtype == torch.bfloat16
    assert q.shape == (cfg.num_attention_heads * 2 * cfg.head_dim, cfg.hidden_size)
    o = prov.get(f"{pbase}.self_attn.o_proj.weight")
    assert o.dtype == torch.bfloat16
    assert o.shape == (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)


@requires_file
def test_embed_rows_lazy_and_bounded():
    prov = GgufWeightProvider(GGUF)
    rows = prov.get(f"{PROV_PREFIX}.embed_tokens.weight")
    assert rows.dtype == torch.float32
    assert len(rows) == 248320
    row = rows[151643]
    assert row.shape == (2048,)
    assert torch.isfinite(row).all() and row.abs().sum() > 0
    with pytest.raises(IndexError):
        rows[248320]
    # Row must equal the file's own dequant of that row (independent slicing).
    from freetoken.layers.quantization.ggml_types import row_bytes

    info = prov._t("token_embd")
    k = int(info.ne[0])
    rb = row_bytes(info.ggml_type, k)
    buf = np.frombuffer(info.data(), dtype=np.uint8, count=rb, offset=151643 * rb).reshape(1, rb)
    ref = torch.from_numpy(np.ascontiguousarray(
        np_dequantize(buf, GGMLQuantizationType(info.ggml_type))
    )).view(-1).float()
    assert torch.equal(row, ref)


@requires_file
def test_expert_banks_shapes_and_independent_slab_dequant():
    prov = GgufWeightProvider(GGUF)
    cfg = prov.config
    inter = cfg.moe_intermediate_size  # 512
    h = cfg.hidden_size  # 2048
    cases = ((0, "gate", inter, h), (0, "down", h, inter), (20, "up", inter, h))
    for layer, proj, rows, cols in cases:
        bank = prov.expert_bank(layer, proj)
        assert (bank.rows, bank.cols, len(bank)) == (rows, cols, 256)
    # Independent slab dequant: re-slice the expert bytes in the test and
    # feed ggml's dequantize directly (bypasses dequant.py's _expert_slab).
    for layer, proj, expert in ((0, "gate", 17), (0, "down", 255), (20, "up", 0)):
        info = prov._expert_info(layer, proj)
        k, n_rows, n_exp = info.ne
        from freetoken.layers.quantization.ggml_types import row_bytes
        rb = row_bytes(info.ggml_type, k)
        per = n_rows * rb
        buf = np.frombuffer(info.data(), dtype=np.uint8, count=per,
                            offset=expert * per).reshape(n_rows, rb)
        ref = torch.from_numpy(np.ascontiguousarray(
            np_dequantize(buf, GGMLQuantizationType(info.ggml_type))
        )).to(torch.bfloat16)
        mine = prov.expert_bf16(layer, proj, expert)
        assert mine.dtype == torch.bfloat16 and mine.shape == ref.shape
        assert torch.equal(mine, ref)
    # Host LRU: same expert twice returns the identical cached tensor.
    a = prov.expert_bf16(0, "gate", 17)
    b = prov.expert_bf16(0, "gate", 17)
    assert a is b


@requires_file
def test_dequant_matrix_native_dtype_reshapes():
    """Regression: non-quantized tensors at their native dtype used to come
    back flat (the dtype short-circuit skipped the reshape)."""
    prov = GgufWeightProvider(GGUF)
    info = prov._t("blk.3.attn_k")  # F16
    out = dequant_matrix(info, dtype=torch.float16)
    assert out.shape == tuple(reversed(info.ne)) == (512, 2048)
    assert out.dtype == torch.float16


@requires_file
def test_matrix_chunking_matches_full_dequant(monkeypatch):
    """Row-chunked _matrix == whole-tensor dequant_matrix, including on the
    chunk-size boundary (monkeypatched small so Q8_0 attn_q takes the path)."""
    prov = GgufWeightProvider(GGUF)
    info = prov._t("blk.3.attn_q")
    full = dequant_matrix(info, dtype=torch.bfloat16)
    monkeypatch.setattr(gp, "_CHUNK_BYTES", 1 << 20)
    monkeypatch.setattr(gp, "_CHUNK_ROWS", 4096)
    chunked = prov._matrix(info, torch.bfloat16)
    assert torch.equal(chunked, full)
    monkeypatch.setattr(gp, "_CHUNK_ROWS", 16384)
    assert torch.equal(prov._matrix(info, torch.bfloat16), full)


@requires_file
def test_lm_head_chunked_path_bit_consistent(monkeypatch):
    """Real >64 MB tensor: two chunk sizes must agree bit-exactly (row-offset
    math across the Q8_0 superblock boundary)."""
    prov = GgufWeightProvider(GGUF)
    info = prov._t("output")
    assert info.nbytes > gp._CHUNK_BYTES  # exercises the chunked path
    monkeypatch.setattr(gp, "_CHUNK_ROWS", 4096)
    a = prov._matrix(info, torch.bfloat16)
    monkeypatch.setattr(gp, "_CHUNK_ROWS", 32768)
    b = prov._matrix(info, torch.bfloat16)
    assert torch.equal(a, b)
    assert a.shape == (248320, 2048) and a.dtype == torch.bfloat16
    assert torch.isfinite(a.float()).all()


@requires_file
def test_metadata_shape_spot_checks():
    """Raw file facts the whole mapping rests on (no provider involved)."""
    md = load_gguf_metadata(GGUF)
    assert md["general.architecture"] == "qwen35moe"
    assert md["qwen35moe.block_count"] == 41
    assert md["qwen35moe.nextn_predict_layers"] == 1
    prov = GgufWeightProvider(GGUF)
    qkv = prov._t("blk.0.attn_qkv")
    assert tuple(qkv.ne) == (2048, 8192)
    assert prov._t("blk.0.attn_gate").ggml_type == int(GGMLQuantizationType.Q8_0)
    ssm_a = prov._t("blk.0.ssm_a")
    assert tuple(ssm_a.ne) == (32,)
