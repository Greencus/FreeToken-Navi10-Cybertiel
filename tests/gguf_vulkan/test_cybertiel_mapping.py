"""Mapping checks for the CyberTiel qwen35moe GGUF against models/gguf/arch_qwen35moe.py.

Run with FREETOKEN_TEST_MODEL_GGUF pointing at the real 22.8 GB file (mmap only; the
test never copies or fully materializes it). Without the env var the real-file tests
skip; the synthetic-metadata tests always run.
"""

from __future__ import annotations

import os

import pytest

from freetoken.models.gguf.arch_qwen35moe import (
    ARCH,
    ExpectedTensor,
    expected_tensors,
    expert_quant_agreement,
    layer_types,
    mapping_summary,
    validate,
)

GGUF = os.environ.get("FREETOKEN_TEST_MODEL_GGUF")
requires_file = pytest.mark.skipif(not GGUF or not os.path.isfile(GGUF), reason="FREETOKEN_TEST_MODEL_GGUF not set")

# The measured CyberTiel metadata subset the mapping rules consume.
CYBERTIEL_MD = {
    "general.architecture": ARCH,
    "qwen35moe.block_count": 41,
    "qwen35moe.nextn_predict_layers": 1,
    "qwen35moe.context_length": 262144,
    "qwen35moe.embedding_length": 2048,
    "qwen35moe.attention.head_count": 16,
    "qwen35moe.attention.head_count_kv": 2,
    "qwen35moe.attention.key_length": 256,
    "qwen35moe.attention.value_length": 256,
    "qwen35moe.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
    "qwen35moe.expert_count": 256,
    "qwen35moe.expert_used_count": 8,
    "qwen35moe.expert_feed_forward_length": 512,
    "qwen35moe.expert_shared_feed_forward_length": 512,
    "qwen35moe.full_attention_interval": 4,
    "qwen35moe.rope.dimension_count": 64,
    "qwen35moe.rope.dimension_sections": [11, 11, 10, 0],
    "qwen35moe.rope.freq_base": 10000000.0,
    "qwen35moe.ssm.conv_kernel": 4,
    "qwen35moe.ssm.state_size": 128,
    "qwen35moe.ssm.group_count": 16,
    "qwen35moe.ssm.time_step_rank": 32,
    "qwen35moe.ssm.inner_size": 4096,
}


def test_layer_typing_rule_matches_file_layout():
    types = layer_types(CYBERTIEL_MD)
    assert len(types) == 40
    assert types.count("full_attention") == 10
    assert types.count("linear_attention") == 30
    # full attention every 4th layer, 0-indexed: blk.3, 7, ..., 39
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    assert full == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def test_layer_typing_recurrent_mask_overrides_interval():
    md = dict(CYBERTIEL_MD)
    # recurrent_layers bitmask: 1 = linear (GDN), 0 = full attention
    mask = [1, 0] + [1] * 38  # only blk.1 full
    mask[3] = 1               # blk.3: interval rule says full, mask keeps it linear
    md[f"{ARCH}.attention.recurrent_layers"] = mask
    types = layer_types(md)
    assert types[1] == "full_attention" and types[0] == "linear_attention"
    # the mask is authoritative over full_attention_interval
    assert types[3] == "linear_attention"


def test_expected_tensor_counts_and_mtp_rule():
    exp = expected_tensors(CYBERTIEL_MD)
    # 40 decoder layers * (10 GDN + 8 MoE-ish shared/router + ... ) counted by role:
    decoder = [e for e in exp.values() if not e.name.startswith("blk.40.")]
    mtp = [e for e in exp.values() if e.name.startswith("blk.40.")]
    # blk.0..39: 30 GDN layers * 19 tensors + 10 full-attn layers * 16 tensors + 3 top-level
    assert len(decoder) == 30 * 19 + 10 * 16 + 3
    # MTP block: a full-attention block's 16 tensors + 4 nextn head tensors
    assert len(mtp) == 20
    assert all(e.role == "mtp" for e in mtp)
    assert sum(1 for e in mtp if ".nextn." in e.name) == 4
    # top-level roles
    assert exp["token_embd.weight"].role == "embed"
    assert exp["output.weight"].role == "lm_head"
    assert exp["output_norm.weight"].role == "norm"


def test_expected_shapes_for_gdn_and_attention():
    exp = expected_tensors(CYBERTIEL_MD)
    # GDN layer 0: qkv 6144 rows (q 2048 + k 2048 + v 2048? no: qkv = 2*nk*dk + nv*dv = 4096+4096)
    assert exp["blk.0.attn_qkv.weight"].ne == (2048, 8192)
    assert exp["blk.0.attn_gate.weight"].ne == (2048, 4096)
    assert exp["blk.0.ssm_conv1d.weight"].ne == (4, 8192)
    assert exp["blk.0.ssm_out.weight"].ne == (4096, 2048)
    # full-attention layer 3: q packs query+gate (num_q*head_dim*2 = 8192 rows)
    assert exp["blk.3.attn_q.weight"].ne == (2048, 8192)
    assert exp["blk.3.attn_k.weight"].ne == (2048, 512)
    assert exp["blk.3.attn_output.weight"].ne == (4096, 2048)
    # experts: 3-D, 256 experts
    assert exp["blk.0.ffn_gate_exps.weight"].ne == (2048, 512, 256)
    assert exp["blk.0.ffn_down_exps.weight"].ne == (512, 2048, 256)


@requires_file
def test_real_file_validates_clean():
    report = validate(GGUF)
    assert report.problems == [], report.problems
    assert report.missing == [], report.missing
    assert report.unexpected == [], report.unexpected
    assert report.type_mismatch == [], report.type_mismatch
    assert report.shape_mismatch == [], report.shape_mismatch
    assert report.ok


@requires_file
def test_real_file_arch_layers_experts():
    summary = mapping_summary(GGUF)
    assert summary["arch"] == ARCH
    assert summary["block_count"] == 41
    assert summary["num_layers"] == 40
    assert summary["nextn_predict_layers"] == 1
    assert summary["layer_types"].count("full_attention") == 10
    assert summary["layer_types"].count("linear_attention") == 30
    assert summary["experts"] == 256
    assert summary["experts_per_tok"] == 8
    assert summary["hidden"] == 2048
    assert summary["vocab"] == 248320
    assert summary["tied"] is False
    assert summary["tensor_count"] == 753


@requires_file
def test_real_file_expert_quant_mix():
    report = validate(GGUF)
    quants = report.expert_quants
    assert len(quants) == 41  # blk.0..40 all carry expert banks
    # ICE mix: 15 Q5_K layers at the bottom, everything above Q4_K; gate==up==down everywhere
    assert all(quants[b] == ("Q5_K", "Q5_K", "Q5_K") for b in range(0, 15))
    assert all(quants[b] == ("Q4_K", "Q4_K", "Q4_K") for b in range(15, 41))
    # decoder layers only (blk.0..39) = 15 Q5_K + 25 Q4_K, matching expert-bank tags
    decoder = [quants[b] for b in range(40)]
    assert decoder.count(("Q5_K",) * 3) == 15
    assert decoder.count(("Q4_K",) * 3) == 25


@requires_file
def test_real_file_mtp_tensors_are_the_only_extras():
    report = validate(GGUF)
    # everything present beyond the decoder expectation sits in blk.40 (the MTP block)
    assert report.mtp_present, "expected the MTP block's tensors to be listed as ignored extras"
    assert all(n.startswith("blk.40.") for n in report.mtp_present)
    assert len(report.mtp_present) == 20
    assert {n for n in report.mtp_present if ".nextn." in n} == {
        "blk.40.nextn.eh_proj.weight",
        "blk.40.nextn.enorm.weight",
        "blk.40.nextn.hnorm.weight",
        "blk.40.nextn.shared_head_norm.weight",
    }


@requires_file
def test_real_file_shapes_shared_expert_and_router():
    from freetoken.models.gguf.reader import gguf_tensor_index

    index = gguf_tensor_index(GGUF)
    assert index["blk.0.ffn_gate_inp.weight"].ne == (2048, 256)      # router: hidden x experts
    assert index["blk.0.ffn_gate_inp_shexp.weight"].ne == (2048,)    # shared-expert gate
    assert index["blk.0.ffn_gate_shexp.weight"].ne == (2048, 512)
    assert index["blk.0.ffn_down_shexp.weight"].ne == (512, 2048)


def test_expert_agreement_detects_mismatch():
    class Fake:
        def __init__(self, ggml_type):
            self.ggml_type = ggml_type

    from freetoken.third_party.gguf import GGMLQuantizationType as T

    index = {
        "blk.0.ffn_gate_exps.weight": Fake(int(T.Q5_K)),
        "blk.0.ffn_up_exps.weight": Fake(int(T.Q5_K)),
        "blk.0.ffn_down_exps.weight": Fake(int(T.Q4_K)),
    }
    out, problems = expert_quant_agreement(index)  # type: ignore[arg-type]
    assert out[0] == ("Q5_K", "Q5_K", "Q4_K")
    assert any("gate experts Q5_K != down experts Q4_K" in p for p in problems)
