"""Declarative mapping of GGUF arch ``qwen35moe`` onto FreeToken's Qwen3.5-MoE family.

The mapping was derived from a real checkpoint (Ornith-1.5-35B-A3B "CyberTiel
Calibrated MTPv2 ICE", 22.8 GB, ``general.file_type = 15`` = MOSTLY_Q4_K_M) and is
validated by :func:`validate` against the file's own tensor index and metadata. It
mirrors what ``freetoken.models.qwen3_5_moe.gguf`` actually reads (``_LAYER_LEAVES``,
``parse_gguf_config``) so the Vulkan path - which cannot run CUDA ggml kernels and
needs plain torch tensors - can map, check and dequantize a file without loading the
engine.

Layout facts this table encodes (all measured from the file, ggml ``ne`` order,
fastest dim first):

- 40 decoder layers + trailing MTP block(s): ``block_count - nextn_predict_layers``;
  blocks ``blk.N`` with ``N >= num_layers`` are excluded from the run (their
  tensors are listed by the file but never loaded; role ``mtp``).
- Layer typing: ``full_attention`` when ``(i + 1) % full_attention_interval == 0``
  else ``linear_attention`` (GDN). A file may instead ship an explicit
  ``attention.recurrent_layers`` bitmask; both rules are implemented.
- Full-attention layers store ``attn_q``/``attn_k``/``attn_v``/``attn_output``;
  GDN layers store fused ``attn_qkv`` + ``attn_gate`` and the ``ssm_*`` tensors.
- MoE: every layer has a F32 router (``ffn_gate_inp``), a F32 shared-expert gate
  (``ffn_gate_inp_shexp``), Q8_0 shared-expert matrices, and three 3-D per-expert
  banks ``ffn_{gate,up,down}_exps`` with ``expert_count`` experts. The per-layer
  expert quant type is uniform across gate/up/down in this file (Q5_K in blk.0-14,
  Q4_K in blk.15-40); upstream enforces gate==up, this module additionally checks
  down==gate when ``strict``.
- ``ssm_a`` is stored as ``-exp(A_log)`` and zero-centred norms as ``w + 1``; the
  loader undoes both on read (see ``qwen3_5_moe/gguf.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .reader import GgufTensorInfo, gguf_tensor_index, load_gguf_metadata
from .tensors import check_scales

ARCH = "qwen35moe"

# Metadata keys the mapping needs, with what they mean and what the loader does with them.
REQUIRED_METADATA: dict[str, str] = {
    "block_count": "decoder blocks INCLUDING trailing MTP block(s); num_layers = block_count - nextn_predict_layers",
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "attention.head_count": "num_attention_heads (Q heads of the full-attention layers)",
    "attention.head_count_kv": "num_key_value_heads (GDN k-head count, group_count-scaled)",
    "attention.key_length": "head_dim of the full-attention layers",
    "attention.value_length": "value head dim (== key_length here)",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "expert_count": "routed experts per MoE layer (3-D bank dim 3)",
    "expert_used_count": "experts selected per token (top-k)",
    "expert_feed_forward_length": "moe_intermediate_size (expert matrix output rows)",
    "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
    "full_attention_interval": "layer i is full attention iff (i+1) % interval == 0 (when no recurrent_layers mask)",
    "rope.dimension_count": "partial rotary factor numerator (dimension_count / key_length)",
    "rope.dimension_sections": "mrope sections [q,k,v,?] - first three feed mrope_section",
    "rope.freq_base": "rope_theta",
    "ssm.conv_kernel": "linear_conv_kernel_dim (GDN depthwise conv width)",
    "ssm.state_size": "linear_key_head_dim (GDN k head dim)",
    "ssm.group_count": "linear_num_key_heads (GDN k heads)",
    "ssm.time_step_rank": "linear_num_value_heads (GDN v heads / dt rank)",
    "ssm.inner_size": "GDN value inner dim; value_head_dim = inner_size / time_step_rank",
}

# ggml type names this arch's roles may carry. "quant" = any quantized ggml type the
# upstream reader accepts; the per-file profile (CYBERTIEL_EXPECTED) pins the exact
# types a specific file uses.
ROLE_QUANTS: dict[str, tuple[str, ...]] = {
    "norm": ("F32",),          # zero-centred: stored as w + 1, loader subtracts 1
    "router": ("F32",),        # ffn_gate_inp, ffn_gate_inp_shexp
    "gdn_float": ("F32",),     # ssm_alpha/beta/a/dt.bias/conv1d/norm
    "attn_k_v": ("F16", "F32", "BF16", "quant"),  # full-attention k/v (F16 in CyberTiel)
    "matrix": ("quant", "F16", "F32", "BF16"),    # everything that goes through a Linear
    "expert_bank": ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "quant"),  # 3-D per-expert banks
}

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class ExpectedTensor:
    """One tensor the file must contain, with its role and the shape (ggml ``ne``) it must have."""

    name: str
    role: str                       # embed | lm_head | norm | router | shared_expert | gdn | attention | experts | mtp
    freetoken: str | None           # FreeTensor state-dict leaf the loader produces (None: bank / mtp / fused part)
    ne: tuple[int, ...]             # expected ggml shape (fastest dim first); () = not checked
    quants: tuple[str, ...] = ()    # accepted ggml type names ("" entry = any)


@dataclass
class MappingReport:
    """Expected-vs-present of one file: what is missing, extra, or typed wrong."""

    arch: str
    num_layers: int
    layer_types: list[str]
    mtp_blocks: list[int]
    expert_quants: dict[int, tuple[str, str, str]]   # block -> (gate, up, down) ggml type names
    expected: dict[str, ExpectedTensor]
    present: set[str]
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)   # present but not expected AND not mtp-ignored
    mtp_present: list[str] = field(default_factory=list)  # present, in an excluded MTP block
    type_mismatch: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)     # structural refusals (gate/up disagree, unread scales...)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.type_mismatch or self.shape_mismatch or self.problems)


def layer_types(metadata: dict[str, Any], num_layers: int | None = None) -> list[str]:
    """``linear_attention`` / ``full_attention`` per decoder layer, same rules as the loader."""
    if num_layers is None:
        num_layers = int(metadata[f"{ARCH}.block_count"]) - int(metadata.get(f"{ARCH}.nextn_predict_layers", 0))
    recurrent = metadata.get(f"{ARCH}.attention.recurrent_layers")
    if recurrent is not None:
        return ["linear_attention" if r else "full_attention" for r in list(recurrent)[:num_layers]]
    interval = int(metadata[f"{ARCH}.full_attention_interval"])
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(num_layers)]


def _kv(metadata: dict[str, Any], key: str) -> Any:
    return metadata[f"{ARCH}.{key}"]


def expected_tensors(metadata: dict[str, Any]) -> dict[str, ExpectedTensor]:
    """Every tensor the file must contain for the decoder, keyed by GGUF name.

    MTP blocks (``blk.N`` for ``N >= num_layers``) are expected to be PRESENT but
    excluded from the run; they are listed here with role ``mtp`` so a validator can
    tell "documented extra" from "unexpected".
    """
    hidden = int(_kv(metadata, "embedding_length"))
    experts = int(_kv(metadata, "expert_count"))
    inter = int(_kv(metadata, "expert_feed_forward_length"))
    shexp = int(_kv(metadata, "expert_shared_feed_forward_length"))
    head_dim = int(_kv(metadata, "attention.key_length"))
    heads = int(_kv(metadata, "attention.head_count"))
    block_count = int(_kv(metadata, "block_count"))
    n_mtp = int(metadata.get(f"{ARCH}.nextn_predict_layers", 0))
    num_layers = block_count - n_mtp
    # GDN dims (also present on full-attention blocks' metadata; only used for GDN tensors)
    nk = int(_kv(metadata, "ssm.group_count"))          # k heads
    nv = int(_kv(metadata, "ssm.time_step_rank"))       # v heads / dt rank
    dk = int(_kv(metadata, "ssm.state_size"))           # k head dim
    dv = int(_kv(metadata, "ssm.inner_size")) // nv     # v head dim
    conv = int(_kv(metadata, "ssm.conv_kernel"))
    types = layer_types(metadata, num_layers)

    out: dict[str, ExpectedTensor] = {}
    add = out.__setitem__

    add("token_embd.weight", ExpectedTensor("token_embd.weight", "embed", "model.embed_tokens.weight", (hidden, 0), ("quant",)))
    add("output_norm.weight", ExpectedTensor("output_norm.weight", "norm", "model.norm.weight", (hidden,), ("F32",)))
    # untied head: file has output.weight -> tie_word_embeddings False (shim derives this)
    add("output.weight", ExpectedTensor("output.weight", "lm_head", "lm_head.weight", (hidden, 0), ("quant",)))

    for i in range(num_layers):
        blk = f"blk.{i}"
        pre = f"model.layers.{i}"
        add(f"{blk}.attn_norm.weight", ExpectedTensor(f"{blk}.attn_norm.weight", "norm", f"{pre}.input_layernorm.weight", (hidden,), ("F32",)))
        add(f"{blk}.post_attention_norm.weight", ExpectedTensor(f"{blk}.post_attention_norm.weight", "norm", f"{pre}.post_attention_layernorm.weight", (hidden,), ("F32",)))
        add(f"{blk}.ffn_gate_inp.weight", ExpectedTensor(f"{blk}.ffn_gate_inp.weight", "router", f"{pre}.mlp.gate.weight", (hidden, experts), ("F32",)))
        add(f"{blk}.ffn_gate_inp_shexp.weight", ExpectedTensor(f"{blk}.ffn_gate_inp_shexp.weight", "router", f"{pre}.mlp.shared_expert_gate.weight", (hidden,), ("F32",)))
        add(f"{blk}.ffn_gate_shexp.weight", ExpectedTensor(f"{blk}.ffn_gate_shexp.weight", "shared_expert", f"{pre}.mlp.shared_expert.gate_up_proj.weight", (hidden, shexp), ("quant",)))
        add(f"{blk}.ffn_up_shexp.weight", ExpectedTensor(f"{blk}.ffn_up_shexp.weight", "shared_expert", f"{pre}.mlp.shared_expert.gate_up_proj.weight", (hidden, shexp), ("quant",)))
        add(f"{blk}.ffn_down_shexp.weight", ExpectedTensor(f"{blk}.ffn_down_shexp.weight", "shared_expert", f"{pre}.mlp.shared_expert.down_proj.weight", (shexp, hidden), ("quant",)))
        add(f"{blk}.ffn_gate_exps.weight", ExpectedTensor(f"{blk}.ffn_gate_exps.weight", "experts", None, (hidden, inter, experts), ("",)))
        add(f"{blk}.ffn_up_exps.weight", ExpectedTensor(f"{blk}.ffn_up_exps.weight", "experts", None, (hidden, inter, experts), ("",)))
        add(f"{blk}.ffn_down_exps.weight", ExpectedTensor(f"{blk}.ffn_down_exps.weight", "experts", None, (inter, hidden, experts), ("",)))
        if types[i] == "linear_attention":
            add(f"{blk}.attn_qkv.weight", ExpectedTensor(f"{blk}.attn_qkv.weight", "gdn", f"{pre}.linear_attn.in_proj_qkvz.weight", (hidden, 2 * nk * dk + nv * dv), ("quant",)))
            add(f"{blk}.attn_gate.weight", ExpectedTensor(f"{blk}.attn_gate.weight", "gdn", f"{pre}.linear_attn.in_proj_qkvz.weight", (hidden, nv * dv), ("quant",)))
            add(f"{blk}.ssm_beta.weight", ExpectedTensor(f"{blk}.ssm_beta.weight", "gdn", f"{pre}.linear_attn.in_proj_ba.weight", (hidden, nv), ("F32",)))
            add(f"{blk}.ssm_alpha.weight", ExpectedTensor(f"{blk}.ssm_alpha.weight", "gdn", f"{pre}.linear_attn.in_proj_ba.weight", (hidden, nv), ("F32",)))
            add(f"{blk}.ssm_a", ExpectedTensor(f"{blk}.ssm_a", "gdn", f"{pre}.linear_attn.A_log", (nv,), ("F32",)))
            add(f"{blk}.ssm_dt.bias", ExpectedTensor(f"{blk}.ssm_dt.bias", "gdn", f"{pre}.linear_attn.dt_bias", (nv,), ("F32",)))
            add(f"{blk}.ssm_conv1d.weight", ExpectedTensor(f"{blk}.ssm_conv1d.weight", "gdn", f"{pre}.linear_attn.conv1d.weight", (conv, 2 * nk * dk + nv * dv), ("F32",)))
            add(f"{blk}.ssm_norm.weight", ExpectedTensor(f"{blk}.ssm_norm.weight", "gdn", f"{pre}.linear_attn.norm.weight", (dk,), ("F32",)))
            add(f"{blk}.ssm_out.weight", ExpectedTensor(f"{blk}.ssm_out.weight", "gdn", f"{pre}.linear_attn.out_proj.weight", (nv * dv, hidden), ("quant",)))
        else:
            q_rows = 2 * heads * head_dim  # q_proj packs query AND the output gate (attention.py: num_q*head_dim*2)
            kv_rows = int(_kv(metadata, "attention.head_count_kv")) * head_dim
            add(f"{blk}.attn_q.weight", ExpectedTensor(f"{blk}.attn_q.weight", "attention", f"{pre}.self_attn.qkv_proj.weight", (hidden, q_rows), ("quant",)))
            add(f"{blk}.attn_k.weight", ExpectedTensor(f"{blk}.attn_k.weight", "attention", f"{pre}.self_attn.qkv_proj.weight", (hidden, kv_rows), ROLE_QUANTS["attn_k_v"]))
            add(f"{blk}.attn_v.weight", ExpectedTensor(f"{blk}.attn_v.weight", "attention", f"{pre}.self_attn.qkv_proj.weight", (hidden, kv_rows), ROLE_QUANTS["attn_k_v"]))
            add(f"{blk}.attn_output.weight", ExpectedTensor(f"{blk}.attn_output.weight", "attention", f"{pre}.self_attn.o_proj.weight", (heads * head_dim, hidden), ("quant",)))
            add(f"{blk}.attn_q_norm.weight", ExpectedTensor(f"{blk}.attn_q_norm.weight", "norm", f"{pre}.self_attn.q_norm.weight", (head_dim,), ("F32",)))
            add(f"{blk}.attn_k_norm.weight", ExpectedTensor(f"{blk}.attn_k_norm.weight", "norm", f"{pre}.self_attn.k_norm.weight", (head_dim,), ("F32",)))

    # Trailing MTP block(s): a full-attention block's worth of decoder tensors plus the
    # nextn head. They ride along in the file (llama.cpp quantizes them) and are never loaded.
    for b in range(num_layers, block_count):
        blk = f"blk.{b}"
        kv_rows = int(_kv(metadata, "attention.head_count_kv")) * head_dim
        q_rows = 2 * heads * head_dim
        mtp = {
            f"{blk}.attn_norm.weight": ((hidden,), ("F32",)),
            f"{blk}.post_attention_norm.weight": ((hidden,), ("F32",)),
            f"{blk}.attn_q.weight": ((hidden, q_rows), ("quant",)),
            f"{blk}.attn_k.weight": ((hidden, kv_rows), ROLE_QUANTS["attn_k_v"]),
            f"{blk}.attn_v.weight": ((hidden, kv_rows), ROLE_QUANTS["attn_k_v"]),
            f"{blk}.attn_output.weight": ((heads * head_dim, hidden), ("quant",)),
            f"{blk}.attn_q_norm.weight": ((head_dim,), ("F32",)),
            f"{blk}.attn_k_norm.weight": ((head_dim,), ("F32",)),
            f"{blk}.ffn_gate_inp.weight": ((hidden, experts), ("F32",)),
            f"{blk}.ffn_gate_inp_shexp.weight": ((hidden,), ("F32",)),
            f"{blk}.ffn_gate_shexp.weight": ((hidden, shexp), ("quant",)),
            f"{blk}.ffn_up_shexp.weight": ((hidden, shexp), ("quant",)),
            f"{blk}.ffn_down_shexp.weight": ((shexp, hidden), ("quant",)),
            f"{blk}.ffn_gate_exps.weight": ((hidden, inter, experts), ("",)),
            f"{blk}.ffn_up_exps.weight": ((hidden, inter, experts), ("",)),
            f"{blk}.ffn_down_exps.weight": ((inter, hidden, experts), ("",)),
            f"{blk}.nextn.eh_proj.weight": ((0, hidden), ("quant",)),
            f"{blk}.nextn.enorm.weight": ((hidden,), ("F32",)),
            f"{blk}.nextn.hnorm.weight": ((hidden,), ("F32",)),
            f"{blk}.nextn.shared_head_norm.weight": ((hidden,), ("F32",)),
        }
        for name, (ne, quants) in mtp.items():
            add(name, ExpectedTensor(name, "mtp", None, ne, quants))
    return out


def expert_quant_agreement(index: dict[str, GgufTensorInfo]) -> tuple[dict[int, tuple[str, str, str]], list[str]]:
    """Per MoE block the (gate, up, down) ggml type names, plus one problem string per violation.

    Upstream's bank loader enforces gate type == up type; this also flags gate != down
    because the model computes gate and down in the same dtype per layer.
    """
    from freetoken.layers.quantization.ggml_types import type_name

    out: dict[int, tuple[str, str, str]] = {}
    problems: list[str] = []
    for name, info in index.items():
        m = _BLOCK_RE.match(name)
        if m is None or not name.endswith("_exps.weight"):
            continue
        blk, role = int(m.group(1)), m.group(2).split("_")[1]  # gate | up | down
        q = out.setdefault(blk, ("", "", ""))  # type: ignore[assignment]
        slot = {"gate": 0, "up": 1, "down": 2}[role]
        out[blk] = tuple(type_name(info.ggml_type) if j == slot else q[j] for j in range(3))  # type: ignore[assignment]
    for blk, (g, u, d) in sorted(out.items()):
        if g != u:
            problems.append(f"blk.{blk}: gate experts {g} != up experts {u} (upstream bank loader refuses this)")
        if g != d:
            problems.append(f"blk.{blk}: gate experts {g} != down experts {d} (same-dtype-per-layer assumption broken)")
    return out, problems


def validate(model_path: str, *, strict: bool = True) -> MappingReport:
    """Build the expected-vs-present report for ``model_path`` using upstream's own readers.

    ``strict`` additionally pins each tensor's ggml type to the profile recorded from
    the CyberTiel file (Q5_K/Q4_K experts, Q8_0 matrices, F16 k/v, F32 norms); with
    ``strict=False`` only structural rules run (shapes, expert agreement, unread-scale
    refusal) so other qwen35moe files can be described too.
    """
    from freetoken.layers.quantization.ggml_types import type_name

    metadata = load_gguf_metadata(metadata_path := model_path)
    index = gguf_tensor_index(metadata_path)
    arch = str(metadata.get("general.architecture", ""))
    num_layers = int(_kv(metadata, "block_count")) - int(metadata.get(f"{ARCH}.nextn_predict_layers", 0))
    types = layer_types(metadata, num_layers)
    expected = expected_tensors(metadata)

    report = MappingReport(
        arch=arch,
        num_layers=num_layers,
        layer_types=types,
        mtp_blocks=list(range(num_layers, int(_kv(metadata, "block_count")))),
        expert_quants={},
        expected=expected,
        present=set(index),
    )
    if arch != ARCH:
        report.problems.append(f"general.architecture is {arch!r}, this mapping is for {ARCH!r}")
        return report
    if "output.weight" not in index:
        # upstream's shim derives tie_word_embeddings from the tensor table: no head = tied
        del expected["output.weight"]

    # upstream refuses unread .scale tensors before anything else
    try:
        check_scales(model_path)
    except NotImplementedError as exc:
        report.problems.append(str(exc))

    report.expert_quants, agree = expert_quant_agreement(index)
    report.problems.extend(agree)
    if strict:
        for blk, qs in sorted(report.expert_quants.items()):
            for q in qs:
                if q and q not in ROLE_QUANTS["expert_bank"]:
                    report.type_mismatch.append(f"blk.{blk}: expert type {q}, not in {ROLE_QUANTS['expert_bank']}")

    for name, want in expected.items():
        info = index.get(name)
        if info is None:
            # MTP-block tensors ride along when the converter wrote them; a file may
            # legally strip them, so their absence is not a mapping failure
            if want.role != "mtp":
                report.missing.append(name)
            continue
        got = type_name(info.ggml_type)
        # "quant" = any quantized type; "" = any type at all (expert banks, pinned per file)
        if want.quants and got not in want.quants and "quant" not in want.quants and "" not in want.quants:
            report.type_mismatch.append(f"{name}: {got}, expected one of {want.quants}")
        if want.ne and len(want.ne) == len(info.ne):
            if any(w != 0 and g != w for w, g in zip(want.ne, info.ne)):
                report.shape_mismatch.append(f"{name}: ne={info.ne}, expected ne={want.ne}")
        elif want.ne:
            report.shape_mismatch.append(f"{name}: ne={info.ne} has {len(info.ne)} dims, expected {len(want.ne)}")

    # present-but-excluded: MTP-block tensors (either matched in the expected table
    # with role "mtp" or unmatched extras that live in an excluded block)
    report.mtp_present = sorted(
        n for n in set(index) if _is_mtp(n, num_layers) or expected.get(n, ExpectedTensor(n, "", None, ())).role == "mtp"
    )
    for name in sorted(set(index) - set(expected)):
        if name not in report.mtp_present:
            report.unexpected.append(name)
    return report


def _is_mtp(name: str, num_layers: int) -> bool:
    m = _BLOCK_RE.match(name)
    return m is not None and int(m.group(1)) >= num_layers


# The measured profile of the CyberTiel "MTPv2-ICE" file, for docs and strict checks.
CYBERTIEL_EXPECTED_EXPERT_QUANTS = {  # block range -> (gate, up, down); 15 + 26 blocks
    "blk.0-14": ("Q5_K", "Q5_K", "Q5_K"),
    "blk.15-40": ("Q4_K", "Q4_K", "Q4_K"),
}


def mapping_summary(model_path: str) -> dict[str, Any]:
    """One dict per file: everything the Vulkan loader needs to know before touching bytes."""
    metadata = load_gguf_metadata(model_path)
    index = gguf_tensor_index(model_path)
    num_layers = int(_kv(metadata, "block_count")) - int(metadata.get(f"{ARCH}.nextn_predict_layers", 0))
    eq, _ = expert_quant_agreement(index)
    from freetoken.layers.quantization.ggml_types import type_name

    def _count(pred) -> int:
        return sum(1 for i in index.values() if pred(i))

    return {
        "arch": metadata.get("general.architecture"),
        "name": metadata.get("general.name"),
        "file_type": metadata.get("general.file_type"),
        "block_count": int(_kv(metadata, "block_count")),
        "num_layers": num_layers,
        "nextn_predict_layers": int(metadata.get(f"{ARCH}.nextn_predict_layers", 0)),
        "layer_types": layer_types(metadata, num_layers),
        "hidden": int(_kv(metadata, "embedding_length")),
        "vocab": int(index["token_embd.weight"].ne[-1]),
        "experts": int(_kv(metadata, "expert_count")),
        "experts_per_tok": int(_kv(metadata, "expert_used_count")),
        "tied": "output.weight" not in index,
        "tensor_count": len(index),
        "expert_quants": {f"blk.{b}": q for b, q in sorted(eq.items())},
        "ggml_types": {type_name(t.ggml_type): _count(lambda i, n=type_name(t.ggml_type): type_name(i.ggml_type) == n) for t in index.values()},
    }


__all__ = [
    "ARCH", "REQUIRED_METADATA", "ROLE_QUANTS", "CYBERTIEL_EXPECTED_EXPERT_QUANTS",
    "ExpectedTensor", "MappingReport", "expected_tensors", "expert_quant_agreement",
    "layer_types", "mapping_summary", "validate",
]
