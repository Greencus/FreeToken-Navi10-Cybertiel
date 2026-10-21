# CyberTiel GGUF mapping (Ornith-1.5-35B-A3B, arch `qwen35moe`)

Ground-truth mapping of the CyberTiel-calibrated checkpoint onto FreeToken's GGUF
machinery, for the Vulkan backend. Everything below was measured from the file
itself (tensor header + KV reads only, never materializing weights) on 2026-10-20.

- File: `~/.cache/huggingface/hub/models--gbuzhf--Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF/blobs/579a20a18649e79a9e1d56bb8c88150fe5764fb83903154de3c4cb9d6de2ef8a` (21.26 GiB of tensors, 753 total).
  The friendly path ending in `...-ICE-GGUF/Ornith-...MTPv2-ICE-GGUF.gguf` is a
  symlink to that blob; read the blob path, it is never copied.
- Arch: `qwen35moe` -> FreeToken `Qwen3_5MoeGGUFForCausalLM`, model_type `qwen35moe`.
- Helpers: `python/freetoken/models/gguf/arch_qwen35moe.py` (expected-tensor table,
  `validate()`, `mapping_summary()`), `python/freetoken/models/gguf/dequant.py`
  (K-quant -> BF16), tests in `tests/gguf_vulkan/`.

## 1. Metadata (KV section, verbatim)

| Key | Value |
| --- | --- |
| `general.architecture` | `qwen35moe` |
| `general.name` | `Src_Hf` |
| `general.file_type` | `15` (`MOSTLY_Q4_K_M`) |
| `qwen35moe.block_count` | 41 |
| `qwen35moe.nextn_predict_layers` | 1 -> **40 real layers** (`num_layers = block_count - nextn`) |
| `qwen35moe.full_attention_interval` | 4 -> full-attention at blk.3,7,...,39 (10 layers) |
| `qwen35moe.embedding_length` | 2048 |
| `qwen35moe.context_length` | 262144 |
| `qwen35moe.attention.head_count` / `head_count_kv` | 16 / 2 |
| `qwen35moe.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35moe.attention.layer_norm_rms_epsilon` | 9.999999974752427e-07 |
| `qwen35moe.expert_count` / `expert_used_count` | 256 / 8 |
| `qwen35moe.expert_feed_forward_length` / `_shared_` | 512 / 512 |
| `qwen35moe.ssm.conv_kernel` / `group_count` / `inner_size` | 4 / 16 / 4096 |
| `qwen35moe.ssm.state_size` / `time_step_rank` | 128 / 32 |
| `qwen35moe.rope.freq_base` / `dimension_count` / `dimension_sections` | 1e7 / 64 / [11, 11, 10, 0] |

`qwen35moe.attention.recurrent_layers` (bitmask override) is **absent**; layer typing
falls back to the interval rule (identical result: blocks 0,1,2,4,5,... are linear).

Parsed by `parse_gguf_config`: heads=16, kv_heads=2, head_dim=256; linear-attention
group key_heads=16 x key_dim=128 (2048), value_heads=32 x value_dim=128 (4096),
conv=4; `RotaryConfig(head_dim=256, rotary_dim=64, max_position=262144, base=1e7)`.

## 2. Layer structure

- **30 GDN (linear-attention) layers**: blk.0,1,2,4,5,6,8,... (everything not
  `blk.(4k+3)`); **10 full-attention layers**: blk.3,7,11,15,19,23,27,31,35,39.
- **blk.40 = MTPv2 block**: full-attention layout (20 tensors) + `nextn.*` head
  tensors. Excluded from `num_layers` by upstream's
  `qwen3_5_moe/gguf.py` (`num_layers = block_count - nextn_predict_layers`).
- Per-layer tensor sets (leaf names, all `.weight`):

| Set | Count | Leaves |
| --- | --- | --- |
| GDN layer | 19 | attn_norm, attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt.bias, ssm_norm, ssm_out, ssm_a, post_attention_norm, ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp |
| Full-attention layer | 16 | attn_norm, attn_q, attn_q_norm, attn_k, attn_k_norm, attn_v, attn_output, post_attention_norm, ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp |
| blk.40 (MTP) | 20 | full-attention set (16) + nextn.eh_proj, nextn.enorm, nextn.hnorm, nextn.shared_head_norm |
| Top level | 3 | token_embd.weight, output.weight (untied), output_norm.weight |

30x19 + 10x16 + 20 + 3 = **753** (matches the file).

## 3. Quant inventory (measured totals)

| Type | Tensors | Size | Where |
| --- | --- | --- | --- |
| Q5_K | 45 | 7.73 GiB | experts, blk.0-14 (15 layers x gate/up/down) |
| Q4_K | 78 | 10.97 GiB | experts, blk.15-40 (26 layers x gate/up/down; blk.40 ignored) |
| Q8_0 | 238 | 2.41 GiB | token_embd/output, all dense projections, shared experts, ssm_out |
| F16 | 22 | 0.04 GiB | attn_k/attn_v of the 11 full-attention blocks |
| F32 | 370 | 0.10 GiB | all norms, router/gate matrices, GDN state tensors |
| **total** | **753** | **21.26 GiB** | |

**Correction to the task brief**: the expert quants are **Q5_K (blk.0-14) + Q4_K
(blk.15-40), not Q6_K**. gate/up/down always agree within a layer (per-layer tags:
`('ggml','Q5_K','Q5_K')` first, `('ggml','Q4_K','Q4_K')` last). Packed size per
layer: 528 MiB (Q5_K) or 432 MiB (Q4_K) for the three expert tensors; all expert
slabs together = 18.70 GiB (18.3 GiB for the 40 real layers, blk.40 excluded).

## 4. Mapping table (GGUF leaf -> FreeToken key -> transform -> Vulkan handling)

Transforms are what `qwen3_5_moe/gguf.py::iter_gguf_weights` applies; "Vulkan
handling" is the backend plan for this repo (packed K-quant experts by default
with the host-dequant BF16 path behind `FTV_GGUF_PACKED=0`, see
`docs/vulkan/porting-guide.md` and `docs/vulkan/kquant-kernels.md`).

### Top level

| GGUF tensor | Type | FreeToken key | Loader transform | Vulkan handling |
| --- | --- | --- | --- | --- |
| `token_embd.weight` | Q8_0 248320x2048 | `model.embed_tokens.weight` | `matrix()` dequant | host-dequant -> BF16 lookup |
| `output.weight` | Q8_0 248320x2048 | `lm_head.weight` | `matrix()` dequant (untied) | host-dequant -> BF16 logits matmul |
| `output_norm.weight` | F32 | `model.norm.weight` | `float_tensor - 1.0` (zero-centred) | fp32, dense cache |

### GDN layer (30x)

| GGUF tensor | Type | FreeToken key | Loader transform | Vulkan handling |
| --- | --- | --- | --- | --- |
| `attn_norm` | F32 | `...input_layernorm.weight` | `x - 1.0` | fp32, `_dense_cache` |
| `post_attention_norm` | F32 | `...post_attention_layernorm.weight` | `x - 1.0` | fp32, `_dense_cache` |
| `attn_qkv` | Q8_0 (8192, 2048) | `linear_attn.in_proj_qkv(z)` | untile v-rows (k-head minor) | BF16 fused in-proj (`linear_bf16_resident_io`) |
| `attn_gate` | Q8_0 | `linear_attn.in_proj_qkv(z)` (fused) | untile (nk, dim) | same fused kernel |
| `ssm_beta` / `ssm_alpha` | F32 | `linear_attn.in_proj_b(a)` (fused) | untile (nk, 1) | BF16 fused |
| `ssm_conv1d` | F32 (8192, 4) | `linear_attn.conv1d.weight` | squeeze, untile v part, `unsqueeze(1)` | `gdn_conv_f32` shader |
| `ssm_dt.bias` | F32 | `linear_attn.dt_bias` | untile (nk, 1) | per-lane scalar |
| `ssm_a` | F32 | `linear_attn.A_log` | `log(-x)` then untile (nk, 1) | `gdn_recur_f32` |
| `ssm_norm` | F32 | `linear_attn.norm.weight` | none | head RMS in `gdn_recur_f32` |
| `ssm_out` | Q8_0 | `linear_attn.out_proj` | `matrix()`, columns stay tiled (input permuted instead) | BF16 resident out-proj |
| `ffn_gate_inp` | F32 | `mlp.gate.weight` (router) | none | fp32 router (`FTV_GPU_ROUTER` path) |
| `ffn_{gate,up,down}_exps` | Q5_K/Q4_K bank (256, 512/2048, 512/2048) | `mlp.experts.0.{gate,up,down}_proj` | offload bank, not iterated by `iter_gguf_weights` | packed slab -> `linear_q{4k,5k}_resident_io` (default); `dequant_expert_banks()` -> BF16 under `FTV_GGUF_PACKED=0` |
| `ffn_gate_inp_shexp` | F32 | `mlp.shared_expert_gate.weight` | `reshape(1,-1)` | fp32 |
| `ffn_{gate,up}_shexp` | Q8_0 | `mlp.shared_expert.gate_up_proj` (fused) | `matrix()` x2 | BF16 `batch_swiglu_io` |
| `ffn_down_shexp` | Q8_0 | `mlp.shared_expert.down_proj` | `matrix()` | BF16 |

### Full-attention layer (10x)

| GGUF tensor | Type | FreeToken key | Loader transform | Vulkan handling |
| --- | --- | --- | --- | --- |
| `attn_q` | Q8_0 (8192, 2048) | `self_attn.qkv_proj` (fused q,k,v) | `matrix()`, fused with k/v | BF16 resident |
| `attn_k` / `attn_v` | F16 | `self_attn.qkv_proj` (fused) | `matrix()` | BF16 resident |
| `attn_q_norm` / `attn_k_norm` | F32 | `self_attn.{q,k}_norm.weight` | `x - 1.0` | fp32 |
| `attn_output` | Q8_0 | `self_attn.o_proj` | `matrix()` (`ne[0]` out dim) | BF16 resident |
| attention QKV is stored as three tensors here (not fused in the file); the loader fuses them into one `qkv_proj`. MoE leaves as in the GDN table. | | | | |

### blk.40 (MTPv2) - ignored by design

`num_layers = block_count - nextn_predict_layers = 40`, so upstream never asks for
blk.40. `arch_qwen35moe.py` treats its 20 tensors as documented extras
(`report.mtp_present`, absent rather than unexpected is also accepted). The 4
`nextn.*` tensors (`eh_proj` Q8_0 4096x2048, `enorm`, `hnorm`, `shared_head_norm`)
have no FreeToken consumer; FreeToken does not ship an MTP head loader.

## 5. Loading status (verified, `/tmp/opencode/loader_exercise.log`)

Run with the interpreter `/home/cus/.venv-vllm-vulkan/bin/python` and
`PYTHONPATH=python`:

1. `build_gguf_shim` + `parse_gguf_config`: architectures, 40 layers, 30/10
   split at [3,7,...,39], vocab 248320, 256 experts top-8, tied=false.
2. `iter_gguf_weights` (meta readers, no weight bytes copied): **503/503 state-dict
   tensors produced, 0 missing, 0 unexpected, 0 shape/type problems**.
3. Expert banks (`_expert_banks`): 40 layers, 18.3 GiB on meta, tags Q5_K then
   Q4_K, **0 problems**.
4. `arch_qwen35moe.validate`: `ok=True`, all lists empty, `mtp_present` = the 20
   blk.40 tensors.
5. **Tokenizer fails on this machine**: `ModuleNotFoundError: No module named
   'transformers.integrations.gguf'` - the venv has transformers 5.16.1, and only
   `transformers>=5.18` (which is exactly what `pyproject.toml` pins) ships the
   `integrations.gguf` package `freetoken/models/gguf/tokenizer.py:77` imports.
   Environment gap, not a mapping or code bug; installing the pinned range fixes it.
6. `scripts/gguf_check/check.py` on the same file: config/weights/banks pass, same
   tokenizer failure (record in `/tmp/opencode/cybertiel_check.jsonl`).

Reproduce:

```bash
PYTHONPATH=python /home/cus/.venv-vllm-vulkan/bin/python /tmp/opencode/loader_exercise.py /tmp/opencode/CyberTiel.gguf
PYTHONPATH=python FREETOKEN_CYBERTIEL_GGUF=/tmp/opencode/CyberTiel.gguf \
  /home/cus/.venv-vllm-vulkan/bin/python -m pytest tests/gguf_vulkan/ -q   # 22 passed
```

### Dequant validation (numbers)

`python/freetoken/models/gguf/dequant.py` is validated three ways:

1. **Q8_0 byte-exact vs an independent scalar unpacker** (test-built, no shared
   code with `third_party/gguf/quants.py`): exact `np.array_equal` on the real
   `blk.0.attn_qkv` (8192x2048) and `blk.0.ffn_gate_shexp` samples and on
   synthetic files.
2. **Bit-exact vs llama.cpp's reference C dequantizers**: the real packed rows go
   through both `dequant_matrix`/`dequant_expert` and llama.cpp's
   `dequantize_row_q4_K/_q5_K/_q8_0` (harness
   `/tmp/opencode/llama_ref_dequant.c` compiled against
   `llama.cpp/ggml/src/ggml-quants.c`): **18 rows over 6 tensors, max_abs =
   0.000e+00, max_rel = 0.000e+00 (all PASS)** - blk.0/blk.5 (Q5_K),
   blk.15/blk.20 (Q4_K), blk.0/blk.39 shared experts (Q8_0); driver
   `/tmp/opencode/llama_crosscheck.py`.
3. **Structural gates** (always-on tests): per-expert slice == whole-tensor
   dequant for Q4_K/Q5_K/Q8_0 banks, scale/min reconstruction bounds from the
   raw superblock fields, packed-size accounting, cache LRU/`FTV_GGUF_DEQUANT_CACHE_MB`
   behavior.


## 6. Failure mode seen with the raw blob path (and the upstream patch)

Feeding the extension-less `blobs/<sha>` path to `check.py` killed the worker:
`is_gguf_path()` (`python/freetoken/models/gguf/reader.py:21`) requires the
`.gguf` suffix, so `cached_load_hf_config` fell through to
`AutoConfig.from_pretrained` (`python/freetoken/utils/hf.py:226`), which tried to
JSON-parse the 21 GiB binary (`transformers/configuration_utils.py` ->
`_dict_from_json_file`) and died at ~20 GB RSS. This is an upstream-side
robustness gap; the fix belongs in `reader.py`, **not** applied here (upstream
scope, and this repo commits only on request):

```diff
--- a/python/freetoken/models/gguf/reader.py
+++ b/python/freetoken/models/gguf/reader.py
@@ def is_gguf_path(model_path: str) -> bool:
-    """A single ``.gguf`` file (the only GGUF layout FreeToken loads directly)."""
-    return isinstance(model_path, str) and os.path.isfile(model_path) and model_path.endswith(
-        ".gguf"
-    )
+    """A single GGUF file (the only GGUF layout FreeToken loads directly).
+
+    Match by magic bytes when the suffix is missing: HF cache blobs
+    (``.../blobs/<sha>``) hold valid GGUF data without a ``.gguf`` name, and
+    falling through to AutoConfig parses the whole binary as JSON until OOM.
+    """
+    if not (isinstance(model_path, str) and os.path.isfile(model_path)):
+        return False
+    if model_path.endswith(".gguf"):
+        return True
+    with open(model_path, "rb") as f:
+        return f.read(4) == b"GGUF"
```

Workaround used by every command above: point at a `.gguf`-suffixed symlink
(`/tmp/opencode/CyberTiel.gguf` -> the blob).

## 7. Related files

- `python/freetoken/models/gguf/arch_qwen35moe.py` - expected-tensor table and
  `validate()`/`mapping_summary()` for this checkpoint family.
- `python/freetoken/models/gguf/dequant.py` - Q8_0/Q4_K/Q5_K -> BF16 helpers plus
  `DequantCache` (`FTV_GGUF_DEQUANT_CACHE_MB`, default 256 MiB) for repeated
  expert reads during serving.
- `tests/gguf_vulkan/test_cybertiel_mapping.py`, `tests/gguf_vulkan/test_gguf_dequant.py`.
- `python/freetoken/models/qwen3_5_moe/gguf.py` - upstream loader (read-only);
  `_LAYER_LEAVES`, transforms (A_log = `log(-x)`, zero-centred norms `w+1`,
  k-head-minor tiling) are documented in its module docstring.
