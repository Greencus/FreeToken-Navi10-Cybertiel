# tests/vulkan - gates for the Vulkan Qwen3.5-MoE engine port

These tests exercise the ported package
`python/freetoken/models/qwen3_5_moe/vulkan/` (from the `freetoken-vulkan`
fork) plus the native extension resolved through
`freetoken/models/qwen3_5_moe/vulkan/_runtime.py`.

## Running

```bash
cd FreeToken-Navi10-Cybertiel
PYTHONPATH=python python -m pytest tests/vulkan/ -q
```

Interpreter: `/home/cus/.venv-vllm-vulkan/bin/python` (py3.12, torch 2.11+cpu).

The default run never touches a checkpoint: model-dependent gates skip
unless their env var is set.

## Always-on gates (ext + small tensors only)

| File | What it gates |
| --- | --- |
| `test_vulkan_smoke.py` | runtime resolution (preferred `freetoken.kernel.vulkan`, staged `FREETOKEN_VULKAN_EXT_DIR` fallback), `.spv` dir populated; `rmsnorm` vs torch <= 1e-5; one `linear_bf16_resident_io` (fork call signature) vs `x @ W.T` (rtol 1e-4 / atol 5e-5); batch upload/download round trip bit-exact; `free_resident` restores the pool |
| `test_ressl_nvfp4_kernel.py::test_nvfp4_kernel_synthetic` | `linear_nvfp4_resident_io` on fabricated E2M1+e4m3 data vs host `dequantize_nvfp4` (rtol/atol 1e-3): shader codebook, scale packing (`pack_scales_u32`), and fp32 accumulation agree |

All tests skip cleanly (pytest.skip) when no Vulkan device/extension is
available - the `vulkan_ext` fixture in `conftest.py` probes with a trivial
allocate/free first.

## Opt-in model gates (env var required, else skip)

| File | Env var | Checkpoint | Gate |
| --- | --- | --- | --- |
| `test_gdn_kernel.py` | `FREETOKEN_TEST_MODEL` | Ressl Ornith-1.5 NVFP4 | GDN GPU conv+recur vs `_gdn_host` maxdiff < 1e-5 |
| `test_batched_fork2.py` | `FREETOKEN_TEST_MODEL` | same | seq-vs-batched maxdiff < 1e-5, batched-vs-batched2 bit-exact 0.0, argmax identical (the fork's headline parity gate; fork-era observation 5.484e-06 router harness, measured 6.199e-06 router-off / 5.543e-06 router-on on the Oct-20 stack - see the test docstring) |
| `test_ressl_nvfp4_kernel.py::test_nvfp4_kernel_real_weights` | `FREETOKEN_TEST_MODEL` | same | real expert gate/down NVFP4 kernels vs CPU dequant, rtol/atol 2e-3 |
| `test_qwen35_moe_parity.py` | `FREETOKEN_TEST_MODEL_QWEN35` | Cyber-Tiel TorchAO-affine | `moe_forward` vs `transformers` `Qwen3_5MoeSparseMoeBlock`, maxdiff < 1e-4 (CPU only, needs several GB RAM) |
| `test_resident_moe.py` | `FREETOKEN_TEST_MODEL_GPT_OSS` | gpt-oss-20b | `moe_mlp_gpt_oss_resident` vs per-call-upload path, maxdiff < 1e-4 |
| `test_gguf_parity.py` | `FREETOKEN_TEST_MODEL_GGUF` | CyberTiel GGUF | GDN (layer 0) and full-attn (layer 3) steps vs a host reference on the same BF16 weights, maxdiff < 1e-5; `generate()` twice from a fixed prompt -> identical ids |
| `test_mtp.py` | `FREETOKEN_TEST_MODEL_GGUF` | CyberTiel GGUF | MTP (nextn) speculative decode: `generate_mtp` token ids == `generate`'s greedy ids on 3 prompts (the identity gate), `prefill_with_h` leaves the production prefill bit-identical, the accept rule, and the `--mtp` CLI guards. The two device-free gates (accept rule, CLI guards) run in the default `pytest tests/vulkan` without a checkpoint |

`test_mtp.py` prints the accept rate, rounds/rollbacks and full-passes per
token for each prompt; `tests/vulkan/bench_mtp.py` (not collected - run it
directly) is the A/B bench for tok/s and accept rate against the same
prompt's non-speculative run.

Example:

```bash
FREETOKEN_TEST_MODEL=/path/to/Ornith-NVFP4 \
  PYTHONPATH=python python -m pytest tests/vulkan/ -q
```

The model gates share one `ResidentNVFP4` instance (session fixture) to
avoid pinning the weights twice; each test resets recurrent state first.

## Env knobs honored

`FREETOKEN_VULKAN_SHADERS`, `FREETOKEN_VULKAN_EXT_DIR` (runtime resolution);
`FTV_EXP_CACHE_MB`, `FTV_HOST_PACK_MB`, `FTV_NVFP4_GPU`, `FTV_NVFP4_M1`,
`FTV_DEBUG_X`, `FTV_DEBUG_TERMS` (engine behavior, unchanged from the fork).
