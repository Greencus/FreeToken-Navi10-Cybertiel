# Vulkan compute backend (AMD GPUs)

FreeToken's engine targets NVIDIA GPUs: CUDA graphs, NCCL, Triton kernels and
`torch.cuda` streams are load-bearing. This directory documents the Vulkan
backend, a from-scratch compute path for AMD RDNA1/RDNA2 cards (the reference
machine is an RX 5700 XT, Navi10 / gfx1010, 8 GB).

The Vulkan backend is a **separate runtime**, not a drop-in replacement for the
CUDA kernels. `freetoken.kernel.vulkan` is the native layer (a pybind11
extension over a small Vulkan runtime plus GLSL compute shaders), and
`freetoken.models.qwen3_5_moe.vulkan` is a complete implementation of the
Qwen3.5-MoE family that runs on it. Upstream's engine and scheduler stay
untouched, so the CUDA path is unaffected.

## Layout

```
python/freetoken/kernel/vulkan/
  ext_module.cpp      pybind11 bindings: 82 ops (resident buffers, batch engine,
                      MoE, attention, GDN, router, K-quant GEMV)
  vk_util.hpp         device, memory, pipeline, descriptor and timestamp helpers
  shaders/*.comp      52 GLSL compute shaders (linear, K-quant, attention, MoE,
                      GDN, norm)
  _build.py           extension build (prebuilt -> JIT -> direct c++)
  build.py            CLI: build the extension and compile the shaders
python/freetoken/models/qwen3_5_moe/vulkan/
  config.py loader.py layer.py moe.py attention.py linear_attn.py
  resident.py         the resident decoder (streaming expert cache)
  nvfp4.py nvfp4_resident.py   the NVFP4 (ModelOpt) expert path
  engine.py cli.py    the decode loop and the command line entry point
  mtp.py              MTP (nextn) speculative decode: draft, verify, accept
python/freetoken/models/gguf/
  arch_qwen35moe.py   the declarative GGUF mapping for the qwen35moe architecture
  dequant.py          K-quant -> torch tensors (the host-dequant path)
  packed.py           K-quant -> packed slabs (the in-shader unpack path)
```

## Requirements

| Need | Detail |
| --- | --- |
| GPU | AMD RDNA1 or RDNA2 with a Vulkan 1.2+ driver (RADV, or the Windows Vulkan driver) |
| Loader | `libvulkan.so` plus the vendor ICD (`vulkan-radeon` on Arch) |
| Headers | `vulkan-headers` (`/usr/include/vulkan/vulkan.h`) |
| Shader compiler | `glslc` (from `shaderc`), used to turn `shaders/*.comp` into SPIR-V |
| Python | 3.12 with a CPU-only PyTorch is enough; the Vulkan path never imports CUDA |

No CUDA, ROCm, Triton or NCCL is required on this path.

## Build

The extension is built on demand. `freetoken.kernel.vulkan` resolves it in
this order:

1. an in-tree shared object for the running interpreter,
2. a JIT build through `torch.utils.cpp_extension`,
3. a direct `c++` invocation (the recipe below).

```bash
# direct build (about 25 s)
V=/path/to/venv
TORCH_INC=$V/lib/python3.12/site-packages/torch/include
PYINC=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
c++ -std=c++20 -O2 -fPIC -DTORCH_EXTENSION_NAME=freetoken_vulkan_ext \
   -DTORCH_API_INCLUDE_EXTENSION_H -I/usr/include -I<pkgdir> \
   -I"$TORCH_INC" -I"$TORCH_INC/torch/csrc/api/include" -I"$PYINC" \
   -c ext_module.cpp -o ext.o
c++ -shared -o freetoken_vulkan_ext.so ext.o -lvulkan \
   -L$V/lib/python3.12/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python
```

To bake it into an install instead, set `FREETOKEN_BUILD_VULKAN=1` before
`pip install -e .`; `setup.py` then adds `freetoken.kernel.vulkan._ext`. That
build path needs the torch version upstream pins (`torch>=2.14,<2.15`), and
`FREETOKEN_BUILD_VULKAN=1` on a machine without `CUDA_HOME` builds only the
Vulkan extension: the CUDA and ROCm extensions are skipped, because they link
against a CUDA or ROCm runtime that an AMD box does not have.

The JIT path (the default at runtime) has no such constraint: it works with
any torch that provides the C++ headers.

Shaders are compiled to SPIR-V with:

```bash
glslc --target-env=vulkan1.2 -fshader-stage=compute shaders/X.comp -o X.comp.spv
```

The runtime looks for the SPIR-V directory in `FREETOKEN_VULKAN_SHADERS`, and
falls back to its own compiled cache under `~/.cache/freetoken/vulkan/shaders`.

## Run

```bash
export FREETOKEN_VULKAN_SHADERS=/path/to/shaders
python -m freetoken.models.qwen3_5_moe.vulkan.cli \
    --model /path/to/checkpoint --prompt "hello" --max-tokens 32
```

### Serve a GGUF

The same CLI serves a K-quant GGUF directly - no safetensors, no conversion.
On this machine the reference file is the Ornith-1.5-35B-A3B "CyberTiel"
calibrated GGUF (41 blocks: 40 real + 1 MTP, 256 experts, top-8, Q5_K experts
on blocks 0-14 and Q4_K on 15-40):

```bash
python -m freetoken.models.qwen3_5_moe.vulkan.cli \
    --gguf /path/to/CyberTiel.gguf --prompt "Name three primary colors." \
    --max-tokens 32 --temperature 0
```

Experts are served through the **packed K-quant shaders** by default: the GGUF
bytes are uploaded verbatim and unpacked inside the compute shader, so no
weight is dequantized on the host and a Q4_K expert costs 0.5625 MiB of VRAM
instead of 2.0 MiB as BF16. `FTV_GGUF_PACKED=0` forces the old host-dequant
BF16 path.

Measured on the RX 5700 XT (8 GB, RADV, loadavg ~10, decode only, 8 tokens
after a 4-token warm-up, `tests/vulkan/bench_gguf_packed.py`):

| Path | decode s/tok | decode tok/s |
| --- | --- | --- |
| packed (default) | **1.309** | **0.764** |
| host-dequant fallback (`FTV_GGUF_PACKED=0`) | 29.7 | 0.034 |

**22x the fallback, and the same token ids on the identity prompt**
(`[5844, 11, 13358, 11]` for `--max-tokens 4 --seed 1234` through both paths).
VRAM ends at 7924 MiB resident - lower it with `--cache-mb` if the card has
other tenants. That is still ~21x short of llama.cpp's 16-17.9 tok/s on the
same file: the remaining cost is one synchronous submit+fence+readback per
expert GEMV (~3900 per token), not the weights. The next lever is wrapping the
8 routed experts of a layer in one `batch_begin`/`batch_submit` and using the
fused `gateup` + `acc` shaders - see the last-mile notes in
[kquant-kernels.md](kquant-kernels.md).

MTP speculative decode works on top of the packed path:
`--mtp --mtp-p-min 0.25 --mtp-n-max 3` measured 0.83 tok/s against the 0.51 tok/s
baseline of the same run (whole `generate()`, prefill included), token-identical
to the non-speculative greedy path. `n_max 5` measured slower (0.37-0.54 tok/s),
which is what the fork reports for this model too. The absolute numbers move
with box load - the non-speculative baseline ran at 0.51, 0.60 and 0.70 tok/s
on three separate invocations - so always compare inside one run. See
[mtp.md](mtp.md).

### Environment knobs

| Variable | Meaning |
| --- | --- |
| `FREETOKEN_VULKAN_SHADERS` | directory holding the compiled `.spv` files |
| `FTV_EXP_CACHE_MB` | VRAM budget for the expert LRU cache (default 2048) |
| `FTV_HOST_PACK_MB` | host-side packed-expert cache (default 256) |
| `FTV_NVFP4_GPU` | `1` uses the NVFP4 GPU kernel, `0` dequantizes on the host to BF16 |
| `FTV_GGUF_PACKED` | `1` (default) serves GGUF experts through the packed K-quant shaders, `0` dequantizes them on the host to BF16 |
| `FTV_NVFP4_M1` | enable the Wave64 single-wavefront NVFP4 kernel (nonzero number) |
| `FTV_GPU_ROUTER` | run the router and top-k on the GPU |
| `FTV_EXP_ARENA` | fixed-slot per-layer expert arena (fused 8-expert MoE dispatch) |
| `FTV_SPEC_PACK` | speculative host pre-pack daemon for the next token's experts |

## Hardware notes (RDNA1 / gfx1010)

- **No native FP4 or FP8.** NVFP4 has to be unpacked to FP16 in the shader,
  so the ALU chain, not the memory system, sets the floor.
- **Wave64 only.** Shuffle-heavy kernels are written for 64-wide waves; there
  is no Wave32 path on this generation.
- **Shared-memory multi-writer is unsafe here.** A barrier reduction over
  shared arrays produced all-NaN output on RADV/Navi10 even when every input
  was finite. Per-lane recompute, or a single-writer reduction, is the safe
  pattern.
- **Host staging buffers cannot stay mapped** for the process lifetime on this
  driver; a persistent mapped staging ring cost a full-range flush per submit
  and was reverted.

## Correctness gates

`tests/vulkan/` holds the ported gates. They are the reference for any change
to the kernels or the decode loop:

| Test | Gate |
| --- | --- |
| `test_batched_fork2.py` | same-state fork: sequential vs batched max diff <= 5.5e-06, batched vs batched **bit-exact (0.0)**, identical argmax |
| `test_gdn_kernel.py` | GPU GDN conv + recurrence vs the host reference (1e-07) |
| `test_gpu_router.py` | GPU router: per-layer h2 <= 1e-5, paired (id, renormalized scale) <= 1e-6, >= 50-token argmax, sentinel contract |
| `test_topk_parity.py` | top-k 256x8 on the GPU vs torch (<= 1e-6) |
| `test_qwen35_{attn,moe,layer}_parity.py` | layer-level parity |
| `test_batched_chunked.py` | 30-token chunked generation matches |

When bisecting a layer divergence, restore the pristine host state
(`gdn_state`, `conv_buf`, `k_cache`, `v_cache`, `_seq_len`) at the **top** of
every loop iteration, before both the sequential and the batched run.
Restoring only between runs inside the iteration lets each sequential run
consume the previous batched output and reports a false divergence at layer 1.

## Further reading

- [porting-guide.md](porting-guide.md) - what came from the old fork and why
  the Vulkan path is a separate runtime.
- [cybertiel-gguf-mapping.md](cybertiel-gguf-mapping.md) - the GGUF mapping
  for the Ornith-1.5-35B-A3B "CyberTiel" checkpoint.
- [kquant-kernels.md](kquant-kernels.md) - the packed-expert GEMV shaders:
  contract, packed provider, measured numbers, wiring recipe.
- [mtp.md](mtp.md) - MTP (nextn) speculative decode: draft/verify/accept,
  gates, measured results.
