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

Decode walks the layers through the **batched command-buffer path** by
default (`FTV_GGUF_BATCHED=1`): one command buffer per layer carries the
dense projections, the 8 routed experts (fused gateup-silu + accumulate-down
via `batch_moe_kquant_record`) and the readbacks, so a token costs ~91
submits instead of a submit per matrix. The path is **identity-verified**
against the sequential decoder - the hazard-flag fix in `batch_kquant_io`
(Y is WRITE for every `kind`; `plain` and `gateup` used to declare it READ
while the shaders wrote it) is what made it correct - and
`tests/vulkan/test_batched_fork2.py` measures seq-vs-batched maxdiff
**4.530e-06** (gate 1e-5), batched-vs-batched2 **bit-exact 0.0**, identical
argmax, and the reference token ids on the identity prompt.
`FTV_GGUF_BATCHED=0` restores the sequential per-matrix decoder (and pins
every fp16 weight at its native dtype).

Measured on the RX 5700 XT (8 GB, RADV, loadavg ~2-5, page cache warm,
decode only, 8 tokens after a 6-token per-path warm-up; sequential and
batched arms run back to back inside one process, median of 3 rounds,
`FTV_EXP_CACHE_MB` at the shipped 2048):

| Path | decode ms/tok | decode tok/s | sync round trips/tok | submits/tok |
| --- | --- | --- | --- | --- |
| batched (default) | **110.3-158.2** | **6.3-9.1** | 222-325 | 91 |
| sequential (`FTV_GGUF_BATCHED=0`) | 462.5-528.2 | 1.9-2.2 | 4246-4338 | 0 |

**3.0-4.3x**, consistent with the 3.43x (149.6 vs 513 ms/tok) of the run
that promoted the path; the spread is page-cache and box-load noise on a
shared machine. "Sync round trips" is the engine's per-call synchronous
stat counter (upload/kernel/download per expert miss and per dense pin);
the batched path moves almost all of them into the 91 submits.

VRAM is the reason the expert-cache default is 2048: with
`FTV_EXP_CACHE_MB=4096` the run saturates the card at **8167 of 8176 MiB**
(over the driver's own 7434 MiB budget), and RADV silently spills the
overflow into system memory; at 2048 it saturates at **~6998 MiB**, ~1.1 GiB
of headroom, which also covers `--max-len 8192` (the KV caches are host
RAM, so context length does not move VRAM). RSS peaks near 15 GiB, almost
all of it the mmap'd 22.8 GB GGUF's page cache - reclaimable, never a
private copy. Raise the budget with `--cache-mb` only when the card is
otherwise idle, and watch `/sys/class/drm/card1/device/mem_info_vram_used`.

The host-dequant fallback (`FTV_GGUF_PACKED=0`) is unchanged and still
produces the same token ids (`[5844, 11, 13358, 11]` for `--max-tokens 4
--seed 1234` through both weight paths) at ~29.7 s/tok - **~22x slower than
the packed sequential path** and ~700x slower than the batched one; it is
kept as the byte-for-byte reference the gates compare against. llama.cpp's
Vulkan build on the same file runs 16.4-17.9 tok/s, so the batched path is
now 1.5-2.5x short of it rather than ~21x - the remaining cost is per-layer
submit/fence overhead, not the weights (see
[kquant-kernels.md](kquant-kernels.md)).

**MTP speculative decode is token-identical but still slower than not
drafting.** `--mtp --mtp-p-min 0.25 --mtp-n-max 3` stays token-identical to
the non-speculative greedy path with `FTV_GGUF_BATCHED=1`. Its verify now
runs the batched multi-row walk (`decode_rows_batched`, one command buffer
per layer phase for all verify rows), which took it from 0.34x to **0.56x**
of the batched baseline (3.28 vs 5.88 tok/s, median of 3 interleaved
rounds, identity 3/3) - verify costs ~1.07x a baseline token per row, but a
round verifies `drafted + 1` rows to commit `accepted + 1` tokens, the
MoE phase must submit per row (an engine multi-row hazard bug, see
[mtp.md](mtp.md)), and the rollback snapshot adds ~190 ms/round. Use it
for correctness work, not speed, until the engine's multi-row MoE batch is
fixed; the win condition and the measured numbers are in
[mtp.md](mtp.md).

### Environment knobs

| Variable | Meaning |
| --- | --- |
| `FREETOKEN_VULKAN_SHADERS` | directory holding the compiled `.spv` files |
| `FTV_GGUF_BATCHED` | `1` (default) batched command-buffer decode, `0` sequential per-matrix decode (identity-verified either way) |
| `FTV_GGUF_PACKED` | `1` (default) serves GGUF experts through the packed K-quant shaders, `0` dequantizes them on the host to BF16 |
| `FTV_EXP_CACHE_MB` | VRAM budget for the expert LRU cache (code-wide default 4096; `ResidentGGUF` defaults it to **2048** when unset - 4096 overcommits an 8 GB card) |
| `FTV_HOST_PACK_MB` | host-side packed-expert cache (default 256) |
| `FTV_GGUF_DEQUANT_CACHE_MB` | host cache for dequantized experts on the `FTV_GGUF_PACKED=0` fallback (default 256) |
| `FTV_STAGING_MAP` | `1` (default) persistent mapping of host-visible buffers, `0` map/unmap per access, `2` persistent except one-shot sync staging |
| `FTV_GGUF_HOTSET` | `0` (default) pin the static expert hot set; identity-correct but **not** a speed win (see [kquant-kernels.md](kquant-kernels.md)) |
| `FTV_NVFP4_GPU` | `1` uses the NVFP4 GPU kernel, `0` dequantizes on the host to BF16 |
| `FTV_NVFP4_M1` | enable the Wave64 single-wavefront NVFP4 kernel (nonzero number) |
| `FTV_GPU_ROUTER` | run the router and top-k on the GPU |
| `FTV_EXP_ARENA` | fixed-slot per-layer expert arena (fused 8-expert MoE dispatch; NVFP4-packed only, forced off for GGUF) |
| `FTV_SPEC_PACK` | speculative host pre-pack daemon for the next token's experts (NVFP4-packed only, forced off for GGUF) |

## Hardware notes (RDNA1 / gfx1010)

- **No native FP4 or FP8.** NVFP4 has to be unpacked to FP16 in the shader,
  so the ALU chain, not the memory system, sets the floor.
- **Wave64 only.** Shuffle-heavy kernels are written for 64-wide waves; there
  is no Wave32 path on this generation.
- **Shared-memory multi-writer is unsafe here.** A barrier reduction over
  shared arrays produced all-NaN output on RADV/Navi10 even when every input
  was finite. Per-lane recompute, or a single-writer reduction, is the safe
  pattern.
- **Host-visible buffers are mapped once and stay mapped** (`FTV_STAGING_MAP=1`,
  the default): per-upload `vkMapMemory`/`vkUnmapMemory` costs ~130 us/MB of
  driver page-table work on this driver and capped the staging path at ~4-5
  GB/s. Coherence is handled by exact-range flushes that fire only when the
  memory type is non-coherent - RADV/Navi10 exposes none (all six host-visible
  types are HOST_COHERENT), so here they are spec-mandated no-ops and the win
  is deleting the map/unmap pair (11.5-12.5 GB/s batched/async at 1.77 MB,
  sync unchanged at ~4.2 GB/s; see [kquant-kernels.md](kquant-kernels.md)).
  An always-mapped staging *ring* that published N+1 during N was tried and
  reverted - it cost a full-range flush per submit on this driver.

## Correctness gates

`tests/vulkan/` holds the ported gates. They are the reference for any change
to the kernels or the decode loop:

| Test | Gate |
| --- | --- |
| `test_batched_fork2.py` | same-state fork: sequential vs batched max diff <= 1e-5 (measured 4.530e-06), batched vs batched **bit-exact (0.0)**, identical argmax; the GGUF variant runs the real `decode_step_batched` |
| `test_gdn_kernel.py` | GPU GDN conv + recurrence vs the host reference (1e-07) |
| `test_gguf_batched_synthetic.py` | K-quant batch recorders on synthetic slabs vs the host reference |
| `test_gguf_parity.py` | GGUF decode vs the host-dequant reference (opt-in via `FREETOKEN_TEST_MODEL_GGUF`) |
| `test_hotset_resident.py` | hot-set pin is token-identical to the plain LRU |
| `test_mtp.py` | MTP draft/verify/accept is token-identical to the greedy path |
| `test_qwen35_moe_parity.py` | MoE layer parity (safetensors checkpoint) |
| `test_resident_moe.py` | gpt-oss resident MoE |
| `test_ressl_nvfp4_kernel.py` | NVFP4 kernel parity |
| `test_vulkan_smoke.py` | rmsnorm / resident linear / batch round trip |

`tests/kernel/vulkan/` holds the shader-level gates (including the K-quant
goldens) and `tests/gguf_vulkan/` the packed-provider and mapping gates; the
model-dependent ones need `FREETOKEN_TEST_MODEL_GGUF=<path>`. Full run on
this box (2026-10-21): **120 passed, 5 skipped in 465 s**.

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
