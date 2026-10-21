# Porting guide: the old Vulkan fork into upstream FreeToken

This records what moved, where it landed, and the decisions taken on the way.
The source of the port is the standalone fork
`/home/cus/working/FreeTokenVulkan5700xt/freetoken-vulkan`, which is many
releases behind upstream and is now read-only reference.

## Why a separate runtime instead of a kernel drop-in

Upstream's engine is irreducibly CUDA-bound:

| Site | Assumption |
| --- | --- |
| `engine/engine.py` | `assert torch.cuda.current_stream() == self.stream` before every forward; CUDA graphs in `engine/graph.py`; `torch.cuda.mem_get_info` for the cache budget |
| `scheduler/scheduler.py` | `torch.cuda.Stream`, `torch.cuda.synchronize`, `torch.cuda.memory_reserved` |
| `engine/sample.py`, `utils/torch_utils.py` | `torch.cuda.nvtx.range` |
| `gpu_select.py` | `torch.cuda.device_count`, NVML, `nvidia-smi` |

Upstream's *model* packages are backend-agnostic (plain torch ops plus Triton
kernels behind `quant_method.apply()` and `ctx.attn_backend.forward()`), so
individual kernels could be replaced - a Vulkan `LinearKernel` or
`MoEKernel` could be appended to the candidate tables in
`layers/quantization/linear/nvfp4.py` and `layers/quantization/moe/nvfp4.py`.
That path only works on a machine that also has a working CUDA/ROCm stack for
everything else (attention, GDN, norms, KV pools, graph capture).

The ported engine is a complete decoder, not a set of kernel replacements, so
it was brought over as its own runtime. The CUDA path is untouched, and the
two can coexist in one tree.

## File mapping

| Old fork | New location |
| --- | --- |
| `python/ext_module.cpp` (3088 lines) | `python/freetoken/kernel/vulkan/ext_module.cpp` |
| `include/vk_util.hpp` (569 lines) | `python/freetoken/kernel/vulkan/vk_util.hpp` |
| `shaders/*.comp` (45 files) | `python/freetoken/kernel/vulkan/shaders/` |
| `python/qwen35/{config,loader,layer,moe,attention,linear_attn,resident}.py` | `python/freetoken/models/qwen3_5_moe/vulkan/` |
| `python/qwen35_nvfp4/{nvfp4,resident}.py` | `python/freetoken/models/qwen3_5_moe/vulkan/{nvfp4,nvfp4_resident}.py` |
| `python/freetoken_vulkan.py` | `python/freetoken/models/qwen3_5_moe/vulkan/engine.py` |
| `python/chat_ressl.py`, `python/chat_qwen35.py` | `python/freetoken/models/qwen3_5_moe/vulkan/cli.py` |
| `python/test_*.py` (the parity gates) | `tests/vulkan/` |
| `python/build_ext.sh`, `python/setup_ext_linux.py` | `python/freetoken/kernel/vulkan/_build.py`, `build.py` |
| `CMakeLists.txt` (CLI demo targets) | dropped; the Python extension never used it |
| `src/*.cpp` (standalone kernel CLI demos) | dropped; phase-1 scaffolding, not part of the engine |
| `docs/*.md` (gpt-oss, Qwen3-4B/8B/14B notes) | superseded; only the Vulkan-specific parts were kept |

## What was carried over unchanged

- Every kernel's math. The shaders, the batch engine, the expert arena, the
  GPU router and the async upload queue are byte-for-byte ports; the only
  edits are include paths and the module import.
- The environment knobs (`FTV_*`), so a configuration that worked in the fork
  still works here.
- The correctness gates and their thresholds.

## What changed

- **Import surface.** `import freetoken_vulkan_ext` became
  `from freetoken.kernel.vulkan import ext`, resolved lazily through
  `freetoken.models.qwen3_5_moe.vulkan._runtime`, so importing the package
  never fails on a machine without Vulkan.
- **Shader resolution.** The fork read `FREETOKEN_VULKAN_SHADERS` with a stale
  Windows fallback path. The port resolves the directory from the package's
  own compiled cache when the variable is unset.
- **Build.** The fork built the extension with a hand-written `build_ext.sh`
  against a Python 3.14 torch that no longer exists on this machine. The port
  derives the include and library paths from the running interpreter and can
  also build through `torch.utils.cpp_extension`, or as a `setup.py`
  extension with `FREETOKEN_BUILD_VULKAN=1`.
- **Stubs.** The fork's `freetoken_vulkan_ext.pyi` was stale (it predated the
  fused-MoE, GPU-router and async-upload bindings). The port's stub is
  regenerated from the actual pybind module.

## Behaviour worth knowing

- The **expert arena** (`FTV_EXP_ARENA`) removes per-dispatch descriptor
  churn but adds write-after-write barriers on the shared per-layer buffers
  (~14% more barrier traffic). It is a trade, not a free win.
- The **GPU router** (`FTV_GPU_ROUTER`) moves the router GEMM and the top-k
  onto the GPU. The host still computes the layer input, so the host and the
  GPU agree on the router output only within the gate's tolerance; the
  `test_gpu_router.py` gates bound that.
- `FTV_NVFP4_M1` only affects the batched decode path. The one-shot sync path
  uses the other NVFP4 shaders, so a mixed prefill+decode session splits
  numerics between the two paths (~1e-4). That is expected, not a bug.
- `FTV_NVFP4_M1` is parsed with `atoi`, so a bare `"true"` leaves the kernel
  off. Only a nonzero number enables it.

## Next steps

1. Serve the CyberTiel GGUF on this path (K-quants are dequantized on the
   host to BF16; see [cybertiel-gguf-mapping.md](cybertiel-gguf-mapping.md)).
2. Measure the GGUF path against the NVFP4 path and decide whether a
   K-quant GEMV shader is worth writing for RDNA1.
3. Re-run the benchmark battery (A/B/C in the project notes) on a quiet
   machine: the numbers in the project notes were measured on a loaded box.
