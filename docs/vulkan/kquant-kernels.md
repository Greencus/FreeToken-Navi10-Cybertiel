# K-quant GEMV shaders (packed experts, unpack in the shader)

The CyberTiel GGUF stores the MoE experts as Q5_K (blocks 0-14) and Q4_K
(blocks 15-40), with Q8_0 for the dense tensors. The Vulkan decode loop has no
K-quant path of its own, so it serves that file through host dequant: every
routed expert is dequantized on the CPU (`freetoken.models.gguf.dequant`),
materialized as a BF16 tensor, and uploaded - work on every cache miss, and
the BF16 slab is twice the packed bytes (Q4_K: 0.5625 vs 2 bytes per weight).

These shaders remove both costs: the packed GGUF bytes are uploaded verbatim
and unpacked inside the compute shader, so a weight is never materialized
outside the file except as the file's own bytes in VRAM.

**Status: wired into the decode loop and measured end-to-end.**
`python/freetoken/models/qwen3_5_moe/vulkan/gguf_resident.py` serves experts
through the packed shaders by default; `FTV_GGUF_PACKED=0` falls back to the
old host-dequant BF16 path. The two paths produce **identical token ids** on
the greedy identity prompt (see [Wired into the decode loop](#wired-into-the-decode-loop)).

## What is in the tree

| Piece | Where |
| --- | --- |
| 7 GLSL compute shaders | `python/freetoken/kernel/vulkan/shaders/linear_q{4k,5k}_*_f32.comp`, `linear_q8_0_resident_f32.comp` |
| Push constant + dispatch | `ext_module.cpp` (`struct KQuantPC`, `linear_kquant_resident_io`, `batch_kquant_io`, `batch_moe_kquant_record`) |
| Packed expert provider | `python/freetoken/models/gguf/packed.py` |
| Gates | `tests/kernel/vulkan/test_kquant_shaders.py`, `tests/gguf_vulkan/test_gguf_packed.py` |

Shaders (52 `.comp` in total; compile with `python -m freetoken.kernel.vulkan.build`
or `compile_shaders()`, output goes to `~/.cache/freetoken/vulkan/shaders`):

| Shader | kind | writes |
| --- | --- | --- |
| `linear_q4k_resident_f32.comp` | `resident` | `Y[T,N] = X[T,K] @ W[N,K]^T` (assignment) |
| `linear_q4k_gateup_silu_f32.comp` | `gateup` | `Y[T,N] = silu(gate[n]) * up[n]`, one call |
| `linear_q4k_acc_resident_f32.comp` | `acc` | `Y[T,N] += ...` (accumulate) |
| `linear_q5k_*` (same three) | | Q5_K unpack, adds the `qh` bit matrix |
| `linear_q8_0_resident_f32.comp` | `resident` | dense-tensor GEMV only |

There is deliberately no `gateup`/`acc` Q8_0 variant: in this file Q8_0
covers dense tensors, not expert matrices, so the plain GEMV is all it needs.

## Shader contract

Three std430 buffers, nothing else:

```
binding 0  QBuf   packed W rows (GGUF bytes verbatim)
binding 1  XBuf   fp32 input row(s)
binding 2  YBuf   fp32 output
```

and one push constant block (`ext_module.cpp`, mirrored in every shader):

```cpp
struct KQuantPC {
    uint32_t T, N, K, wpr, ng32;
    float gscale;
    uint32_t slot0, rowstride;
};
```

| Field | Meaning |
| --- | --- |
| `T, N, K` | output rows, output columns, packed input width |
| `wpr` | packed words per row (K-quant: `K/8` qs words; Q8_0: `K/8` quarter-block chunks) |
| `ng32` | one packed block, in `wpr` units: words per block for the K-quants (Q4_K 36, Q5_K 44), quarter-block chunks per 32-weight block for Q8_0 (4) |
| `gscale` | multiplies the whole row dot before it is written (see below) |
| `slot0` | **row** offset of this matrix inside `QBuf`, never an offset into `YBuf` |
| `rowstride` | packed stride of one W row, in words |

`kind` selects the write:

- `resident` - `Y[t*N+n] = gscale * dot(X[t], W[slot0+n])`
- `gateup` - rows `slot0 .. slot0+N-1` are gate, `slot0+N .. slot0+2N-1` are
  up (the layout `packed_gate_up_expert` returns); writes
  `Y[t*N+n] = silu(gate) * up`. `N` here is the bundle's output width.
- `acc` - `Y[t*N+n] += gscale * dot(...)`, so N sequential expert dispatches
  into the same `Y` accumulate into one buffer. The engine clears that buffer
  once per layer with `batch_fill` and folds each expert's renormalized router
  scale into `gscale`; the ggml per-superblock scales live in the packed bytes
  and are read by the shader.

Width constraints come from the packed block granularity: `K` a multiple of
**256** for Q4_K/Q5_K and **64** for Q8_0; for `gateup`/`acc`, `N` a multiple
of 256 as well (it is counted in K-quant blocks). See each shader's header
comment - it carries the byte layout of its block type.

Q8_0 rows are 34 bytes per 32 weights, so the shader rebuilds unaligned
32-bit windows and can read **at most one word past a row**: keep `>= 4`
bytes of slack after a packed Q8_0 slab in the resident buffer. The K-quant
rows (144/176 bytes) are 4-byte aligned and need no slack.

## Host side: `freetoken.models.gguf.packed`

```python
from freetoken.models.gguf.reader import gguf_tensor_index
from freetoken.models.gguf.packed import (
    kquant_qtype, packed_expert_info, packed_expert,
    packed_gate_up_expert, packed_expert_bytes, packed_tensor,
)

index = gguf_tensor_index("/path/to/CyberTiel.gguf")   # metadata only, mmap
```

- `kquant_qtype(ggml_type)` -> `"q4_k" | "q5_k" | "q8_0"`, else `None` for
  types these shaders do not handle.
- `packed_expert_info(index, tensor, expert)` -> `PackedExpert`
  (`qtype`, `rows`, `cols` (the GEMV's K), `row_bytes`, `nbytes`, `experts`,
  `aligned`) - metadata only, no copy; raises for a type with no unpacker.
- `packed_expert(index, tensor, expert)` -> `(rows, row_bytes)` uint8 slab,
  a zero-copy view into the mmap'd file. Upload it, drop it; the model is
  never held in RAM.
- `packed_gate_up_expert(index, layer, expert)` -> `(2N, row_bytes)` with gate
  rows first then up rows, i.e. the `slot0` / `slot0+N` bundle layout the
  `gateup` shader expects (falls back to the gate rows alone for M-shaped
  MoE, which has no up tensor).
- `packed_expert_bytes(index, tensor, expert)` -> slab size, to budget the
  arena.
- `packed_tensor(slab)` -> contiguous, writable uint8 torch tensor (an mmap
  view is read-only, torch would warn) ready for `ext.upload_resident`.

Slab bytes per expert matrix (512x2048, the shape this file uses): Q4_K
0.5625 MiB, Q5_K 0.6875 MiB, against 2.0 MiB as BF16. A decode step fetches
`8 experts x 3 matrices`: **13.5 MiB packed vs 27 MiB** as BF16 over the Q4_K
half of the file (16.5 MiB over the Q5_K half).

## Dispatch recipe (what the wiring needs)

```python
# plain GEMV, sync path
ext.linear_kquant_resident_io(qtype, "resident", h_x, h_w, h_y, T, N, K,
                              1.0, slot0)          # or the named wrapper
# inside an open batch
ext.batch_kquant_io(qtype, "acc", h_w, h_x, h_y, T, N, K, router_scale, slot0)
# one pybind call for a whole layer: fused gateup per expert + accumulated down
ext.batch_moe_kquant_record(h_x, h_acc, qtype,
                            h_slots,            # per-expert silu output buffers
                            gu_handles, gu_scales, N, K,
                            dn_handles, dn_scales, dn_rows, N,
                            gu_slot0, dn_slot0)
```

`batch_moe_kquant_record` is the fused form the decode loop should use (one
Python crossing per layer instead of one per expert); its contract is pinned
by `test_batch_moe_record_matches_batch_kquant_io`, which requires it to
reproduce the explicit `batch_kquant_io` calls bit-for-bit in tolerance.

## Wired into the decode loop

Two files carry the wiring, and nothing else:

| Piece | Change |
| --- | --- |
| `gguf_provider.py` | `ExpertBank` gained `qtype` / `row_bytes` / `packed_bytes` (keyword-only, default `None`/0, so the existing positional construction still works); `GGUFProvider.expert_bank()` sets them from `kquant_qtype(ggml_type)`, and `expert_packed(layer, proj, expert)` returns the slab |
| `gguf_resident.py` | `_packed_qtype()` decides the lane, `_expert_handle()` uploads the packed slab on a miss (or `expert_bf16` when the lane is off), `_expert_affine{,_batch}()` dispatch `linear_q{4k,5k,8_0}_resident_io` instead of `linear_bf16_resident_io` |

`FTV_GGUF_PACKED` (default `1`) selects the lane; `0` forces the old
host-dequant BF16 path, unchanged. Both reach the experts only through
`model.expert_mats` + `_moe_step` / `_moe_chunk`, which is also how the MTP
head reads them, so the speculative path picks the packed lane up for free.

Only the plain `resident` shader is used today: each routed expert is still
three synchronous round trips (upload the token row, GEMV, download the
result). The fused `gateup` + `acc` shaders, `batch_kquant_io` and
`batch_moe_kquant_record` are built, shader-tested and **not yet wired** -
they are the remaining lever (see the last-mile notes below).

### Measured end-to-end (RX 5700 XT, 41-block CyberTiel GGUF)

`python tests/vulkan/bench_gguf_packed.py --tokens 8 --warmup 4` times the
decode loop only (prefill is measured separately); `tests/vulkan/bench_mtp.py`
times whole `generate()` calls including the 14-token prefill.

| Path | decode s/tok | decode tok/s | host dequant |
| --- | --- | --- | --- |
| packed (`FTV_GGUF_PACKED=1`, default) | **1.309** | **0.764** | none |
| host-dequant fallback (`FTV_GGUF_PACKED=0`) | 29.7 | 0.034 | every cache miss |

That is **22x** the fallback. Prefill of the 14-token prompt: 16.1 s cold
(the expert LRU is empty, so it pays the miss path) - the packed lane removes
the dequant from that path too. VRAM at the end of the packed run: 7924 MiB
resident (`--cache-mb` lowers it; the pinned dense tensors alone are 3741 MiB);
RSS peak 10.4 GiB, most of which is the mmap'd file.

The `report_stats()` split for the packed run says it outright: `exp_packed`
6.04 s / 6879 calls (slab view) and `exp_up` 7.24 s / 6879 calls (upload) are
present, and **`exp_dequant` does not appear at all**. Per expert affine:
0.317 ms token upload + 0.164 ms GEMV + 0.159 ms result download, i.e. the
cost is now the ~3900 synchronous round trips per token, not the weights.

**Token identity.** The same prompt through both paths:

```
--prompt "Name three primary colors." --max-tokens 4 --temperature 0 --seed 1234
FTV_GGUF_PACKED=1  ->  token_ids: [5844, 11, 13358, 11]   "Red, yellow,"
FTV_GGUF_PACKED=0  ->  token_ids: [5844, 11, 13358, 11]   "Red, yellow,"
```

identical. This is the expected outcome rather than a coincidence: the shader
unpacks to fp32 and the fallback dequantizes to BF16, so the two differ in the
last mantissa bits of every weight (~2^-8) - the greedy argmax has to be
robust to that, and on this prompt it is. Treat the identity as a gate, not as
a proof of bit-equality.

### Last mile (not done)

llama.cpp runs the same file at 16.4-17.9 tok/s, so the packed path is still
~21x short, and the gap is dispatch, not math: the GPU does ~0.16 s/step of
GEMV work while the wall clock is 1.31 s. Closing it means (in order):

1. Wrap the 8 routed experts of `_moe_step` in one `batch_begin` /
   `batch_submit` (`batch_upload`, `batch_kquant_io`, `batch_download`),
   collapsing 24 round trips per layer into 1 submit + 8 readbacks. The batch
   engine is proven bit-exact by the fork (`bat-vs-bat2 = 0.0`) and by
   `test_batch_acc_sums_experts`.
2. Bundle gate+up per expert with `packed_gate_up_expert` and dispatch the
   `gateup` shader, then feed the resident silu slot straight into the `acc`
   kernel for the down GEMV - two dispatches per expert, no host silu, no
   second upload.
3. Only after those: a persistent arena for expert slots, as the NVFP4 path
   already does.

Estimated floor with (1)+(2) done: ~0.5-0.6 s/tok. Do not claim more than that
without measuring.

## Measured (RX 5700 XT, RADV, `glslc --target-env=vulkan1.2`)

All numbers below are microbenchmarks. The end-to-end tok/s the wired path
actually delivers is in [Wired into the decode loop](#wired-into-the-decode-loop).

Correctness - golden dequant is `freetoken.third_party.gguf.quants`, which
llama.cpp's C dequantizers match bit-for-bit (checked: 0.0 difference):

| Case | worst abs diff vs numpy dequant |
| --- | --- |
| synthetic Q4_K, 512x2048 | 1.2e-4 |
| synthetic Q5_K, 512x2048 | 2.5e-4 |
| synthetic Q8_0 | 2.3e-5 |
| fused `gateup` (q4_k/q5_k) | 4.3e-2 (rel ~8.6e-7) |
| `acc` x3 experts | 4.3e-4 |
| **real `blk.0.ffn_gate_exps.weight` (Q5_K, expert 7)** | **1.9e-8 abs / 3.2e-7 rel** |

The gate tolerance is `5e-4 * max|ref| + 1e-4` (fp32 accumulation over K
products carries ~K*eps of rounding); a wrong scale index alone shifts output
by percent-level amounts, so the bound is not loose in practice.

Bandwidth, resident upload + submit (medians):

| copy | wall |
| --- | --- |
| sustained H2D at 4/8/16 MiB | 5.4 GB/s |
| 0.5625 MiB (one Q4_K expert matrix) | 0.164 ms |
| 2.25 MiB (4 matrices) | 0.505 ms |
| 9 MiB (16 matrices) | 1.907 ms |

At that rate the packed 13.5 MiB/token costs about 2.5 ms of upload against
5.1 ms for 27 MiB of BF16 - and, unlike the host path, the unpack work is on
the GPU and the CPU is free to run the rest of the step.

## Gates

```bash
cd FreeToken-Navi10-Cybertiel
PYTHONPATH=python python -m pytest tests/kernel/vulkan -q                    # no checkpoint
PYTHONPATH=python FREETOKEN_TEST_MODEL_GGUF=<path> \
  python -m pytest tests/kernel/vulkan/test_kquant_shaders.py -q             # + real expert
PYTHONPATH=python FREETOKEN_TEST_MODEL_GGUF=<path> \
  python -m pytest tests/gguf_vulkan -q                                      # packed provider
```

Measured on this box (interpreter `/home/cus/.venv-vllm-vulkan/bin/python`):

| Command | Result |
| --- | --- |
| `pytest tests/kernel/vulkan -q` (no checkpoint) | **29 passed, 1 skipped** in 4.06 s |
| `FREETOKEN_TEST_MODEL_GGUF=<path> pytest tests/kernel/vulkan/test_kquant_shaders.py -q` | **24 passed** in 12.61 s (adds the real `blk.0` expert) |
| `FREETOKEN_TEST_MODEL_GGUF=<path> pytest tests/gguf_vulkan -q` | **45 passed, 1 skipped** in 18.87 s |
| `pytest tests/gguf_vulkan -q` (no checkpoint) | 18 passed, 28 skipped |

Every GPU test skips when no Vulkan device is present. Regenerate the binding
stubs after touching `ext_module.cpp`:

```bash
PYTHONPATH=python python -m freetoken.kernel.vulkan._gen_stub   # 82 functions
```
