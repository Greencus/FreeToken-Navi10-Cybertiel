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

**Status: wired into the decode loop and measured end-to-end, batched.**
`python/freetoken/models/qwen3_5_moe/vulkan/gguf_resident.py` serves experts
through the packed shaders by default; `FTV_GGUF_PACKED=0` falls back to the
old host-dequant BF16 path. Decode itself runs the **batched command-buffer
path by default** (`FTV_GGUF_BATCHED=1`, `gguf_batched.py`): one command
buffer per layer records the dense projections, the fused `gateup` + `acc`
expert dispatches and the readbacks, and `batch_moe_kquant_record` does a
whole layer's MoE in one pybind crossing. The two weight paths and both
decode modes produce the same token ids on the greedy identity prompt (see
[Wired into the decode loop](#wired-into-the-decode-loop)).

## What is in the tree

| Piece | Where |
| --- | --- |
| 7 GLSL compute shaders | `python/freetoken/kernel/vulkan/shaders/linear_q{4k,5k}_*_f32.comp`, `linear_q8_0_resident_f32.comp` |
| 4 fused 8-expert shaders | `linear_q{4k,5k}_{gateup8,acc8}_f32.comp` (see [Fused 8-expert dispatch](#fused-8-expert-dispatch-arena-slot-addressing)) |
| Push constant + dispatch | `ext_module.cpp` (`struct KQuantPC`, `linear_kquant_resident_io`, `batch_kquant_io`, `batch_moe_kquant_record`) |
| Fused bindings | `ext_module.cpp` (`KQuant8PC`/`KQuant8AccPC`, `linear_kquant_{gateup8,acc8}_io`, `batch_kquant_{gateup8,acc8}_io`, `batch_moe_kquant_fused_record`) |
| Packed expert provider | `python/freetoken/models/gguf/packed.py` |
| Gates | `tests/kernel/vulkan/test_kquant_shaders.py`, `tests/kernel/vulkan/test_kquant_fused.py`, `tests/gguf_vulkan/test_gguf_packed.py` |

Shaders (56 `.comp` in total; compile with `python -m freetoken.kernel.vulkan.build`
or `compile_shaders()`, output goes to `~/.cache/freetoken/vulkan/shaders`):

| Shader | kind | writes |
| --- | --- | --- |
| `linear_q4k_resident_f32.comp` | `resident` | `Y[T,N] = X[T,K] @ W[N,K]^T` (assignment) |
| `linear_q4k_gateup_silu_f32.comp` | `gateup` | `Y[T,N] = silu(gate[n]) * up[n]`, one call |
| `linear_q4k_acc_resident_f32.comp` | `acc` | `Y[T,N] += ...` (accumulate) |
| `linear_q5k_*` (same three) | | Q5_K unpack, adds the `qh` bit matrix |
| `linear_q8_0_resident_f32.comp` | `resident` | dense-tensor GEMV only |
| `linear_q{4k,5k}_gateup8_f32.comp` | `gateup8` | 8 experts' `silu(gate)*up` in ONE dispatch, `Y[8*T,N]` slot `j` = expert `Ids[j]` |
| `linear_q{4k,5k}_acc8_f32.comp` | `acc8` | 8 experts' down-accumulate in ONE dispatch, ordered `j=0..7` |

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

Two files carry the sequential wiring, plus the mixin for the batched one:

| Piece | Change |
| --- | --- |
| `gguf_provider.py` | `ExpertBank` gained `qtype` / `row_bytes` / `packed_bytes` (keyword-only, default `None`/0, so the existing positional construction still works); `GGUFProvider.expert_bank()` sets them from `kquant_qtype(ggml_type)`, and `expert_packed(layer, proj, expert)` returns the slab |
| `gguf_resident.py` | `_packed_qtype()` decides the lane, `_expert_handle()` uploads the packed slab on a miss (or `expert_bf16` when the lane is off), `_expert_affine{,_batch}()` dispatch `linear_q{4k,5k,8_0}_resident_io` instead of `linear_bf16_resident_io` |
| `gguf_batched.py` | `GgufBatchedMixin` (first in `ResidentGGUF`'s MRO) overrides the three batch record hooks: dtype-dispatching dense records, and `_record_moe_in` -> `batch_moe_kquant_record` (8 fused gateup-silu + 8 accumulated downs per layer, router value folded into each acc `gscale`) |

`FTV_GGUF_PACKED` (default `1`) selects the lane; `0` forces the old
host-dequant BF16 path, unchanged. Both reach the experts only through
`model.expert_mats` + `_moe_step` / `_moe_chunk`, which is also how the MTP
head reads them, so the speculative path picks the packed lane up for free.

The sequential lane uses the plain `resident` shader (three synchronous
round trips per expert affine). The batched lane (`FTV_GGUF_BATCHED=1`,
default) uses `gateup` + `acc` through `batch_kquant_io` /
`batch_moe_kquant_record` inside one open batch per layer - that is the
difference between ~4300 and ~260 synchronous round trips per token.

**What made the batched lane correct.** `batch_kquant_io` declares its
hazard flags per `kind`, and the `plain` and `gateup` arms used to declare
`Y` **READ** while the shaders write it; only `acc` was WRITE. The engine's
scoped barriers therefore ordered those writes away on some interleavings,
and the batched walk diverged from the sequential one. Making `Y` WRITE for
every kind (matching `batch_nvfp4_io`) fixed it: the feared sync-path
collapse did not reproduce, all 40 layers match at 0 or ~1e-7, and
`test_batched_fork2.py` measures seq-vs-batched **4.530e-06** (gate 1e-5),
batched-vs-batched2 **bit-exact 0.0**, identical argmax.

### Measured end-to-end (RX 5700 XT, 41-block CyberTiel GGUF)

`/tmp/opencode/closeout_bench.py` (2026-10-21; the same recipe as
`tests/vulkan/bench_gguf_packed.py`, extended so the sequential and batched
arms run **back to back inside one process** with a per-arm warm-up, median
of 3 rounds, decode only, warm page cache, loadavg ~2-5) times the decode
loop only (prefill is measured separately).

At the shipped defaults (`FTV_GGUF_BATCHED=1`, `FTV_EXP_CACHE_MB=2048`):

| Path | decode ms/tok | decode tok/s | sync round trips/tok | submits/tok |
| --- | --- | --- | --- | --- |
| batched (default) | **110.3-158.2** | **6.3-9.1** | 222-325 | 91 |
| sequential (`FTV_GGUF_BATCHED=0`) | 462.5-528.2 | 1.9-2.2 | 4246-4338 | 0 |

**3.0-4.3x**, the same ratio as the 3.43x (149.6 vs 513 ms/tok) of the run
that promoted the path. With the expert budget raised to 4096 the batched
path reaches 89.9-132.6 ms/tok (7.5-11.1 tok/s) at 127-166 round
trips/token - but that budget saturates the card at 8167 of 8176 MiB and
RADV spills the overflow into system memory, which is why the GGUF decoder
defaults it to 2048 (peak ~6998 MiB, ~1.1 GiB of headroom; the KV caches
are host RAM, so `--max-len 8192` does not move VRAM).

The host-dequant fallback (`FTV_GGUF_PACKED=0`) is unchanged at ~29.7
s/tok and still produces the same token ids. That is ~22x slower than the
packed sequential path and ~700x slower than the batched one; it is kept
as the byte-for-byte reference the golden tests compare against, not as a
runnable option (its `exp_dequant` dominates every cache miss).

Prefill of the 14-token prompt: 4.1-5.0 s warm (the packed lane removes the
host dequant from that path too; a cold LRU pays the miss path on top).
RSS peaks near 15 GiB across a multi-round bench, almost all of it the
mmap'd 22.8 GB GGUF's page cache - reclaimable, never a private copy.

The `report_stats()` split says it outright: `exp_dequant` does not appear
at all on the packed path. On the sequential lane the per-expert affine is
still 0.317 ms token upload + 0.164 ms GEMV + 0.159 ms result download,
i.e. the cost is the synchronous round trips, not the weights; the batched
lane moves those round trips into the 91 submits a token costs.

**Token identity.** The same prompt through both weight paths:

```
--prompt "Name three primary colors." --max-tokens 4 --temperature 0 --seed 1234
FTV_GGUF_PACKED=1  ->  token_ids: [5844, 11, 13358, 11]   "Red, yellow,"
FTV_GGUF_PACKED=0  ->  token_ids: [5844, 11, 13358, 11]   "Red, yellow,"
```

identical, and the batched decode mode reproduces the sequential one's ids
as well (the `fork2` gate above). This is the expected outcome rather than
a coincidence: the shader unpacks to fp32 and the fallback dequantizes to
BF16, so the two differ in the last mantissa bits of every weight (~2^-8) -
the greedy argmax has to be robust to that, and on this prompt it is. Treat
the identity as a gate, not as a proof of bit-equality.

### Last mile: done, and what is left

llama.cpp runs the same file at 16.4-17.9 tok/s. The batched path closed
most of the old ~21x gap (now 1.5-2.5x); the three steps that did it, in
order, were:

1. Wrap the 8 routed experts of `_moe_step` in one `batch_begin` /
   `batch_submit` (`batch_upload`, `batch_kquant_io`, `batch_download`) -
   done, via `GgufBatchedMixin` (`_record_moe_in`). Proven bit-exact by the
   fork (`bat-vs-bat2 = 0.0`) and by `test_batch_acc_sums_experts`.
2. Bundle gate+up per expert with `packed_gate_up_expert` and dispatch the
   `gateup` shader, then feed the resident silu slot straight into the `acc`
   kernel for the down GEMV - done (`batch_moe_kquant_record`).
3. Persistent mapping of the host-visible staging/scratch buffers
   (`FTV_STAGING_MAP=1`) - done; see the bandwidth section below.

What is left is per-layer submit/fence overhead on the main queue (91
submits/token) plus the attention host math; the expert bytes are no longer
in the wall. The next levers, in order of expected value: overlap the
per-layer H2D with the previous layer's compute (async upload queue hooks
already exist in `ext_module.cpp`), move the attention KV math onto the GPU
(it is host-side today), and only then re-tune kernels.

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

**H2D after the persistent-mapping rewrite** (`vk_util.hpp`: `Buffer` gains
`mapped` + `host_coherent`; `make_host_ssbo` prefers a non-coherent
HOST_VISIBLE type and falls back to coherent; `host_write_begin`/
`host_write_end`/`download` flush exactly the touched range when - and only
when - the memory is non-coherent; all five host-visible paths route through
the hooks). At a 1.77 MB transfer:

| path | before (map/unmap per use) | after (persistent map) |
| --- | --- | --- |
| sync one-shot | 4.14 GB/s | 4.19-4.33 GB/s (unchanged) |
| batched / async queue | 5.4 GB/s | **11.5-12.5 GB/s (2.1-2.3x)** |

On this box RADV exposes **no** non-coherent host type (all six host-visible
types are HOST_COHERENT), so the flushes are spec-mandated no-ops here and
the entire win is deleting the per-upload `vkMapMemory`/`vkUnmapMemory`
(~130 us/MB of driver page-table work). The knob is `FTV_STAGING_MAP`:
`1` (default) persistent mapping everywhere, `0` the old per-use map/unmap,
`2` persistent except `upload_via_staging`'s one-shot sync staging.

At that rate the packed 13.5 MiB/token costs about 2.5 ms of upload against
5.1 ms for 27 MiB of BF16 - and, unlike the host path, the unpack work is on
the GPU and the CPU is free to run the rest of the step.

## Fused 8-expert dispatch (arena slot addressing)

The batched decode path pays **16 K-quant dispatches per layer** (8 gateup +
8 acc) inside its 91 submits/token. The fused pair collapses each half to
**ONE dispatch** - 16 -> 2 per layer - by taking the eight routed experts of a
layer in a single grid, addressed through the engine's fixed-slot per-layer
expert arena (28 slots/layer = 2047 MiB within the 2048 MiB cache budget,
73.125 MiB per slot per layer).

| Shader | grid | replaces |
| --- | --- | --- |
| `linear_q{4k,5k}_gateup8_f32.comp` | `(N, 8)` - WG `(n, j)` = expert `j`, gate row `n` | 8 x `linear_q{4k,5k}_gateup_silu_f32` |
| `linear_q{4k,5k}_acc8_f32.comp` | `(N, 1)` - one 128-lane WG per output row, all 8 dots | 8 x `linear_q{4k,5k}_acc_resident_f32` |

Five std430 bindings, same for both halves:

```
0  arena packed K-quant bytes (per-slot stride = rows_per_slot rows)
1  X fp32 (gateup: the layer input; acc: the silu-all vector)
2  Y fp32 (gateup: [8*N], expert j -> rows [j*N, (j+1)*N); acc: [N], RMW)
3  Ids u32[8]  - expert j -> arena slot index
4  Gs  f32[16] - gateup scale at Gs[j], acc scale at Gs[8+j]
```

Push constants: `KQuant8PC {T,N,K,wpr,ng32,rows_per_slot,rowstride}` (28 B)
for gateup; `KQuant8AccPC` adds `silu_N` (32 B) for acc. Slot 0 is the
engine's `0x7FFFFFFF` no-content sentinel - an out-of-slab row offset makes
the packed fp16 scale halves decode to NaN, so the sentinel doubles as an
in-bounds probe (Gate B asserts poisoned where hit, finite everywhere else).

**Bit-identity is structural, not approximate.** `row_dot` in both fused
shaders is the exact source of the per-expert shaders (same lane/word split,
super-block indexing, `k4_sm` 6-bit extraction, FMA order, butterfly) - only
address/value plumbing moves (`Ids[j]*rows_per_slot` for the row base,
`Gs[j]` for gscale, a shared output slot for the write). The acc half is
deliberately `(N, 1)` rather than `(N, 8)`: with eight workgroups sharing a
row the 8 fp32 RMW adds would race, so each lane keeps 8 private
accumulators, one tree reduce yields all `d_j`, and lane 0 sums them in
canonical `j = 0..7` order into ONE add over a `+0.0`-prefilled `Y` - the
same strictly-ordered chain as the 8 sequential dispatches.

Bindings (90 total in the extension):

| Binding | Kind |
| --- | --- |
| `linear_kquant_gateup8_io` / `linear_kquant_acc8_io` | sync, one submit each |
| `batch_kquant_gateup8_io` / `batch_kquant_acc8_io` | inside an open batch |
| `batch_moe_kquant_fused_record` | one pybind crossing for a whole layer: gateup8 + acc8 + the shared-expert tail (bf16 gate/up/gscale, `batch_swiglu`, down, both downloads) |

Engine call site (pending the engine lane): swap `batch_moe_kquant_record`'s
16 `batch_kquant_io` (`gguf_batched.py` `_record_moe_in`) for
`batch_moe_kquant_fused_record` over the per-layer arenas - the expected
saving is 14 dispatches' record+drain overhead per layer, which is what lets
the token shrink to 1-2 command buffers and lets the verify row batch amortize
its drain.

Measured (`KFUSED_BENCH`, layer-0-sized real expert bytes, bit-exact vs the
per-expert arms in both GEMVs):

| Case | 16 dispatches (per-expert) | 2 dispatches (fused) |
| --- | --- | --- |
| wall, 8 experts gateup+acc | 0.1999-0.2277 ms | **0.1467-0.1719 ms** |
| dispatch count | 16 | **2** |
| quantized-weight bandwidth | - | **84.4-93.2 GB/s** |
| H2D content bytes | 14,155,776 | **14,155,776 (equal - arena is contiguous)** |

plus `arena_sentinel_overhead_bytes = 1,769,472` (1.77 MB of reserved
slot-0 poison the engine uploads on top of the content bytes - the price of
the sentinel).

## BAR readback (GDN pull-down) fix

`make_host_ssbo` asked only for `HOST_VISIBLE|HOST_COHERENT`, and RADV
resolved the staging pool to the **uncached** host type (property bits
`0x0006`, heap mapped into GTT) - every CPU read of a completed copy went
through the uncached aperture at **~340 MB/s**. The GDN pull-down (30 states
of 2 MB + 30 conv buffers of 96 KB = 60 readbacks, 62.8 MB) therefore cost
**187.6 ms wall per snapshot** (fence 2.56-2.79 ms + ~185 ms memcpy) and a
sync `download_resident` of the same bytes **204.7 ms**. The cached host type
(`0x000e`) was present all along - a memory-type probe measured 4.5 GB/s with
it, against 340 MB/s without.

Fixes, both in `vk_util.hpp`:

1. `make_host_ssbo` now prefers `HOST_VISIBLE|HOST_CACHED` and `make_buffer`
   degrades by dropping `HOST_CACHED` only if no such type exists;
   `FREETOKEN_VULKAN_LOG_MEMTYPE` prints the chosen type. Both candidate
   types are HOST_COHERENT here, so the persistent-mapping flush/invalidate
   contract documented above is untouched.
2. `memcpy_bar` (used by `vku::download`, covering sync and batch readbacks)
   takes a `movntdqa` non-temporal path for >= 1024 B with an
   `__builtin_cpu_supports("sse4.1")` fallback to `std::memcpy`. This is
   second-order next to (1): A/B on the snapshot arm gave std 5.80 / avx
   5.24 / nt 4.95 / mt 9.44 ms - multithreaded was *worse* and was dropped.

Uploads are unaffected (5.62 vs 5.61 ms for the same 62.8 MB, 11.2 GB/s).

Measured back to back after the fix (4 runs of the snapshot arm, load avg
4.56-4.99, `Cached` 15.8-16.5 GB, RSS 575 MB):

| copy | before | after |
| --- | --- | --- |
| snapshot arm wall (fence + 60 readbacks) | 187.6 ms | **7.47-8.13 ms** |
| readback bandwidth | 0.34 GB/s | **11.1-12.8 GB/s (33-38x)** |
| sync `download_resident` 62.8 MB | 204.7 ms (0.31 GB/s) | **9.74-10.40 ms (6.0-6.4 GB/s)** |
| DRAM->DRAM control, same bytes | ~6.7 ms (~9.4 GB/s) | same - the readback now sits at DRAM speed |

Every arm asserted bit-exact against the pre-fix buffers. For MTP this
retires the per-round `snapshot()`/`restore()` pull-down (was ~186 ms/round
on the engine's 60-readback path) down to ~6 ms of memcpy.

## Gates

```bash
cd FreeToken-Navi10-Cybertiel
PYTHONPATH=python python -m pytest tests/kernel/vulkan -q                    # no checkpoint
PYTHONPATH=python FREETOKEN_TEST_MODEL_GGUF=<path> \
  python -m pytest tests/kernel/vulkan -q                                     # + real experts + fused/readback gates
PYTHONPATH=python FREETOKEN_TEST_MODEL_GGUF=<path> \
  python -m pytest tests/kernel/vulkan/test_kquant_shaders.py -q             # + real expert
PYTHONPATH=python FREETOKEN_TEST_MODEL_GGUF=<path> \
  python -m pytest tests/gguf_vulkan -q                                      # packed provider
```

Measured on this box (interpreter `/home/cus/.venv-vllm-vulkan/bin/python`,
2026-10-21):

| Command | Result |
| --- | --- |
| `pytest tests/kernel/vulkan -q` (no checkpoint) | **40 passed, 2 skipped** |
| `FREETOKEN_TEST_MODEL_GGUF=<path> pytest tests/kernel/vulkan -q` | **42 passed** in 32.46 s |
| `FREETOKEN_TEST_MODEL_GGUF=<path> pytest tests/kernel/vulkan/test_kquant_shaders.py -q` | **24 passed** in 12.61 s (adds the real `blk.0` expert) |
| `FREETOKEN_TEST_MODEL_GGUF=<path> pytest tests/gguf_vulkan -q` | **45 passed, 1 skipped** in 18.87 s |
| `pytest tests/gguf_vulkan -q` (no checkpoint) | 18 passed, 28 skipped |

The fused pair has its own gates in `tests/kernel/vulkan/test_kquant_fused.py`
(9 tests: fused-vs-per-expert sync bit-exact on real Q4_K/Q5_K bytes + numpy
sanity, shuffled-arena + sentinel edge probes, the batch pair,
`batch_moe_kquant_fused_record` vs a manual 16-dispatch + shared-tail
reference with a byte-identical shared down, and the bench), and
`tests/kernel/vulkan/test_bar_readback.py` checks the readback fix (snapshot
round trip bit-exact + a >= 1000 MB/s floor against the measured 11-12
GB/s).

Every GPU test skips when no Vulkan device is present. Regenerate the binding
stubs after touching `ext_module.cpp`:

```bash
PYTHONPATH=python python -m freetoken.kernel.vulkan._gen_stub   # 90 functions
```
