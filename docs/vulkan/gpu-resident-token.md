# GPU-resident token (design) for the CyberTiel GGUF Vulkan path

**Status:** design only. No engine, kernel or other-test file was changed by
this pass; the two files this lane owns are this document and
`tests/vulkan/bench_token_overhead.py` (the measurement harness).

**Target hardware:** AMD RX 5700 XT (Navi10 / gfx1010, 8 GB), RADV Vulkan 1.4.357.
**Model:** Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF
(Qwen3.5-MoE: 40 real layers + MTP block 40, 30 GDN + 10 full-attention,
256 experts top-8, Q4_K on blocks 15-40 and Q5_K on blocks 0-14).
**Repo:** `/home/cus/working/FreeTokenVulkan5700xt/FreeToken-Navi10-Cybertiel`
(branch `feature/vulkan-navi10-port`).

---

## 0) Summary of the argument

The token today is **96 ms** (10.4 tok/s on a quiet box). It splits into two
halves that need opposite fixes:

| Half | Measured | What it is | Fix |
|---|---|---|---|
| **host work** | 54.5 ms/tok (56%) | expert pack+fetch (25.0), command recording (11.4), host router (7.6), host attention (4.0), host RMSNorm (2.9) | arena + fused recorders + GPU router + GPU attention + GPU norms |
| **fence wait** | 45.4 ms/tok (44%) | 91 pipeline drains; the median drain waits **0.42 ms** while an empty submit costs **0.034 ms** | collapse 91 drains into 1-2 command buffers per token |

The fence-wait profile is the decisive measurement of this pass: **73% of the
91 drains wait 0.1-0.5 ms each, and no drain waits less than 0.1 ms.** That is
pipeline-drain latency (submit -> fence signal -> host wake), not kernel
execution. The GPU's real per-token work is much smaller than the 45.4 ms the
fence bucket attributes to it.

So the plan is, in order:

1. cut the host work to ~10 ms/tok (arena, fused K-quant recorder, GPU router,
   GPU attention, GPU norms),
2. collapse the 91 drains to 1-2 (one command buffer per token),
3. keep the expert traffic off the critical path (async queue-2 uploads,
   prewarm) so the PCIe cost overlaps the GPU work.

Projection: **30-40 ms/tok -> 24-33 tok/s**, versus llama.cpp's Vulkan build
at 16.4-17.9 tok/s on the same file. The projection's biggest uncertainty is
the GPU's true per-token critical path, which section 10 gives the cheapest
experiment to settle.

---

## 1) Measured baseline

### 1.1 The quiet-box run (this pass, the numbers the design uses)

Command:

```bash
cat /tmp/opencode/CyberTiel.gguf > /dev/null          # warm the page cache
FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \
  /home/cus/.venv-vllm-vulkan/bin/python tests/vulkan/bench_token_overhead.py \
  --tokens 16 --warmup 4 --runs 2 --arm idle --json /tmp/opencode/overhead_final.json
```

Median of 2 runs (0.092 and 0.100 s/tok):

| Metric | Value |
|---|---|
| token time | **0.096 s/tok (10.43 tok/s)** |
| gpu_wait (fence+sync) | **45.35 ms/tok (45.4%)** |
| host_work | **54.51 ms/tok (54.6%)** |
| env at the median run | `cached=10932 MiB  mem_avail=9331 MiB  load=[3.07, 2.01, 1.83]  rss_max=12755 MiB` |
| env at the end of the run | `cached=10932 MiB  mem_avail=9290 MiB  load=[3.46, 2.11, 1.86]  rss_max=12756 MiB` |
| calibration | empty submit **0.034 ms**, batch H2D **16.58 GB/s**, batch D2H **13.17 GB/s**, sync H2D 16.40 GB/s, sync D2H 12.98 GB/s |

(The bucket table in 1.2 is the median-picked run, 0.0999 s/tok, so the
buckets sum to that run rather than to the 2-run median.)

The box was quiet (load 3.07 and falling; four other lanes had been running
before the machine was power-cycled). `Cached` was ~10.9 GiB of the 22.8 GB
file; the RSS peak of 12.8 GiB is the mapped file plus the page cache, not a
copy of the checkpoint.

### 1.2 Bucket breakdown (ms/tok, calls/tok)

| Bucket | ms/tok | calls/tok | What it is |
|---|---|---|---|
| `fence` | 45.35 | 91.0 | `batch_submit` + `vkWaitForFences` + readbacks (GPU work + drain latency) |
| `host_fetch` | 25.01 | 640.0 | `_gguf_gu_handle` / `_gguf_dn_handle`: LRU lookup, and on a miss the **host pack** of the expert out of the mmap |
| `record` | 11.38 | 968.7 | host-side `vkCmd*` recording (every `batch_*` op except submit) |
| `host_moe` | 7.56 | 40.0 | the host router: `F.linear` + softmax + `topk` + Python |
| `host_attn` | 3.95 | 10.0 | `_full_attn_host`: KV append + RoPE + per-head RMSNorm + SDPA on the CPU |
| `host_rms` | 2.92 | 81.0 | host RMSNorm |
| `host_embed` | 0.13 | 1.0 | the embedding row |
| `async_up` / `host_gdn_sync` | 0.00 | 1.0 | neither is active in the GGUF path |
| remainder (Python/torch glue) | ~3.6 | | not inside any wrapped call |

### 1.3 Interactions per token

| Metric | Value | Note |
|---|---|---|
| drains (fence + sync round trips) | **91.0** (submits 91.0, sync 0.0) | one pipeline drain per layer-ish |
| host calls | 772.0 | record-only, no host wait |
| barriers | 702.0 | hazard barriers inside the command buffers |
| queue-2 submits | 0.0 | the async upload path is unused here |
| bytes/token up / down | **167.1 MiB / 2.2 MiB** | the up-traffic is the packed expert bytes |
| expert cache hit / miss | **460.5 / 179.5** (56% miss) | 320 routed requests per token |
| drain fixed overhead | 3.09 ms/tok | 91 x 0.034 ms |

The changelog's "222-325 GPU->host round trips per token" is this list from a
different angle: **91 submits + ~120 per-layer downloads** (3 per MoE layer:
`acc_out`, `sd_out`, `gsc_out`) + the attention and logits readbacks. The
downloads are the half that a GPU-resident design has to delete.

### 1.4 The fence-wait profile (the decisive measurement)

All 1456 drains of the run (16 tokens x 91):

| Statistic | Value |
|---|---|
| median wait | **0.421 ms** |
| p90 | 0.792 ms |
| p99 | 3.081 ms |
| max | 3.441 ms |
| waits < 0.1 ms | **0** |
| waits < 0.5 ms | 1067 (73%) |
| waits >= 1 ms | 93 (6.4%) |

An empty `batch_begin`+`batch_submit` costs 0.034 ms, so a 0.42 ms median wait
is **12x the fixed submit cost** and is dominated by the drain round trip
(submit -> GPU executes -> fence -> host wakes). If the GPU's real per-token
work were 45 ms, the drain waits would be a few large ones; they are not.

Independent cross-check on the GPU side: the fused K-quant gate's own bench
(the parallel lane's `test_fused_dispatch_bench`, q4_k, real dimensions)
measures **0.1475 ms for 8 experts**, i.e. **5.9 ms/token for 40 layers of
MoE** and **96.98 GB/s of quantized-weight throughput**. The GPU's MoE work is
therefore small; the 45.4 ms fence bucket is mostly waiting.

### 1.5 The loaded-box run (contrast, not the design baseline)

An earlier run of the same harness with four other lanes on the box (median of
3 per arm) gave idle 0.337 s/tok (2.97 tok/s) and busy 0.326 s/tok (3.07
tok/s), busy/idle = 0.97x. Env: `loadavg 3.06 2.95 2.91`, `Cached 11350660
kB`, `MemAvailable 7961668 kB`, `rss_max 9625 MiB`.

Its buckets disagreed with the quiet run in one important way:
`host_fetch` was 141.22 ms/tok idle vs 30.71 busy, and `host_attn` was 5.21
idle vs 98.54 busy. **Both are contention artifacts** (the same code path
cannot be 5x faster when the box is busier). The quiet-box run is the one to
design against; the loaded run only establishes that the host paths are
contention-sensitive and that the *split* (gpu_wait ~45%, host_work ~55%)
survives both.

---

## 2) Target: host interactions per token

### 2.1 The count

**Target: <= 5 host-visible interactions per token, of which exactly 1 is a
blocking wait.**

| # | Interaction | Payload | Why it cannot be removed |
|---|---|---|---|
| 1 | `batch_begin` | - | open the command buffer |
| 2 | ~40-120 `batch_*` **record** calls | none (host-only `vkCmd*`) | these are not round trips; they are the work itself |
| 3 | `batch_submit` + fence | - | the one hard drain |
| 4 | logits readback (1 MiB) | D2H | the token id has to come back |
| 5 | token id / argmax | D2H | can be folded into (4) |

Against today: **91 drains -> 1**, **~120 per-layer downloads -> 0**, **772
host calls -> ~120**, **702 barriers -> ~40** (one per layer, not three).

The per-layer downloads are the ones that are easy to miss. Today every MoE
layer hands the host `(acc, sd, gsc)` and the host computes
`x = x1 + acc + gs*sd`. For the GPU to stay authoritative, that add must move
onto the GPU (`batch_vecadd_io` already exists for the GDN path), the
accumulator must never leave VRAM, and the next layer's input must be read
from VRAM instead of downloaded and re-uploaded.

### 2.2 Arithmetic: interactions

| | today | target | how |
|---|---|---|---|
| submits / drains | 91 | 1 | one command buffer for the whole token |
| GPU->host downloads | ~120 + 11 | 1-2 | keep `acc`, `sd`, `gsc` device-local; fuse the residual add |
| host calls (record) | 772 | ~120 | fused recorders: 1 pybind crossing per layer instead of ~24 |
| host round trips (blocking) | 91 | 1 | the single fence |

### 2.3 Arithmetic: time

Measured, quiet box: **96 ms/token**.

```
 96.0 ms  =  54.5 ms host work  +  45.4 ms fence wait
           =   25.0 host_fetch  + 11.4 record + 7.6 host_moe
             +  4.0 host_attn   +  2.9 host_rms  + 0.1 embed + ~3.6 glue
           =   45.4 fence = W_gpu (~20 ms) + 91 drains x ~0.28 ms latency
```

The brief's "222-325 round trips and ~170 ms/token" is a **moderately loaded**
box: this pass measured 96 ms/token quiet and 337 ms/token on a box with four
other lanes (the changelog's own range for this path is 110-158 ms/tok). The
interaction count is load-independent; the time is not.

| Step | Removes | Token after |
|---|---|---|
| baseline | - | 96.0 ms (10.4 tok/s) |
| arena + fused recorder + GPU router + GPU attention + GPU norms | 25.0 -> 4.0 (fetch), 11.4 -> 4.5 (record), 7.6 -> 0.5 (router), 4.0 -> 0.5 (attn), 2.9 -> 0.5 (norms), glue 3.6 -> 2.0 | **75.3 ms (13.3 tok/s)** |
| collapse 91 drains -> 2 | 91 x 0.28 ms of drain latency (the GPU work `W_gpu` stays) | **38 ms (26 tok/s)** |
| overlap the expert PCIe with the GPU work | 5.4-7.2 ms of steady-state PCIe, hidden by the queue-2 path or not | **32-38 ms (26-31 tok/s)** |

The brief's "22-26 ms compute floor" is `W_gpu` plus a small host residue.
**The floor is only reachable if `W_gpu` is really ~20 ms**, and that is the
one number this pass did not measure directly (it is inferred from the drain
profile and the fused bench). Section 10 gives the experiment that measures it.

**Honest projection: 26-33 tok/s** (32-38 ms/token; the upper end needs the GPU attention and norms of step 5). It beats llama.cpp's
16.4-17.9 tok/s on the same file, because llama.cpp's speed comes from
multiplying the experts **on the CPU** (41 ms/token of AVX2 Q4_K GEMV at
13.7 GB/s) while the GPU does the rest, whereas the fused K-quant kernel
consumes quantized weights at **96.98 GB/s** - 7x the CPU tier's rate - and
only pays the PCIe transfer for what the arena does not hold.

---

## 3) Call sequence (per layer) with the exact runtime ops

### 3.1 What already exists in the runtime (no kernel work needed)

| Op | Exists | Role in the new sequence |
|---|---|---|
| `batch_kquant_gateup8_io` | yes (parallel lane, gated) | fused 8-expert gate+up+SiLU over the arena |
| `batch_kquant_acc8_io` | yes (parallel lane, gated) | fused 8-expert down accumulate over the arena |
| `batch_moe_kquant_fused_record` | yes (parallel lane, gated) | gateup8 + acc8 + the shared-expert tail in one crossing |
| `batch_topk256_io` | yes (the NVFP4 path uses it) | the GPU router: top-k over 256 experts, writes arena ids + scales |
| `batch_rms1p_io`, `batch_fp32_io` | yes | the router's RMSNorm(1+w) and its fp32 GEMM |
| `batch_upload_async`, `batch_upload_async_flush`, `batch_cross_barrier`, `batch_submit([cookie])`, `batch_upload_async_drain` | yes | the queue-2 upload path (unused by the GGUF path today: `q2_submits = 0`) |
| `batch_gdn_conv_io`, `batch_gdn_recur_io`, `batch_vecadd_io` | yes | the GPU-resident GDN (already used by the GGUF path) |
| `batch_upload_multi` | yes | one union copy for a layer's misses |
| `batch_download` | yes | the readback path (now used for the snapshot only, at ~10 GB/s, see section 9) |
| `batch_moe_kquant_record` | yes | today's per-expert recorder (16 dispatches/layer) |

### 3.2 The target sequence

```text
per token:
  batch_begin()                                  # ONE command buffer for the token
  batch_upload_async_drain()                     # safety net: previous token's async uploads
  for layer i in layer_types:
    if layer is GDN:                             # 30 layers, already GPU-resident
        batch_upload_multi([h_x, h_x_resid], ...) # the residual stream + the normalized input
        batch_gdn_conv_io(...)                    # conv taps + silu
        batch_gdn_recur_io(...)                   # the recurrence (device-local state)
        batch_vecadd_io(x_resid, ao, x1)          # x1 = x + ao, ON the GPU
        # h2 = RMSNorm(x1) stays on the GPU:
        batch_rms1p_io(x1 -> h2)
    else:                                        # 10 full-attention layers
        batch_rms1p_io(x1 -> h2)
        # KV append + RoPE + flash-attn on the GPU (new recorder, see section 5)
        batch_flash_attn_kv_io(...)               # writes K/V into the capacity-strided slab
    # MoE, all on the GPU:
    batch_fp32_io(router_w, h2 -> logits256)      # the router GEMM
    batch_topk256_io(logits256 -> arena_ids, arena_gs)
    batch_moe_kquant_fused_record(h2, arena_ids, arena_gs, arena)   # 2 dispatches
    batch_vecadd_io(x1, acc, x1)                  # x1 += acc (+ the shared expert's tail)
  batch_rms1p_io(x -> h_last)                     # the final norm
  batch_download(logits)                          # 1 MiB
  batch_submit()                                  # the ONE fence
```

Per-layer GPU work in that sequence: 1 RMSNorm + 1 router GEMV + 1 top-k +
2 MoE dispatches + 1-2 attention/GDN dispatches + 1 residual add. Roughly
**8 dispatches per layer, 320 per token, in one command buffer**.

### 3.3 The differences vs the NVFP4 path (which already has the arena)

| Aspect | NVFP4 (proven) | GGUF / K-quant (this design) | Action |
|---|---|---|---|
| Arena contents | NVFP4 packed blocks + fp16 scales | **packed K-quant slabs** (Q4_K / Q5_K) | the arena slot holds the *packed* bytes; the shader dequantizes in-shader. Slot stride = the row stride of the packed format. |
| Expert bytes | pre-packed at load, one copy per expert | the mmap holds gate and up as **separate** tensors; the fused gateup8 shader wants them **stacked** (rows 0..N-1 gate, N..2N-1 up) | one host-side stack per expert (`packed_gate_up_expert`), done once per expert and cached in host RAM; the arena slot then holds the stacked bundle |
| Fused recorder | `batch_moe_fused_record` | `batch_moe_kquant_fused_record` | already exists and is gated |
| Router -> arena ids | `batch_topk256_io` writes `arena_ids` / `arena_gs` | same, but **every K-quant gateup scale is 1.0** and the acc scale is the renormalized router value | `Gs[0..8) = 1.0`, `Gs[8+j] = router_value` |
| Downloads | the fused record returns 3 tensors | same | they must stop returning (see section 2.1) |
| Init | the arena is on | `ResidentGGUF.__init__` **forces `FTV_EXP_ARENA=0` and `FTV_SPEC_PACK=0`** | the flags must be relaxed, with a guard |

### 3.4 The four fused K-quant shaders (parallel lane, on disk and gated)

`linear_q4k_gateup8_f32.comp`, `linear_q5k_gateup8_f32.comp`,
`linear_q4k_acc8_f32.comp`, `linear_q5k_acc8_f32.comp`.

**Arena contract** (from the lane's own gate file, and the contract the engine
must satisfy):

* gateup arena slot `j` = rows `[slot*rows_per_slot, slot*rows_per_slot + 2N)`,
  gate rows 0..N-1 then up rows N..2N-1 (one packed bundle per slot);
* down arena slot `j` = rows `[slot*rows_per_slot, +dn_rows)`;
* `Ids[j]` = the arena slot of canonical expert `j` (u32[8]);
* `Gs[j]` = the gateup scale, `Gs[8+j]` = the acc scale (fp32[16]);
* **slot 0 is the engine's `0x7FFFFFFF` sentinel.** The shaders address it
  like any slot, and the lane fills unused slots with the `0x7FFFFFFF` word
  pattern whose fp16 halves are NaN, so an accidental read of the sentinel
  poisons the output and fails the bit gate. The engine must therefore never
  route an expert to slot 0.

**Bit-identity** is enforced by construction, not by tolerance: the acc8
shader keeps 8 private accumulators per lane and sums them in the canonical
expert order `j = 0..7` with a single add chain
(`Y[n] = ((d0+d1)+...+d7)`), reusing the exact `row_dot` logic of the
per-expert resident shaders. The gate file asserts **byte equality** against
the per-expert sequence.

**Measured by the lane's bench (I re-ran their gate):**

| Metric | Value |
|---|---|
| dispatches per 8 experts | **16 -> 2** |
| wall per 8 experts (q4_k, 512x2048 gate + 2048x512 down) | 0.1971 ms -> **0.1475 ms** (1.34x) |
| quantized-weight throughput | **96.98 GB/s** |
| H2D bytes for the 8 experts | 14 155 776 B (1.77 MiB per expert) |
| sentinel overhead bytes | 1 769 472 B |

**Impact on this design:** the MoE's GPU time is 5.9 ms/token (40 layers x
0.1475 ms). The kernel half of "no per-expert host round trip per MoE layer"
is done and gated; **the engine half (the arena that feeds it) is the missing
piece and is the top step of the plan.**

---

## 4) Exact engine-side changes

Order matters: each step must leave the gates green, so the fused recorder
cannot be enabled before the arena exists, and the GPU router cannot be
enabled before the arena ids exist.

### 4.1 The seven steps of engine work

| # | File | Function | Change | Correctness argument | Gate |
|---|---|---|---|---|---|
| 1 | `python/freetoken/models/qwen3_5_moe/vulkan/gguf_resident.py` | `ResidentGGUF.__init__` | Stop forcing `FTV_EXP_ARENA=0` / `FTV_SPEC_PACK=0` for the GGUF batched path. Allocate the per-layer K-quant arena (gate+up bundle buffer + down buffer, `slots` slots each), sized from the layer's qtype. **Fill lazily** (an expert is packed and uploaded the first time it is routed), never at load: a full 2340 MiB arena would take ~4 minutes to fill at 16.6 GB/s. | The arena only changes *where* expert bytes live; the bytes are the same packed bytes the current LRU uploads. The current path stays the fallback when `FTV_EXP_ARENA=0`. | fork2 (seq-vs-batched maxdiff <= 5.5e-06, batched-vs-batched2 bit-exact 0.0), VRAM peak < 7.5 GB, identity `[5844, 11, 13358, 11]` |
| 2 | `python/freetoken/models/qwen3_5_moe/vulkan/gguf_batched.py` | `_record_moe_in` | With the arena on, replace the per-expert `batch_moe_kquant_record` (16 dispatches) by `batch_moe_kquant_fused_record` (2 dispatches) over the arena slots, with `Ids`/`Gs` from the arena's slot table. Keep the host router for now (step 4 moves it). | The fused pair is **byte-identical** to the sequence it replaces (the lane's gate asserts it for both qtypes, including the sentinel slot). The shared-expert tail is unchanged. | `tests/kernel/vulkan/test_kquant_fused.py` (9 green), fork2, batched-vs-batched2 0.0 |
| 3 | `python/freetoken/models/qwen3_5_moe/vulkan/gguf_batched.py` | `_record_moe_in` (miss path) | On an arena miss: pack the expert (stack gate+up) once, cache the packed bytes in host RAM, upload into a free slot with `batch_upload_multi`, update the slot table. Use `batch_upload_async` + `batch_cross_barrier` for the upload so it does not sit on the critical path. | Fetch-then-record (the current code already collects the misses before recording) keeps every eviction/free inside the pre-record window, so no command references a victim. The slot table is the only mutable state and is updated before any dispatch that reads it. | fork2, miss-rate counter, `batch_upload_async_stats` shows queue-2 submits > 0 |
| 4 | `python/freetoken/models/qwen3_5_moe/vulkan/nvfp4_resident.py` + `gguf_batched.py` | the MoE branch of `decode_step_batched` | Move the router to the GPU: `batch_rms1p_io` -> `batch_fp32_io` -> `batch_topk256_io`, writing `arena_ids`/`arena_gs` directly. Reuse the fork's `_gpu_router_sync_map` / `_gpu_router_prewarm_async_uploads` machinery, adapted to the K-quant arena. | The GPU router is the fork's proven path (the NVFP4 path uses it); the K-quant difference is only that all gateup scales are 1.0 and the acc scale is the renormalized router value. The sentinel (`0x7FFFFFFF`) must never appear in a routed id. | the GPU router's CHECK mode (per-layer h2 <= 1e-5, paired id/scale <= 1e-6, >= 50-token argmax, sentinel contract), fork2 |
| 5 | `python/freetoken/models/qwen3_5_moe/vulkan/gguf_batched.py` + `nvfp4_resident.py` | the residual add | Keep `acc`, `sd`, `gsc` device-local: add them on the GPU (`batch_vecadd_io`, and the shared-expert tail fused into the same op) instead of downloading 3 tensors per layer. The next layer's input is read from VRAM, not downloaded and re-uploaded. | The add is the same arithmetic in the same order (fp32 accumulate into the residual), just on the device. Removing a download cannot change the arithmetic, but the ORDER of the adds must match. | fork2 (maxdiff <= 5.5e-06), batched-vs-batched2 0.0 |
| 6 | `python/freetoken/models/qwen3_5_moe/vulkan/nvfp4_resident.py` + `gguf_batched.py` | `decode_step_batched` | **One command buffer per token**: open the batch once before the layer loop and submit once after it, instead of 91 begin/submit pairs. All the per-layer hazard barriers stay (they are inside the buffer); only the drain points go. | The barriers are what order the work; removing the submits does not remove the barriers. The GPU's execution order is unchanged. The risk is a missing hazard, which the barrier counters expose. | fork2, barrier counts, `submits/token` drops from 91 to 1-2, no hang |
| 7 | `python/freetoken/models/qwen3_5_moe/vulkan/mtp.py` | snapshot / rollback | With the GDN state device-local and authoritative, snapshot `gdn_state_gpu` / `conv_buf_gpu` through `batch_download` (the parallel lane's fixed path, ~10 GB/s instead of 333 MB/s) instead of reading the WC BAR mapping. | The readback is bit-exact (the lane's gate asserts byte equality with NaN-safe int32 comparison), so the rollback restores the same bytes. | the MTP token-identical gate, `tests/kernel/vulkan/test_bar_readback.py` |

### 4.2 Things this design must NOT do

* **No requantization.** The expert formats stay Q4_K/Q5_K exactly as shipped
  (the project's safety rule: fixed weights stay bit-exact).
* **No change to the K-quant shaders.** They are gated and byte-identical; the
  engine has to feed them, not touch them.
* **No per-expert host round trip.** The whole point of the fused recorder.
* **No new format for the GDN state.** Only where it is read from changes.

---

## 5) The attention path

### 5.1 Current state

The 10 full-attention layers are computed **on the host**
(`_full_attn_host` in the Vulkan engine): the KV append into
`k_cache`/`v_cache`, the partial RoPE, the per-head Q/K RMSNorm(1+w), the
GQA `repeat_interleave`, the SDPA, and the sigmoid gate. Measured today:
**3.95 ms/tok over 10 calls** (quiet box; 5.2-98.5 ms/tok on the loaded box,
so it is contention-sensitive).

The 30 GDN layers are already **GPU-resident** (`batch_gdn_conv_io` +
`batch_gdn_recur_io`), with zero host downloads.

### 5.2 Can the runtime's flash-attention shaders serve them?

**Yes, structurally.** The runtime already ships
`flash_attention_gpt_oss_kv_cache_f32.comp`, whose interface is exactly the
shape this model needs:

| Requirement (Qwen3.5 decode) | Shader capability |
|---|---|
| Q `[B, H_q, S_q, D]` = `[1, 16, 1, 64]` | yes |
| K/V `[B, H_kv, S_kv, D]` = `[1, 2, pos+1, 64]` | yes, `S_q != S_kv` is supported |
| GQA with `H_q/H_kv = 8` | yes |
| `past_len = pos` (the decode offset) | yes, explicit |
| KV layout | capacity-strided `[bh*capacity + start + s]*D + d`, which is **exactly** what `kv_cache_append_f32.comp` writes, so the existing host KV tensors can be reused as the slab |
| sliding window | supported (not needed here) |
| sinks | not needed for Qwen |
| RoPE, per-head RMSNorm(1+w), the sigmoid gate | **not** in the shader: they must be separate ops or new shader variants |

### 5.3 What it costs to move attention to the GPU

| Work item | Effort | Note |
|---|---|---|
| `batch_flash_attn_kv_io` (a new batch recorder) | new kernel-side work | the recorder family exists; this one does not |
| RoPE + per-head RMSNorm(1+w) as separate recorded ops | small | `rope_partial_f32.comp` and `rmsnorm_1pw_f32.comp` exist; the per-head layout has to be checked |
| the KV slab must be device-resident and capacity-strided | medium | the host path's `k_cache`/`v_cache` are torch tensors; the GPU path needs the same tensors uploaded once (40 MB at 4096 ctx - cheap) |
| the sigmoid gate and the output projection | small | the O projection is already a batched linear |
| **parity with the host SDPA** | the real risk | the host path uses torch SDPA; a flash-attention shader will differ in the last bits. The fork's gates allow a tolerance on the sequential-vs-batched comparison (<= 5.5e-06) but a *different* SDPA implementation can exceed it. |

**Recommendation: do not block the plan on it.** The saving is 3.95 ms/tok on
a quiet box (8% of the projected 38 ms token) for a new kernel-side recorder
plus a parity fight. **Do it as step 5 of the plan, after the arena and the
submit collapse**, and only if the token is still host-bound. The
recommendation changes if the attention is moved to a GPU-resident KV cache
for other reasons (it is a prerequisite for a GPU-resident GDN snapshot and
for MTP).

**If attention stays on the host**, the cost is 3.95 ms/tok of host work that
cannot overlap with the GPU (the host is busy while the GPU is idle). It can
be reduced by avoiding the `repeat_interleave` materialization (8x the KV
bytes) and by using a fused SDPA on the GQA view, but the structural fix is
the GPU.

---

## 6) The feed strategy

### 6.1 What has to be fed, and how much of it

Per token the engine routes **40 layers x 8 experts = 320 expert requests**,
each 1.6875 MiB of packed Q4_K or 2.0625 MiB of packed Q5_K - **566 MiB
(Q4_K layers) to 660 MiB (Q5_K layers) of expert weights consumed per
token**, of which the GPU must actually see whatever the arena does not
already hold.

### 6.2 The measured routing distribution (this pass)

Captured by wrapping `torch.topk` in the host router: **1280 router calls**
(32 tokens x 40 layers), 10 240 expert requests.

| Quantity | Value |
|---|---|
| distinct experts touched per layer (median over the 40 layers) | **38.5** of 256 (per-layer range 27-59) |
| mean overlap between a token's routed set and the previous token's | **3.14 of 8** (39%) |

**LRU arena miss rate vs slots per layer** (the arena holds the last S
distinct routed experts of that layer). The miss rate has two parts: the
**cold-start** misses (the first touch of each distinct expert - paid once per
expert, while the arena fills) and the **eviction** misses (an expert that was
seen before and has been pushed out - these recur every token). The
cold-start component is `distinct_per_layer / requests_per_layer`, i.e. 15.0%
at 32 tokens (38.5 / 256) and 53% at 4 tokens (17 / 32); the 4-token smoke
run reproduces exactly that (55.4% saturated), which is the check that the
sweep behaves.

| slots/layer | miss rate (cold + eviction) | eviction-only component | steady-state uploads/token | arena VRAM (Q5_K x15 + Q4_K x25 layers) |
|---|---|---|---|---|
| 8 | 73.3% | 58.3% | 187 | 585 MiB |
| 16 | 52.6% | 37.6% | 120 | 1170 MiB |
| 24 | 41.4% | 26.4% | 85 | 1755 MiB |
| **28** | **~36%** | **~21%** | **67** | **2047 MiB** |
| **32** | **31.9%** | **16.0%** | **51** | **2340 MiB** |
| 48 | 17.9% | 2.0% | 6 | 3510 MiB (over budget) |
| 64+ | 15.9% | 0.9% | 3 | 4680 MiB (over budget) |

The steady-state PCIe traffic that follows from the eviction column is
**90-120 MiB/token at 28-32 slots** (5.4-7.2 ms at the measured 16.6 GB/s),
not the 184-207 MiB the raw miss rate suggests - the cold-start experts are
fetched once each (2.7 GB for a full arena fill, spread over the first ~150
tokens) and then stay resident.

**Predictive arena** (the arena holds the union of the last T tokens' routed
sets - the regime the NVFP4 prewarm runs in):

| T | slots/layer | miss rate |
|---|---|---|
| 1 | 8 | 62.0% |
| 2 | 16 | 54.9% |
| 3 | 24 | 49.0% |
| 4 | 32 | 45.0% |
| 6 | 48 | 28.8% |

**What this means for the arena size:**

* The NVFP4 arena's **32 slots/layer** is *not* enough to hold the K-quant
  working set: the K-quant routing touches **38.5 distinct experts per layer**
  over 32 tokens, so 32 slots leave a **31.9% miss rate**, and the union of
  the last 4 tokens (32 slots) still misses **45%** of the requests. The
  NVFP4 path gets away with 32 slots because its arena is prewarmed from the
  router's own prediction; the K-quant path must reproduce that prediction.
* The **VRAM budget is the binding constraint, not the miss rate.** With the
  expert budget at **2048 MiB**, the arena can hold **28 slots/layer**
  (2047 MiB for this checkpoint's Q4_K/Q5_K mix). 32 slots/layer needs
  **2340 MiB**, i.e. the expert budget has to rise from 2048 to ~2400 MiB.
  The 4096 MiB experiment hit 8167 of 8176 MiB and spilled to GTT, so the
  arena must **replace** the expert LRU in the budget, not sit next to it
  (today's 7010 MiB peak already contains 2048 MiB of LRU; swapping the LRU
  for a 2340 MiB arena lands at ~7300 MiB).
* At 28-32 slots the arena still streams **90-120 MiB/token** over PCIe in
  steady state (5.4-7.2 ms at the measured 16.6 GB/s), plus a one-off
  **2.7 GB** while it fills (the cold-start column of the table above).
  **That traffic cannot be deleted on this hardware** - the whole expert bank
  is 21.3 GB and the card has 8 GB. It can only be **overlapped** with the
  GPU's work, which is why the queue-2 upload path is not optional in this
  design.
* The extra lever, if the budget is tight: the **input embedding table**
  (2048 x 248320 Q8_0 = 485 MiB) is only needed one row per token. Keeping it
  on the host and uploading 2 KB/token frees 485 MiB of VRAM, i.e. ~6 more
  arena slots per layer. The **output head** cannot move (it is a full
  485 MiB read per token on the GPU).

### 6.3 The three tiers

The changelog's framing is right and the measurement sharpens it:

| Tier | What it serves | Rate | Per token |
|---|---|---|---|
| 1. VRAM arena (hot) | the ~68% of requests the arena holds | 448 GB/s HBM, no PCIe | ~380 MiB/token, ~0.85 ms |
| 2. PCIe streaming (misses) | the ~16-21% that miss in steady state | **16.6 GB/s measured** (batch H2D) | 90-120 MiB/token, 5.4-7.2 ms, **must overlap** |
| 3. CPU tier | a fallback for the misses the PCIe cannot keep up with | 13.7 GB/s of quantized weights on 8 cores (llama.cpp's route) | 0.13 ms per expert, no PCIe |

**Tier 3 is a fallback, not the plan.** The fused K-quant kernel consumes
quantized weights at 96.98 GB/s - **7x the CPU tier's rate**. Moving expert
math to the CPU would cost 41 ms/token of CPU time to save 5.4-7.2 ms of PCIe.
The CPU tier only wins if the PCIe is saturated *and* the GPU is otherwise
idle. The parallel lane's `python/freetoken/kernel/csrc/cpu_moe/` is the
upstream CUDA/ROCm implementation of that tier (a persistent worker pool, a
GPU doorbell, ggml-backed GGUF experts, one MoE layer in flight); it is not
Vulkan-callable today (its doorbell is `cudaLaunchHostFunc` / HIP), so on this
path it is an architecture to copy, not a component to call.

### 6.4 Where the CPU slots in

* **Packing (host).** The mmap holds `gate_proj` and `up_proj` as separate
  tensors; the fused gateup8 shader wants them stacked. The stack is a host
  copy of 1.18 MiB per expert, done **once per expert** and cached in host
  RAM (`FTV_HOST_PACK_MB`, default 256 MiB - 1024 MiB caused swap death).
* **Prefetch.** The GPU router's output for layer L predicts layer L's
  experts for the next token; the predicted set is packed and uploaded on
  queue 2 while the GPU works on the current token. This is the fork's
  `_gpu_router_prewarm_async_uploads`, and it is what lifts the arena above
  the 31.9% LRU miss (the predictive table above shows what the *union of
  the last tokens* achieves; a real router prediction is better because it
  uses the current hidden state, not the previous token's set).
* **Page-cache advice.** The parallel lane's `python/freetoken/models/gguf/madvise.py`
  gives the engine `advise_expert_fetch` / `advise_expert_evict` /
  `proc_counters`. The two fetch sites (the pack and the LRU eviction) must
  call them, with `FTV_GGUF_MADV=2` for the full mode. On this box the file
  (22.8 GB) does not fit in RAM (~31 GB total, other tenants), so the kernel's
  page-cache LRU otherwise evicts the hot expert set and the decode swings
  between 0.11 s/tok and 15+ s/tok.

---

## 7) The ordered plan

**Baseline: 0.096 s/tok = 10.4 tok/s** (quiet box; 0.337 s/tok = 2.97 tok/s
loaded). The tok/s column is the **projected steady state after that step**,
derived from the measured buckets; it assumes the box is quiet.

| Step | Change | Projected tok/s | Risk | Cheapest validating experiment | Gate |
|---|---|---|---|---|---|
| **1. The K-quant arena** | `gguf_resident.py`: stop forcing `FTV_EXP_ARENA=0`; allocate the per-layer slots (28-32/layer, 2047-2340 MiB), fill lazily, upload misses with `batch_upload_multi`. Keep the LRU as the fallback. | **13.3** (host_fetch 25.0 -> 4.0) | VRAM: the arena must *replace* the LRU in the budget, not add to it; a full arena takes ~4 min to fill at 16.6 GB/s, so the first tokens are slow | Run with the arena on and `FTV_EXP_CACHE_MB` cut to 0 (the arena is the only store); watch the VRAM peak, the fill rate and the miss counter over 32 tokens | fork2 (maxdiff <= 5.5e-06, batched-vs-batched2 0.0), identity `[5844, 11, 13358, 11]`, VRAM peak < 7.5 GB |
| **2. The fused K-quant recorder** | `gguf_batched.py`: `batch_moe_kquant_record` -> `batch_moe_kquant_fused_record` over the arena (16 -> 2 dispatches/layer) | **14.7** (record 11.4 -> 4.5, GPU MoE 7.9 -> 5.9 ms) | The arena's slot layout must match the shader's `rows_per_slot` / `rowstride` exactly; the sentinel slot must never be routed to | The lane's gate already proves the kernel half (`test_kquant_fused.py`, 9 green, byte-identical for q4_k and q5_k). The engine half is one crossing: run the fused recorder against the arena and diff against the current path's output | `test_kquant_fused.py` green, fork2, batched-vs-batched2 bit-exact 0.0 |
| **3. The GPU router** | the MoE branch of `decode_step_batched`: `batch_rms1p_io` -> `batch_fp32_io` -> `batch_topk256_io` writing `arena_ids`/`arena_gs`; port the fork's prewarm | **16.4** (host_moe 7.6 -> 0.5) and it unlocks the prewarm, which is what lifts the arena above its 16% eviction miss | The top-k -> slot-id mapping and the sentinel contract; a tie or a stale id would route the wrong expert | Run the GPU router in CHECK mode against the host router for 50+ tokens and compare the paired (id, renormalized scale) | the GPU router's CHECK mode (h2 <= 1e-5, pair <= 1e-6, sentinel never routed) |
| **4. One command buffer per token** | `decode_step_batched`: open the batch once for the whole token, submit once (91 -> 1-2 drains) | **26** (the drain latency: 91 x ~0.28 ms) | A missing hazard barrier inside the merged buffer would reorder the work silently | Merge two adjacent layers' batches first (a 1-layer change): if the drain count halves and the token time halves, the effect is linear and the mechanism is confirmed | submits/token 91 -> ~2, barrier counters unchanged in kind, fork2, no hang |
| **5. GPU-resident attention and norms** | a `batch_flash_attn_kv_io` recorder + the GPU RMSNorm, so the host attention (3.95 ms) and host norms (2.9 ms) leave the critical path | **30-33** | The parity of a flash-attention shader with the host SDPA; the per-head layout | Implement the attention for **one** layer and diff it against `_full_attn_host` for 32 positions | the attention parity gate, fork2 maxdiff <= 5.5e-06 |
| **6. Overlap the PCIe** | the queue-2 upload path for the arena misses + the page-cache advice wiring (`FTV_GGUF_MADV=2`) at the two fetch sites | **26-33** (the 5.4-7.2 ms of steady-state PCIe either hides or it does not) | The upload must be ordered against the layer that consumes it (`batch_cross_barrier`); the advice must not free pages that are still mapped | Upload one layer's misses on queue 2 and check `batch_upload_async_stats` (queue-2 submits > 0, cross-barriers issued) plus the fence-wait median | `q2_submits > 0`, the fence-wait median does not grow, fork2 |
| **7. Re-test MTP** | with the GDN pull-down fixed by the parallel lane (~10 GB/s instead of 333 MB/s), the snapshot's 186 ms/round becomes ~6 ms | **+1.3x on the MTP arm** if the accept rate holds | MTP's verify and catch-up still use the sequential T>1 lane; it needs the batched K-quant recorder before it pays | Re-run the MTP benchmark with `FTV_MTP_SKIP_KV_SNAPSHOT=0` and compare the snapshot time and the accept rate | the MTP token-identical gate |

**Target: 26-33 tok/s**, versus 10.4 today and llama.cpp's 16.4-17.9 on the
same file. Steps 1-4 are the whole difference (10.4 -> 26); steps 5-7 are the
difference between "beats llama.cpp" and "beats it comfortably".

**If the projection is wrong**, the most likely reason is that `W_gpu` (the
GPU's true per-token work) is much larger than the ~20 ms inferred here. In
that case the plan's step 4 does not pay, and the answer is either a kernel
track (the GDN's per-layer cost) or the CPU tier. Section 10 is the experiment that
distinguishes these before any of the seven steps is built.

---

## 8) Risks

| Risk | What goes wrong | Mitigation | Residual |
|---|---|---|---|
| **The arena miss path and its WAW barriers** | A routed expert is not resident: it must be packed, uploaded into a slot, and the slot table updated *before* the MoE dispatch that reads it. A write-after-write on the same slot buffer (two layers writing the same slot) corrupts the other layer's weights. | Per-layer arena buffers (a slot in layer L is never touched by layer L+1); the fetch-then-record discipline already in `_record_moe_in` (collect every miss before recording any dispatch); the hazard-scoped barriers (`batch_ensure_ordered`) rather than full-memory barriers; the arena's slot table is host-side state updated before any dispatch that reads it. | The arena still re-misses 16-21% of the requests in steady state (51-67 uploads/token, plus the one-off cold-start fill). The upload is the WAW point. |
| **The GPU router's top-k output -> arena slot ids** | `batch_topk256_io` must emit the *slot* of each routed expert, not the expert id, and must never emit the sentinel `0x7FFFFFFF` for a routed expert. A stale slot table (the expert moved) routes the wrong weights: the output is garbage but not obviously broken. | The slot table is updated by the same host code that performs the upload, under one lock; the GPU router runs after the uploads are ordered; CHECK mode cross-validates the GPU's (id, scale) against a host shadow router; on divergence, fall back to the host router for that layer. The sentinel slot 0 is never allocated to a real expert (the lane's gate poisons it with NaN so a stray read fails loudly). | A divergence costs a fallback to the host router for that layer (7.6 ms/token worst case, once). |
| **MTP rollback with a GPU-resident GDN state** | The GDN state becomes device-local and authoritative. A rollback must restore the state as of the last accepted position, but the state now lives in a VRAM buffer that the host cannot cheaply read. | The parallel lane's `test_bar_readback.py` already gates the fix: the state is read back through `batch_download` into the pooled HOST staging (bit-exact, >= 1000 MB/s, ~10 GB/s measured) instead of reading the write-combining BAR mapping at 333 MB/s. The snapshot must be taken at the MTP draft/verify boundary and restored on rejection, exactly as the host state is today. | The MTP snapshot still costs a copy of ~63 MB (GDN) + the KV; the KV copy was measured at 6-7 ms/round, the GDN pull-down at 186 ms/round before the fix. |
| **The GDN pull-down at ~333 MB/s** | Reading the GPU's GDN state out of write-combining BAR memory costs 186 ms/round, which is the dominant cost of the MTP snapshot. | **Resolved by the parallel lane**: read from a device-local source into a pooled, non-write-combining host staging buffer via `batch_download`. The lane's gate asserts bit-exactness and a >= 1000 MB/s floor (the measured rate is ~10 GB/s, ~30x the old path). The engine must route every GDN-state read through `batch_download` and never touch the mapped BAR memory from the host. | The fix is a kernel-side/host-side path change, not an engine change; the engine only has to stop doing the slow thing. |
| **The VRAM ceiling** | 8 GB card; 4096 MiB of experts hit 8167 of 8176 MiB and RADV spilled to GTT. The arena must fit alongside the dense weights, the KV and the lm head. | The arena replaces the LRU in the expert budget (2048 -> ~2340 MiB), not add to it. Optional: keep the input embedding on the host (frees 485 MiB). Hard check: VRAM peak < 7.5 GB, no GTT growth. | Tight: ~900 MiB of headroom at 32 slots/layer. |
| **PCIe saturation** | The arena streams 90-120 MiB/token in steady state (plus a 2.7 GB cold-start fill). At 16.6 GB/s that is 5.4-7.2 ms/token, which does not fit inside a 32 ms token if it is not overlapped. | Queue-2 async uploads + the prewarm; the uploads for layer L are issued during layer L-1's GPU work. | If the overlap fails, the token is PCIe-bound at ~38-40 ms (25-26 tok/s) instead of 32 ms. |
| **The page cache** | The 22.8 GB file does not fit in RAM; the kernel's LRU evicts the hot expert set and the decode swings between 0.11 and 15+ s/tok. | The parallel lane's `madvise.py` (`FTV_GGUF_MADV=2`) at the two fetch sites, plus `proc_counters()` to prove the disk is not being hit. | The advice is opt-in and default-off; the engine lane has to wire it. |
| **Bit-identity regressions** | The fused recorder, the GPU router or a moved residual add changes the numerics. | Every step is gated: fork2 (seq-vs-batched <= 5.5e-06, batched-vs-batched2 bit-exact 0.0), `test_kquant_fused.py` (byte equality), the GPU router CHECK mode, the MTP token-identical gate, walk parity 0.0/0.0/<= 1e-5, identity `[5844, 11, 13358, 11]`. | The attention step (5) is the one that may not reach 5.5e-06 (a different SDPA implementation); if it cannot, it must stay behind the tolerance or be skipped. |
| **Host contention** | The box is shared; other lanes' load moves the absolute numbers by 3x. | Report the env with every number (`cached`, `mem_avail`, loadavg, RSS); use the quiet-box run for the design and the loaded run only for the split. | The 2026-10-21 GPU crash (a hard lockup under concurrent load) is a reminder to keep GPU runs short and never run two decodes at once. |

---

## 9) What the parallel lanes' work changes in this design

| Lane | State on disk | How it changes this design |
|---|---|---|
| **Fused K-quant shaders** (`linear_q{4,5}k_{gateup8,acc8}_f32.comp`) | on disk; `batch_kquant_gateup8_io`, `batch_kquant_acc8_io`, `batch_moe_kquant_fused_record` are bound in `ext_module.cpp`; gated by `tests/kernel/vulkan/test_kquant_fused.py` (**9 green in my run**: byte-identical vs the per-expert sequence for q4_k and q5_k, the sentinel-slot case, the batch path, the fused recorder vs a manual sequence, and the bench) | **The kernel half is done.** The plan's step 2 is now an engine change only, and the arena contract (slot layout, `Ids`, `Gs`, the slot-0 sentinel) is fixed by the lane's gate. The measured 0.1475 ms per 8 experts sets the GPU's MoE time at 5.9 ms/token and the plan's step 2 gain at ~3 ms/token. |
| **The BAR readback path** (`tests/kernel/vulkan/test_bar_readback.py`) | on disk; **green in my run**. The old path read write-combining BAR memory at ~340 MB/s (63 MB in ~185 ms); the new path reads through `batch_download` into pooled HOST staging at ~10 GB/s, with a bit-exactness assertion (int32 comparison, so NaN cannot hide a mismatch) and a >= 1000 MB/s floor | **It retires the worst risk in section 8.** The GDN pull-down is no longer a design decision: the engine must read the GDN state through `batch_download` and never from the mapped BAR memory. The MTP snapshot's 186 ms/round becomes ~6 ms, which is what makes step 7 (MTP) worth re-testing at all. |
| **The CPU MoE tier** (`python/freetoken/kernel/csrc/cpu_moe/`) | on disk (upstream, CUDA/ROCm: a persistent worker pool, a GPU doorbell via `cudaLaunchHostFunc`/HIP, ggml-backed GGUF experts in `kernels/gguf.cpp`, one MoE layer in flight) | **It is tier 3 of section 6.3, and it is the architecture to copy, not a component to call** (its doorbell is CUDA/HIP; there is no Vulkan variant). The measurement says the GPU path is 7x faster per unit of quantized weight (96.98 vs 13.7 GB/s), so the CPU tier is the fallback when the PCIe is saturated and the GPU is idle - not the plan. Its **ggml-cpu API bridge** (`ft_ggml_cpu_api`) is worth borrowing: it is how a Vulkan path could reach ggml's AVX2 K-quant dequant/GEMV without a from-scratch CPU kernel. |
| **The page-cache advice** (`python/freetoken/models/gguf/madvise.py` + `reader.py`) | on disk; opt-in via `FTV_GGUF_MADV` (0 off / 1 willneed / 2 dontneed); `advise_expert_fetch` / `advise_expert_evict` / `expert_slab_range` / `proc_counters` (`/proc/self/io` `read_bytes` is the only honest disk-read signal for an mmap'd file) | **It adds two call sites to the plan** (the pack and the LRU eviction) and it changes the measurement protocol: the design's numbers assume a warm cache, and `proc_counters()` is how one proves it. The `MADV_DONTNEED` + `fadvise(DONTNEED)` two-step is mandatory (fadvise alone frees nothing while the pages still have PTEs). |
| **The per-token breakdown lane** | the changelog's "222-325 round trips, ~90% of the time is synchronization" | **It is confirmed and sharpened.** This pass's fence-wait profile shows the 91 drains are latency, not kernel time (the median drain waits 0.42 ms, 12x the empty-submit cost, and 73% of them are under 0.5 ms). The breakdown lane's 83.7 MiB/token traffic figure is *lower* than the 167.1 MiB/token this pass measures; the difference is the expert cache's state (a cold-ish 16-token window with a 56% miss rate here vs the longer, warmer runs behind that figure). **I trust this pass's number for the traffic and the changelog's for the ratio.** |

---

## 10) The cheapest experiment that would falsify this design

**Open one command buffer for the whole token instead of 91, and measure the
fence bucket.**

*Where:* `decode_step_batched` in
`python/freetoken/models/qwen3_5_moe/vulkan/nvfp4_resident.py` (the shared
base for the GGUF path) - open the batch before the layer loop, submit after
it, and let the existing per-layer hazard barriers do the ordering inside.

*Cost:* one file, ~20 lines, no kernel work, no new op. It is step 4 of the
plan, done first and on its own.

*What it decides:*

| Outcome | Meaning |
|---|---|
| the fence bucket collapses from 45.4 ms to ~20 ms | the drain latency was ~0.28 ms x 91, the GPU's real work is ~20 ms, and the whole projection (26-33 tok/s) holds. **Build steps 1-3.** |
| the fence bucket stays near 45 ms | the GPU's per-token work really is ~45 ms, the 22-26 ms floor is out of reach without a kernel track (the GDN's per-layer cost first), and the CPU tier (llama.cpp's route) becomes the only way to 16+ tok/s. **Re-scope the plan around tier 3.** |

*Why this is the cheapest falsifier:* every other step of the plan (the arena,
the fused recorder, the GPU router) changes both the host work and the GPU
work at once, so a null result would not tell us which half failed. The
submit collapse changes **only** the synchronization, so it isolates the
quantity the whole design rests on.

*Supporting numbers already in hand:* the empty-submit cost (0.034 ms) and
the drain-wait profile (median 0.421 ms, p90 0.792 ms, 93 drains >= 1 ms)
bound the drain latency at 0.1-0.3 ms per drain, i.e. 9-27 ms/token of pure
synchronization - the range the experiment collapses.

---

## 11) Attribution, disagreements, and what I trust

* **My attribution vs the changelog's.** The changelog says "roughly 90% of
  our per-token time is synchronization, dispatch and host work, not
  bandwidth". This pass measures 54.6% host work + 45.4% fence wait, with
  the fence bucket being mostly drain latency. The difference is the traffic
  figure: the changelog says 83.7 MiB/token (an 85.8% expert hit rate), this
  pass measures 167.1 MiB/token and a 56% hit rate over a 16-token window
  with only 4 warm-up tokens. **I trust this pass's traffic** (it is the
  number this box produces right now) **and the changelog's ratio** (it comes
  from longer, warmer runs). The design is built so that either number works:
  the arena's benefit is the *hit rate*, and the traffic only has to be
  overlapped.
* **My attribution vs the other lanes'.** The fused lane measures the GPU's
  MoE at 0.1475 ms per 8 experts (96.98 GB/s of quantized weights). That
  implies a GPU-side MoE cost of 5.9 ms/token, which is **inconsistent with
  a 45 ms fence bucket being GPU work**. I trust the fused lane's number (it
  is a direct measurement of the kernel) and therefore read the fence bucket
  as synchronization. That is the single most consequential inference in this
  document, and section 10 tests it directly.
* **Absolute numbers.** Everything here is from a quiet box
  (`load=[3.07, 2.01, 1.83]`, `cached=10932 MiB`, `mem_avail=9331 MiB`,
  `rss_max=12755 MiB`) except where the loaded run is explicitly labelled.
  On the loaded box the same code ran at 2.97 tok/s with `host_fetch` at
  141 ms/tok. **Trust the split, not the tok/s, when the box is busy.**

---

## 12) Reproduction

```bash
# the benchmark that produced every number in section 1
cat /tmp/opencode/CyberTiel.gguf > /dev/null
FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \
  /home/cus/.venv-vllm-vulkan/bin/python tests/vulkan/bench_token_overhead.py \
  --tokens 16 --warmup 4 --runs 2 --arm idle --json /tmp/opencode/overhead_final.json

# the parallel lanes' gates this design depends on
PYTHONPATH=python /home/cus/.venv-vllm-vulkan/bin/python -m pytest \
  tests/kernel/vulkan/test_kquant_fused.py tests/kernel/vulkan/test_bar_readback.py -v
```

The benchmark writes `tests/vulkan/token_overhead_batched.json` by default
(next to itself) and takes `--json PATH`. It changes no engine or kernel
behaviour: every number comes from wall-clock wrappers around the already
exported extension functions and around the decoder's Python methods.
