# Why llama.cpp is faster on this box, and what we do about it

This is the comparison that drove the last round of work: our Vulkan engine
decodes the CyberTiel GGUF on an RX 5700 XT at ~10.4 tok/s (quiet box), while
llama.cpp's own Vulkan build on the *same file and the same card* does
16.4-17.9 tok/s. Everything below is measured on this machine.

## The headline

llama.cpp's advantage here is **not** that it is a better GPU engine. It is that
it does **almost no per-layer GPU synchronization in the MoE phase**, because it
computes most of the MoE **on the CPU**, reading the experts straight out of
RAM with ggml's AVX2 K-quant GEMV.

llama.cpp's own configurations on this box:

| llama.cpp config | prompt t/s | generation t/s |
| --- | --- | --- |
| `-ncmoe 41 -ngl 999` (ALL experts on the CPU) | 24.4 | 16.4 |
| `-ncmoe 30 -ngl 999` (30 layers' experts on the CPU) | 12.7 | **17.9** |
| default (`--fit on`) | 11.1 | 14.6 |
| `-ncmoe 30 -ngl 999 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.25` | 20.9 | 17.5 |

The configs that put the MoE **on the CPU** are the fast ones. Its thread sweep
at `-ncmoe 41`: t1 = 4.78, t2 = 8.04, t4 = 11.6-23.6, t8 ~19.6-22.3 tok/s, and
**t16 collapses to 3.8-4.2** (reproduced 3x). So llama.cpp's MoE sustains
~13.7 GB/s of quantized weights on this 5800X - about **half the PCIe link** -
and it saturates at 4-8 threads. (`-fa off` beats `-fa on` by ~13% here:
22.32 vs 19.61 tok/s.)

## Our side, attributed

From `tests/vulkan/bench_token_overhead.py` (quiet box, 10.4 tok/s = 0.096
s/tok):

| bucket | ms/token | calls/token |
| --- | --- | --- |
| **fence** | **45.35** | 91 |
| host_fetch | 25.01 | 640 |
| record | 11.38 | 969 |
| host_moe | 7.56 | 40 |
| host_attn | 3.95 | 10 |
| host_rms | 2.92 | 81 |

gpu_wait **45.4%** / host_work **54.6%**; 91 drains, 0 sync round trips, 772
host calls, 702 barriers, 167.1 MiB uploaded and 2.2 MiB downloaded per token.

Three things follow immediately:

1. **The GPU is not the slow part.** The architect's asymptote for today's
   kernels is 40 x (t_mix 0.20-0.35 + t_fill 0.10-0.25 + t_moe 0.30) =
   **22-26 ms/token = 40-45 tok/s**. Our GPU-side work is ~27-33 ms/token.
2. **The traffic is not the slow part.** 167 MiB/token at 16.58 GB/s is ~10 ms.
   (llama.cpp's C floor for a copy is 62.7 us CPU->BAR + 54.5 us GPU copy, i.e.
   ~15 GB/s with zero Python; our async path measures 146.6 us/copy.)
3. **The drains are the slow part.** The drain profile is median **0.421 ms**,
   p90 0.792, p99 3.081, max 3.441 - against an **empty submit at 0.034 ms**.
   91 drains per token is the whole gap: each one is a pipeline drain, not
   kernel time.

So the comparison is: **llama.cpp spends its time on the CPU (13.7 GB/s from
RAM, no GPU round trips in the MoE phase); we spend ours on 91 GPU pipeline
drains and 1,609 host calls per token.** Neither engine is bandwidth-bound.

## Why the tiers alone are not the answer

The three tiers (VRAM-hot experts on the GPU, PCIe-streamed misses on the GPU,
CPU-computed experts on the CPU) attack the *feed*, and our feed is already
only ~10 ms/token. On today's engine the tiers would move a few milliseconds
and add a CPU/GPU split with its own synchronization.

They become decisive only **after** the drain count collapses. The order is
therefore:

1. **One command buffer per token** (91 drains -> 1-2). This is the cheapest
   falsifier in the whole project: open a single batch for the whole token
   instead of one per layer and watch the fence bucket. If it does not
   collapse, the plan is wrong.
2. **A K-quant expert arena** (fixed slots per layer, replacing the LRU) plus
   the fused 8-expert recorder over it (16 -> 2 dispatches per layer) and the
   GPU router (ids and scales never round-trip).
3. **GPU-resident attention and norms** (host_attn 3.95 + host_rms 2.92 ms).
4. **The CPU as a feed tier** - prefetch and dequantize the next layer's
   experts on the 5800X while the GPU computes the current one, so the GPU
   never waits on PCIe. This is llama.cpp's shape, used as a *feed* rather than
   as the compute engine.
5. **MTP on top** - 1/passes_per_token, ~1.5x at the measured 0.69 accept rate.

Sizing from the measured routing distribution (38.5 distinct experts per layer,
median, range 27-59): 73.125 MiB per slot per layer, so **28 slots = 2047 MiB**,
which fits the 2048 MiB expert budget. Steady-state churn at 28 slots is ~16%
(51 uploads/token, 90-120 MiB/token) plus a one-off ~2.7 GB cold-start fill
that must be **overlapped**, not done synchronously.

## Where that lands

Projected: **26 tok/s** after steps 1-2, **30-33 tok/s** after step 3, and
1.3-1.5x more from MTP once the snapshot is cheap (the BAR-readback fix
already moved the GDN pull-down from 187.6 ms at 0.34 GB/s to 7.5-8.1 ms at
11-12.8 GB/s, bit-exact - so the published MTP verdict of 0.57-0.58x is stale
and must be re-measured).

That beats llama.cpp's 16.4-17.9 tok/s on the same card, which is the point:
**the GPU is ~100x better at this math than the 5800X is, but only if the
weights can reach the GPU without a pipeline drain per layer.** llama.cpp
avoids the drains by moving the work to the CPU; FreeToken should avoid them by
keeping the token on the GPU.

## Reproducing

    # our side: the attribution harness
    PYTHONPATH=python python tests/vulkan/bench_token_overhead.py --tokens 16 --warmup 4 --runs 2 --arm idle
    PYTHONPATH=python python tests/vulkan/bench_token_breakdown.py --tokens 16
    # the llama.cpp side
    llama-bench -m /tmp/opencode/CyberTiel.gguf -ngl 999 -ncmoe 30 -fa on -n 64 -p 0
    llama-bench -m /tmp/opencode/CyberTiel.gguf -ngl 999 -ncmoe 41 -fa on -n 64 -p 0

Both sides must be measured back to back on the same box: this shared machine's
page cache swings the same command by up to **1.83x** (cold 4.48 vs warm 8.20
tok/s), because the 22.8 GB GGUF does not fit in the ~13-19 GB of free RAM. The
page-cache state alone is a larger effect than most of the optimisations being
compared, so report the Cached figure from /proc/meminfo, the load average and
the RSS with every number.

See also [gpu-resident-token.md](gpu-resident-token.md) for the design and the
per-step plan, [kquant-kernels.md](kquant-kernels.md) for the fused dispatch and
the BAR-readback fix, and [page-cache.md](page-cache.md) for the page-cache
measurements (including why the advice knobs ship OFF).
