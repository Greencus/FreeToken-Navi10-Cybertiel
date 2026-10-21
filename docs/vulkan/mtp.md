# MTP (nextn) speculative decoding on the Vulkan Qwen3.5-MoE decoder

The CyberTiel GGUF ships a trained MTP block (`blk.40`: a full-attention
layer with its own experts plus four `nextn` tensors). Upstream FreeToken
drops that block; llama.cpp's server drafts with it. This port implements the
same draft/verify/accept loop on the resident Vulkan backend so a round can
buy more than one token per 40-layer pass.

Code: `python/freetoken/models/qwen3_5_moe/vulkan/mtp.py`.
Gates: `tests/vulkan/test_mtp.py`. Bench: `tests/vulkan/bench_mtp.py`.

## 1. What the block is

`qwen35moe` with `block_count = 41` and `nextn_predict_layers = 1`: the trunk
is layers 0..39, the MTP block is layer 40. It is a full-attention layer with
its own attention weights and its own routed + shared experts (`blk.40.ffn_*`),
plus:

| Tensor | Shape here | Role |
| --- | --- | --- |
| `blk.40.nextn.eh_proj.weight` | 4096x2048 Q8_0 | projects `norm(embed) ++ norm(h_prev)` to the block input |
| `blk.40.nextn.enorm.weight` | 2048 F32 | RMSNorm of the token embedding (raw weight) |
| `blk.40.nextn.hnorm.weight` | 2048 F32 | RMSNorm of the previous hidden row (raw weight) |
| `blk.40.nextn.shared_head_norm.weight` | 2048 F32 | RMSNorm before the shared lm_head (raw weight) |

There is no `nextn.embed_tokens` and no `shared_head_head`, so the block
reuses the main `token_embd` rows and the main `output` (lm_head) matrix. The
three nextn norms are **raw** weights (`y = norm(x) * w`), unlike the trunk's
zero-centered norms (`y = norm(x) * (1 + w)`); `mtp._rms_raw` keeps that
straight.

The MTP block is **not** a GDN layer: it has no `ssm_*` tensors and must never
go through the GDN recurrence.

## 2. The loop (mirrors llama.cpp)

Reference: `common/speculative.cpp::common_speculative_impl_draft_mtp`
(`draft()` and `process()`), `common/sampling.cpp::common_sampler_sample_and_accept_n`,
and the verify/accept block in `tools/server/server-context.cpp`.

State carried between rounds:

- `pending_id` - the last emitted token, **already in the output but not yet
  decoded** by any layer. Its KV cell gets written when the next round decodes it.
- `pending_h` - the target's post-`output_norm` hidden row at position
  `pos - 1` (zeros before the first round).

One round at trunk position `pos` (the number of committed cells):

1. **draft** (`MtpHead.draft`) - up to `n_max` steps through the MTP block
   alone. Step 0 pairs `pending_id` with `pending_h`; step j>0 pairs each
   drafted token with the draft's own `shared_head_norm` output. A step is
   kept only while its top-1 probability, renormalized over the draft
   sampler's top-10, is >= `p_min` (the gate fires *before* the token is
   accepted as a draft, as in llama.cpp).
2. **verify** (`MtpHead.verify`) - one batched trunk pass over the `k+1`
   candidate positions: the pending token plus the `k` drafts. On the
   batched decode base this is the multi-row walk `decode_rows_batched`
   (section 7, "Batched walk"): one command buffer per layer phase for all
   rows (GDN in-projections + recurrence, attention qkv / host SDPA / o_proj,
   and one shared MoE batch for all rows), with the GPU GDN mirrors
   authoritative afterwards. On the sequential base it falls back to the T>1
   chunk lane (`_gdn_chunk` / `_full_attn_chunk` / `_moe_chunk`).
3. **accept** (`accept_greedy`) - `common_sampler_sample_and_accept_n`:
   greedy sample each verify row, stop at the first draft mismatch, always
   take one bonus sample when every draft matched. The round commits
   `n_acc = len(result)` tokens and advances to `pos + n_acc`.
4. **catch-up** (`MtpHead.catchup`) - llama.cpp's `process()`: re-decode the
   same positions through the MTP block with the target's hidden rows shifted
   right by one, seeded with `pending_h`. After it every MTP KV cell carries a
   *target-side* h input, which is the distribution the block was trained on;
   the draft's own `h_row` feedback only affects the drafted token inputs.
5. **rollback** - a partial accept (`n_acc < k+1`) restores the trunk from a
   host snapshot taken before the verify, then re-runs the kept prefix as one
   batched pass. Causality makes that re-run row-for-row what the verify
   already computed; only the positions past `n_acc` stay undone. A full
   accept skips the copy entirely.

`pending_h` is then the verify batch's hidden row `n_acc - 1`, exactly
llama.cpp's `accept()` (`verify_h[min(n_accepted, n_rows - 1)]`).

0. **prompt catch-up** - before the first round, the MTP block decodes the
   whole prompt itself with the target's hidden rows shifted right, seeded with
   a zero `pending_h` (llama.cpp runs `process()` over every prefill ubatch and
   warns in `begin()` when the draft ctx misses prompt positions). Without it
   the first round's drafts attend to an empty KV cache and always reject -
   measured: the multi-sentence prompt's accept rate was 0.33 before this step
   and 0.67 after.

Greedy only (`temperature <= 0`): the accept rule is llama.cpp's server path,
which compares drafts against greedy samples. The modified rejection sampling
(`min(1, p_tgt/p_dft)`) lives only in llama.cpp's `examples/speculative` and is
not implemented here.

## 3. Why the rollback is a snapshot and not a KV trim

The trunk's per-token state is three things: the full-attention KV caches (10
layers, positional, easy to trim), the GDN recurrent state, and the GDN conv
history. The latter two are *cumulative*, not positional: a rejected row has
already mixed itself into them. So a partial accept cannot be undone by
truncating KV; it needs the pre-verify values. `MtpHead.snapshot`/
`restore` copy them (plus the KV caches) to a pre-allocated host buffer
(~95 MB at `max_len = 4096`).

Accepted cells beyond the new position are never touched: every attention
query only reads keys `<=` its own position, so stale draft cells are
invisible until a later round overwrites them.

## 4. Isolation from the trunk

The MTP KV lives in `MtpHead.k_cache` / `v_cache` and is installed under key
`40` only for the duration of a block call (`MtpHead._mtp_kv`). The trunk's
per-layer walk uses `cfg.layer_types` (40 entries) and never sees index 40,
and the trunk's own caches never see the block's cells. The block's dense
weights are pinned through the GGUF decoder's own `_pin_gguf`; its routed
experts join the decoder's VRAM LRU as one more `ExpertBank` (they are Q4_K in
this file, served by the same packed K-quant path as the trunk's blocks 15-40).

`enable_mtp(model)` is idempotent and additive: it extends the weight
provider's name set with the layer-40 names (the provider's `_fetch` already
parses `.layers.N` generically, so no mapping change was needed), pins the
block, and allocates the KV/snapshot buffers. Cost: ~80 MB of pinned VRAM
plus whatever the block's experts add to the LRU when they are routed to.

**Expert-weight seam.** `MtpHead` never touches weight bytes itself: the block's
routed experts are registered as one more `ExpertBank` in `model.expert_mats`
and every draft/verify/catch-up row goes through the decoder's own
`_moe_step` / `_moe_chunk` (the same path the trunk uses). That seam held when
the K-quant path switched from host dequant to packed in-shader kernels: the
swap happened entirely inside `gguf_resident.py` / `gguf/packed.py`,
`mtp.py` needed no change, and every MTP run stayed token-identical (section
7). The dense pins (`eh_proj`, the
nextn norms, lm_head) go through `_pin_gguf` once at `enable_mtp` time and are
per-token free.

## 5. Knobs

| Knob | Default | Meaning |
| --- | --- | --- |
| `n_max` | 3 | drafts per round (llama.cpp default; the fork reports `n_max 3` beats 5 sustained on a 35B-A3B) |
| `n_min` | 0 | discard the draft entirely when fewer than this many were kept |
| `p_min` | 0.0 | draft top-1 probability gate (llama.cpp server default 0.0; the fork recommends 0.25 for this model) |
| draft top-k | 10 | the probability gate renormalizes over the draft sampler's top-10, as llama.cpp's draft sampler does |

`generate_mtp(model, ids, max_new, eos=..., n_max=..., n_min=..., p_min=...)`
is the entry point; per-round timings and accept counters land on
`model.mtp.stats`.

CLI (`python -m freetoken.models.qwen3_5_moe.vulkan.cli`):

```bash
python -m freetoken.models.qwen3_5_moe.vulkan.cli --gguf CyberTiel.gguf \
    --mtp --mtp-n-max 3 --mtp-p-min 0.25 --temperature 0 \
    --prompt "..." --max-tokens 32
```

`--mtp` requires `--gguf` (the nextn block is a GGUF-only tensor set here)
and `--temperature 0` - both are rejected up front, before the weights pin.
The run prints the accept rate, rounds, rollbacks, passes/token against the
greedy baseline of 1.000, and the per-phase milliseconds; with
`--emit-json` those counters are stored under the prompt's `"mtp"` key.

End-to-end check on this box (`--mtp --mtp-n-max 3 --mtp-p-min 0.25
--temperature 0 --max-tokens 3`, prompt "The capital of France is", 171-token
chat-template prompt, `FTV_EXP_CACHE_MB=512`):

```
--- (368.0s, 2 tokens, 184.02s/tok, 0.01 tok/s) ---
mtp: drafted 3 accepted 2 rate 0.67 rounds 1 rollbacks 1 passes/tok 1.050
     (greedy baseline 1.000, -5.0% fewer full passes)
mtp ms: draft 515 verify 30982 catchup 744 rollback 20614
Paris
GGUF_CHAT_OK
```

Output is `Paris` + EOS (`token_ids [57590, 248046]`, so `max-tokens 3`
stopped early on EOS); the wall clock is dominated by the 171-token prefill
at a 512 MiB expert-cache budget, not by the two generated tokens. Equality
with the non-MTP greedy path is what `test_mtp_identity` above proves. This
check predates the packed path: at 184 s/tok it is the host-dequant one, and
the packed path runs the same CLI at ~1.3 s/tok (section 7).

## 6. Gates

`tests/vulkan/test_mtp.py` (needs `FREETOKEN_TEST_MODEL_GGUF`, skips cleanly
otherwise):

- `test_accept_greedy_rule` - the accept rule, no checkpoint needed.
- `test_prefill_with_h_matches_prefill_chunked` - the MTP prefill runs the
  production `prefill_chunked` unchanged and only captures the post-final-norm
  hidden rows through a temporary hook on the decoder's `_rms`; the last-row
  logits must stay **bit-identical**.
- `test_mtp_walk_parity` - the batched walk (`decode_rows_batched`) against
  both restore-siblings and the shipped sequential decoder, from one
  pristine snapshot with the harness gotcha (restore at the top of every
  arm) respected: walk-vs-walk and walk-vs-chained-`decode_step_batched`
  **bit-exact (0.0)**, walk-vs-`decode_step` within the fork2 bar
  (9.775e-06 <= 1e-5). Skips when the batched MTP lane is off.
- `test_mtp_identity` - 3 prompts x 6 greedy tokens by default
  (`FREETOKEN_TEST_MTP_TOKENS` raises it), MTP output identical to
  `model.generate`. This is the gate that catches a wrong `pending_h` shift, a
  MTP cell fed with the wrong hidden row, or an off-by-one verify row. It also
  prints each prompt's accept rate, rounds, rollbacks and passes/token.
- `test_mtp_skip_kv_snapshot_identity` - `FTV_MTP_SKIP_KV_SNAPSHOT=1` (the
  KV-eliding rollback snapshot, section 7) produces the same greedy ids as
  the full snapshot. Re-inits the head (`model.mtp = None` first -
  `enable_mtp` is idempotent, so a test must drop the cached head to flip
  the gate) and asserts the flag took.
- `test_mtp_cli_guards` - no checkpoint: `--mtp` without `--gguf`, and `--mtp`
  with `--temperature 1`, must both exit before the weights pin, with a message
  that names the missing piece.

The non-speculative numerics are untouched by this feature (no engine file was
modified), so `test_batched_fork2.py` and `test_gdn_kernel.py` still gate the
paths MTP reuses.

With the packed K-quant path as the weight path,
`FREETOKEN_TEST_MODEL_GGUF=<path> PYTHONPATH=python pytest tests/vulkan
tests/gguf_vulkan tests/kernel/vulkan -q` -> **120 passed, 5 skipped in
414-465 s** (2026-10-21, defaults as shipped: `FTV_GGUF_BATCHED=1`,
`FTV_EXP_CACHE_MB=2048`); the 5 skips are the safetensors/gpt-oss checkpoint
gates whose checkpoints are not on this machine. The closeout re-run after
the power cycle (2026-10-21 evening, page cache deliberately warmed, shipped
defaults): `tests/vulkan/test_mtp.py` **6 passed in 60.7 s** (walk parity
walk-walk **0.0**, walk-vs-chained-`decode_step_batched` **0.0**,
walk-vs-sequential **9.775e-06** with the shared accumulator, identity 3/3,
skip-KV identity green), `tests/vulkan` **32 passed, 4 skipped in 551.7 s**,
`tests/kernel/vulkan` **32 passed in 21.9 s**, and the CLI reference
`"Name three primary colors." --max-tokens 4 --temperature 0 --seed 1234`
returns **`[5844, 11, 13358, 11]` on both arms** (with and without `--mtp
--mtp-n-max 3 --mtp-p-min 0.25`). The A/B numbers in section 7 were taken
with `tests/vulkan/bench_mtp.py`, which is the same code path.

## 7. Measured results (RX 5700 XT, RADV, CyberTiel GGUF, batch 1)

Environment: `FTV_EXP_CACHE_MB=4096 FTV_HOST_PACK_MB=128`, 8 torch threads,
`max_len=2048`, prompt = 14 tokens (see `bench_mtp.py`), warm-up generation
discarded so the expert LRU/host dequant caches are hot. Same file, same box.

There are two weight paths in the tree. The tables under "Identity gate" and
"A/B bench" are the **host-dequant BF16** path (`FTV_GGUF_PACKED=0`), which was
the only path when they were taken and is kept as the reference. The last
subsection, "Packed path", is the **packed K-quant** path that is the default
today (see [kquant-kernels.md](kquant-kernels.md)).

### Identity gate (`tests/vulkan/test_mtp.py::test_mtp_identity`, 3 prompts x 6 tokens)

`passes/token` = `(rounds + rollbacks + (drafted + rounds)/40) / tokens the
executed rounds produced` (`1 + accepted + rounds`, so a final round that
over-delivers past `max_new` is not charged as wasted):

| prompt | drafted | accepted | rate | rounds | rollbacks | passes/token | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short factual | 9 | 2 | 0.22 | 3 | 3 | 1.050 | identical |
| short instruction | 12 | 3 | 0.25 | 4 | 3 | 0.925 | identical |
| multi-sentence | 6 | 4 | 0.67 | 2 | 1 | **0.457** | identical |

All three **IDENTICAL** to `model.generate`'s greedy ids. Full file,
`FREETOKEN_TEST_MODEL_GGUF=<path> pytest tests/vulkan/test_mtp.py -q` with
`FTV_EXP_CACHE_MB=4096 FTV_HOST_PACK_MB=128`: **4 passed in 626.85 s** (the
4th is the CLI flag-guard test). The same ids come back from the CLI with
and without `--mtp` (`[11751, 13, 198, 32, 13, 2912]`), so the gate's
baseline is the shipped driver's greedy path.

The prompt catch-up (section 2, step 0) is what took the multi-sentence prompt
from 0.33 to 0.67 accept: before it, round 1's drafts attended to an empty
draft KV.

### A/B bench (`bench_mtp.py --tokens 10 --warmup 4 --p-min 0.0,0.25`)

| config | s/tok (incl. 14-token prefill) | tok/s | accept | rounds | rollbacks | passes/token | identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline greedy | 10.602 | 0.09 | - | - | - | 1.000 | - |
| `n_max=3 p_min=0` | 15.940 | 0.06 | 7/9 = 0.78 | 3 | 1 | 0.430 | IDENTICAL |
| `n_max=3 p_min=0.25` | 10.429 | 0.10 | 5/7 = 0.71 | 5 | 1 | 0.630 | IDENTICAL |

Phase ms, `p_min=0.25`: prefill 45088, draft 872, verify 52082, catchup 1596,
rollback 4582. RSS max 11.4 GB, resident 7.9 GiB.

Batching probe (8 rows replayed from one snapshot):

| arm | total ms | per row |
| --- | --- | --- |
| sequential `decode_step` | 38585 | 4823 |
| batched verify, 1 row | 8537 | - |
| batched verify, 8 rows | 33499 | 4187 |

### What the numbers say

- **Correctness: yes.** Every timed run produced byte-identical tokens.
- **`passes/token` is not the cost.** At `p_min=0` the bench reports 0.430
  full passes per token - yet it was the *slowest* configuration (15.9 vs
  10.6 s/tok). The unit that costs money is the **row**: a round commits
  `rounds + accepted` tokens from `rounds + drafted` verify rows, plus one
  rollback row on a partial accept. Measured 12 verify + 1 rollback rows for
  10 committed tokens = **1.3 rows/token vs the baseline's 1.0**.
- **Batching is what MTP lives on, and here it is weak**: 4187 ms/row batched
  vs 4823 ms/row sequential = only **13% savings**, because a row's cost is
  dominated by host dequant of that row's routed experts, which batching
  cannot amortize (each verify row routes to different experts). Draft
  (0.13 s/step) and catch-up (0.13 s/row) are single-layer and cheap;
  rollback is one extra batched pass (4.6-8.5 s each).
- **Verdict on the current path: break-even at best** (p_min 0.25: 10.43 vs
  10.60 s/tok, inside the run-to-run prefill noise; p_min 0 loses). This
  matches llama.cpp's own measurement on this file (17.5 with MTP vs 17.9
  without - see `changelog.md`).
- **The unlock is the K-quant weight path**: once a row's cost stops being
  dequant and becomes dispatch + packed upload, a batched row amortizes the
  fixed per-dispatch cost that a sequential row pays alone, and MTP's
  rows/token (1.0 + a cheap draft/catch-up) can approach the ideal
  `k/(k+1)`. Until then the accept rate - not the algorithm - is the
  ceiling. **That unlock has since landed** - see the packed-path subsection
  below, where the same batching probe went from 13% to 2.3x per-row savings.

### Packed path (the default today: `FTV_GGUF_PACKED=1`)

Same bench, same prompt, same box, now with the experts served by the packed
K-quant shaders (`bench_mtp.py --tokens 16 --warmup 4`). Each `n_max` is one
process, so each row's MTP numbers sit against the baseline measured in the
same run - the non-speculative baseline moved between processes (0.51, 0.60
and 0.70 tok/s on three invocations) because the box was under load, which is
why only within-run ratios should be quoted.

| config | s/tok (incl. 14-token prefill) | tok/s | accept | rounds | rollbacks | passes/token | identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline greedy | 1.975 | 0.51 | - | - | - | 1.000 | - |
| `n_max=3 p_min=0` | 1.988 | 0.50 | 11/18 = 0.61 | 6 | 4 | 0.662 | IDENTICAL |
| `n_max=3 p_min=0.25` | **1.201** | **0.83** | 9/14 = 0.64 | 8 | 2 | 0.659 | IDENTICAL |
| `n_max=3 p_min=0.5` | 1.311 | 0.76 | 5/6 = 0.83 | 11 | 1 | 0.777 | IDENTICAL |
| baseline greedy (2nd run) | 1.423 | 0.70 | - | - | - | 1.000 | - |
| `n_max=5 p_min=0` | 2.691 | 0.37 | 13/30 = 0.43 | 6 | 5 | 0.744 | IDENTICAL |
| `n_max=5 p_min=0.25` | 1.843 | 0.54 | 11/20 = 0.55 | 8 | 2 | 0.669 | IDENTICAL |

Phase ms at `n_max=3 p_min=0.25`: prefill 3962, draft 518, verify 12293,
catchup 543, rollback 1776. RSS max 11.1 GiB, resident 8650 MiB (12.4 GiB RSS
at `n_max=5`, whose wider verify rows are the peak).

- **Best config: `n_max=3, p_min=0.25`, 0.83 vs 0.51 tok/s = 1.63x**, and
  token-identical to the non-speculative greedy path (the bench's own gate).
  Every one of the six MTP configurations came back **IDENTICAL**, including
  both `n_max=5` ones.
- **`n_max 5` is slower than `n_max 3`** at every `p_min` (0.37-0.54 vs
  0.50-0.83 tok/s) with a lower accept rate (0.43-0.55 vs 0.61-0.83). That
  reproduces the fork's finding on this model and llama.cpp's default of 3.
- **`p_min` has the expected shape**: 0.25 is the sweet spot; 0 drafts
  everything (0.50, break-even) and 0.5 drafts too little (0.76) to pay for
  the rounds it still needs.
- **Batching is what MTP lives on, and the packed path is what makes batching
  pay.** The same 8-row probe (section 7, "Batching probe") went from 4187 to
  458 ms/row at `n_max=3`: sequential `decode_step` 9369 ms for 8 rows vs
  batched verify 3667 ms, i.e. **2.3x per-row savings** against the
  host-dequant path's 13%. The per-row cost is no longer dominated by dequant
  of experts only that row routed to, so a multi-row verify amortizes dispatch.
  The remaining floor is still dispatch-bound (~3900 synchronous round trips
  per token on the sequential path), which is why the ideal `k/(k+1)` is not
  reached yet.

### On the batched decode base (2026-10-21, the current default)

The tables above predate two changes: the batched decode path became the
default (`FTV_GGUF_BATCHED=1`, see [kquant-kernels.md](kquant-kernels.md))
and the measurements below were taken with a **warm page cache** on a quiet
box, arms back to back inside one process (median of 3 rounds, whole
`generate()` including the 14-token prefill, `n_max=3 p_min=0.25`,
`/tmp/opencode/closeout_mtp.py`).

| base | non-speculative baseline | with MTP | ratio | identity |
| --- | --- | --- | --- | --- |
| batched (`FTV_GGUF_BATCHED=1`, default) | 292.8 ms/tok = **3.42 tok/s** | 868.9 ms/tok = 1.15 tok/s | **0.34x** | IDENTICAL (3/3 rounds) |
| sequential (`FTV_GGUF_BATCHED=0`) | 547.3 ms/tok = 1.83 tok/s | 744.5 ms/tok = 1.34 tok/s | **0.74x** | IDENTICAL (3/3 rounds) |

Accept rate 0.64 (9/14 drafted) in every round of every run, rollbacks 2 per
16 tokens - the accept machinery is unchanged and correct; what changed is
the baseline. Phase ms per round on the batched base: verify 7695-7862,
draft 355-412, catch-up 370-388, rollback 1226-1274.

**Verdict at that point: MTP is token-identical on the batched base but no
longer pays on either base.** Its verify and catch-up walked the sequential
T>1 K-quant lane (`_moe_chunk`, per-matrix keys), so a verify round cost
about as much as `k+1` sequential tokens while committing only `k/(k+1)` of
them; the 1.63x in the packed-path table above was measured against a 0.51
tok/s baseline that no longer exists. **The fix for that has since landed**
- the batched walk below.

### Batched walk, shared accumulator (2026-10-21, after `decode_rows_batched` + engine fix 5425148)

`MtpHead.verify` on the batched base runs the multi-row walk
`decode_rows_batched` (`nvfp4_resident.py`): per layer, ONE command buffer
for all verify rows - GDN in-projections + conv + recurrence + out_proj +
residual + readback (batch A, T rows), attention in-projections (batch 1)
and out_proj (batch 2) for T rows, and one **shared MoE batch for all T
rows** (upload -> zero `h_y_acc` -> `_record_moe_in` per row, all inside a
single `batch_begin`/`batch_submit`). The GPU GDN mirrors are authoritative
after a walk; `snapshot`/`restore` handle the pull-down. `FTV_MTP_BATCHED=0`
falls back to the old chunk lane; the gate (`head._batched`) also refuses
GPU-router and host-dequant sessions.

The shared MoE batch was blocked by an engine bug: all T rows sharing one
`h_y_acc` inside ONE command buffer made the `batch_download` readbacks come
back all-NaN for rows 0..T-2 (the walk therefore used one submit per row as
a workaround). The kernel lane fixed the hazard bookkeeping in
`ext_module.cpp` (commit `5425148`: mid-batch `free_resident` and
descriptor-pool resets are now deferred to `batch_deferred_drain()` at
`batch_submit`), verified shared-vs-per-row **bit-exact 0.0**, and the
workaround is gone from `nvfp4_resident.py`.

Gates (all green, 2026-10-21): `test_mtp_walk_parity` walk-walk **0.0**,
walk-vs-chained-`decode_step_batched` **0.0**, walk-vs-sequential
**9.775e-06** (bar 1e-5) - with the shared MoE batch in place;
`test_mtp_skip_kv_snapshot_identity` (below); `test_mtp_identity` and the
CLI identity (`[5844, 11, 13358, 11]` with and without `--mtp --mtp-p-min
0.25 --mtp-n-max 3`) unchanged; `tests/vulkan/test_mtp.py` **6 passed**,
`tests/vulkan` **32 passed / 4 skipped**.

`bench_mtp.py` - final closeout numbers (interleaved arms, median of 3
rounds, 32 tokens, `--n-max 3 --p-min 0.25`, shipped env defaults
`FTV_EXP_CACHE_MB=2048` / `FTV_HOST_PACK_MB=256`). Two sessions after the
power cycle, both quoted because the box is shared: session A (18:33, after
a deliberate 3-pass `cat` of the 22 GB GGUF - Cached 10.9 -> ~18 GiB, the
file cannot fully fit alongside running apps so the warm is partial; load
4.2-6.6, swap 1.6-1.9 GiB, Cached ~14.7-15.0 GiB) and session B (19:00,
both snapshot variants run back to back in one shell; load 0.8 -> 4.8, swap
4.7 GiB pushed up by other tenants, Cached 12.9 -> 10.9 GiB declining
through the run). RSS max 14.1-14.3 GB; resident 6416 MiB (VRAM end
7.1-7.8 GiB whole-GPU including other tenants); round trips counted with
`bench_gguf_packed.py`'s `_SYNC_KEYS` (sync submit+fence ops via
`model.stat_calls`):

| arm | wall ms/tok | tok/s | decode ms/tok (tok/s) | identity | accept | passes/tok | rows/tok | submits/round | rt/tok | snapshot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (`batched=True`) | 170.2-177.7 | 5.63-5.88 | 100.7-105.7 (9.46-9.93) | - | - | 1.000 | 1.000 | 2913 (91.0/tok) | 769.5 | - |
| MTP `n_max=3 p_min=0.25` (default snapshot) | 298.1-314.2 | 3.18-3.35 | 221.2-232.4 (4.30-4.52) | **3/3 both sessions** | 0.69 (20/29) | 0.564 | 1.312 | 1692 (52.9/tok) | 868.3 | 190-194 ms/round (77-79 ms/tok) |
| MTP `FTV_MTP_SKIP_KV_SNAPSHOT=1` | 298.1-306.6 | 3.26-3.35 | 221.2-226.5 (4.41-4.52) | **3/3 both sessions** | 0.69 | 0.564 | 1.312 | 1692 (52.9/tok) | 868.3 | 184-187 ms/round (75 ms/tok) |

MTP is **2.9x faster than on the old sequential T>1 lane** (868.9 ->
298-314 ms/tok) but still measures **0.57-0.58x wall / 0.44-0.46x decode**
of the baseline - on both snapshot variants. Phase stats per 32-token run
(session A default: prefill 2404, draft 304, verify 3597, catch-up 320,
rollback 423, snapshot 2464; session B default: 2590 / 317 / 3753 / 328 /
429 / 2526), i.e. per token: draft 9.5-9.9 + verify 112.4-117.3 +
catch-up 10.0-10.3 + rollback 13.2-13.4 + snapshot 77.0-78.9 = 222-230 vs
the measured decode-only 226-232 (the remainder is lm_head/sampling).

1. **Rows, not passes, are the unit.** At accept 0.69 a round commits
   `accepted + 1` tokens from `drafted + 1` rows: 42 verify rows for 32
   tokens = **1.31 rows/token**. `passes/token` 0.564 counts a T-row verify
   as one pass and overstates the saving.
2. **Verify costs 85.6-89.4 ms/row (3597-3753/42), 0.81-0.89x the
   baseline token (100.7-105.7).** The walk's batching works - the bench's
   own probe agrees (84.1 ms/row at T=8 vs 223.6-231.0 ms/row sequential,
   2.7x) - but the per-row saving is only ~11-19%, so 1.31 rows/token turns
   ~101-106 into ~221-232. The shared MoE batch brought MTP to **52.9
   submits/token vs the baseline's 91** (1692 vs 2913 per run; the walk
   itself is 30 GDN + 20 attention + 40 MoE = **90 submits per verify round
   regardless of T**, plus draft/catch-up/rollback walks), yet the wall
   clock did not move (305.3 before the shared batch, 298-314 after): those
   fence waits were GPU-work attribution, not the tax.
3. **Fewer submits, MORE sync round trips.** The `stat_calls` counter
   (sync submit+fence ops: `_pinned_affine` x3 per call, batch variants
   x2, expert io) measures MTP at **868.3 rt/tok vs the baseline's
   769.5** - MTP's extra host sync work (lm_head per draft step, catch-up,
   rollback re-verify, embed) more than cancels the submit savings. The
   round trips are deterministic (identical across every round of every
   run).
4. **`snapshot` is 190-194 ms per round (77-79 ms/tok)** - 68-78% of a
   whole baseline token spent before any drafting shows up - timed as
   `ms_snapshot`. Decomposed with probe scripts: the GDN/conv pull-down is
   **186 ms**, the host KV copies only **6 ms**. The pull-down is linear in
   bytes (6.1 ms per 2 MiB -> 177 ms per 60 MiB, ~333 MB/s): a CPU
   `memcpy` out of WC BAR memory - `make_host_ssbo` requests
   `HOST_VISIBLE` only, so `vku::download` reads mapped GPU memory at PCIe
   read speed. That fix is kernel-lane territory: `HOST_CACHED` download
   staging (or a D2D snapshot op that keeps the copy on the GPU) would cut
   it to milliseconds.
5. **`FTV_MTP_SKIP_KV_SNAPSHOT=1` (opt-in) removes the 6 ms KV half and
   CHANGES ROLLBACK SEMANTICS.** Measured gain: 190-194 -> 184-187
   ms/round (~2.6 ms/tok of snapshot; decode medians 226.3 -> 221.2 and
   232.4 -> 226.5 ms/tok), identity **3/3 in both sessions** and green in
   `test_mtp_skip_kv_snapshot_identity` - so it does not break tokens.
   The semantic difference: with the flag, `MtpHead.restore()` puts back
   only the GDN state and conv history; the trunk KV is NOT restored, so
   rejected rows' KV cells past the rollback position keep their **stale
   draft content** instead of the pre-verify values. That is sound today
   only by the unread-cells argument (section 3: every KV cell is written
   before it is read - queries attend only `<=` their own position, and a
   later round overwrites the stale cells before they can be read), so a
   future change that read those cells would silently diverge from the
   full-copy path. It stays **opt-in** for exactly that reason; the default
   keeps the full-copy rollback it shipped with.

**Win condition, honestly stated (measured, not projected).** The
decode-only arithmetic: verify (1.31 rows x 85.6-89.4 = 112-117) +
snapshot (77-79) + draft/catch-up/rollback (32.7) = 221-232 ms/tok against
the baseline's 100.7-105.7. Removing the snapshot entirely (kernel fix)
still leaves ~145-150 (0.69-0.71x); perfect accept (1.0 row/token) without
it still leaves ~118 (0.87x). Break-even needs **both** levers that live
outside this lane: (a) the snapshot pull-down fixed kernel-side
(-77 to -79 ms/tok), and (b) per-row verify cost from 85.6-89.4 down to
~52 ms (-40%, cross-row expert-weight amortization in the MoE GEMM - the
M>1 row direction), with any accept-rate gain on top; the skip-KV flag's
6-7 ms/round does not change this picture. The algorithm itself measures
healthy: accept 0.69, identity 3/3 on every arm, fewer command-buffer
submits/token than the non-speculative baseline (52.9 vs 91). Until those
two levers land, `--mtp` is token-identical and correct but still slower
than not drafting - use it for correctness work, not speed; the numbers
live here.


## 8. Known limits / follow-ups

- **Greedy only.** Sampling targets need rejection sampling (not implemented,
  matching llama.cpp's server path).
- **Rollback cost.** A partial accept pays one extra batched trunk pass over
  the kept prefix (`n_acc` rows, usually 1). Avoiding it needs per-layer
  snapshots of the *inputs* rather than the states (valid by causality: the
  GDN layer output at row j depends only on rows <= j), or a per-row GDN state
  snapshot - both are strictly more memory for a smaller tail.
- **Draft h feedback is off-distribution** for a single trained MTP module:
  step j>0 feeds the draft's own `shared_head_norm` output back into `eh_proj`,
  which training never saw (with `nextn_predict_layers = 1` the block's
  training input was always the main model's hidden). This is what llama.cpp
  does; the accept rate is what it is because of it.
- **Wired into `cli.py` (`--mtp`), not into `engine.py`.** The driver stays a
  standalone function so the non-speculative path is untouched - `engine.py`
  and `resident.py` are shared by the K-quant and batch work, and nothing in
  MTP needs to live there: `generate_mtp` is one import plus one call.
- The verify is a *batched* trunk pass, not `k+1` sequential ones. On this
  backend that is the assumption the bench measures (see the batching probe in
  `bench_mtp.py`); if batching stops amortizing dispatch, MTP loses its edge
  and the gate stays green while the bench shows it. That happened once
  (0.34x on the batched base while the verify still walked the sequential
  T>1 lane) and the batched walk `decode_rows_batched` fixed it - and the
  engine hazard fix (commit 5425148) then let the MoE phase collapse to one
  shared batch (52.9 vs 91 submits/token, section 7). MTP still measures
  **0.57-0.58x the baseline after the closeout run** because two costs
  dominate instead: 1.31 rows/token at 0.69 accept x 85.6-89.4 ms/row, plus
  the 190-194 ms/round rollback snapshot whose GDN pull-down reads WC
  staging memory at ~333 MB/s (the skip-KV flag only removes its 6 ms KV
  half, and stays opt-in because it changes rollback semantics - section 7).
  The next levers are the kernel's download-staging mode (HOST_CACHED or a
  D2D snapshot op) and per-row verify cost (the M>1 row direction), not the
  draft/verify/accept logic.
