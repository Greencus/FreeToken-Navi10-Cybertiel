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
   candidate positions: the pending token plus the `k` drafts. This is the
   production prefill layer walk (`_gdn_chunk` / `_full_attn_chunk` /
   `_moe_chunk`), so its rows cost one dispatch per kernel, not one per token.
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
- `test_mtp_identity` - 3 prompts x 6 greedy tokens by default
  (`FREETOKEN_TEST_MTP_TOKENS` raises it), MTP output identical to
  `model.generate`. This is the gate that catches a wrong `pending_h` shift, a
  MTP cell fed with the wrong hidden row, or an off-by-one verify row. It also
  prints each prompt's accept rate, rounds, rollbacks and passes/token.
- `test_mtp_cli_guards` - no checkpoint: `--mtp` without `--gguf`, and `--mtp`
  with `--temperature 1`, must both exit before the weights pin, with a message
  that names the missing piece.

The non-speculative numerics are untouched by this feature (no engine file was
modified), so `test_batched_fork2.py` and `test_gdn_kernel.py` still gate the
paths MTP reuses.

With the packed K-quant path as the weight path,
`FREETOKEN_TEST_MODEL_GGUF=<path> FREETOKEN_TEST_MTP_TOKENS=12 PYTHONPATH=python
pytest tests/kernel/vulkan tests/vulkan -q` -> **45 passed, 4 skipped in
194.5 s**; the 4 skips are the safetensors/gpt-oss checkpoint gates whose
checkpoints are not on this machine. The A/B numbers in section 7 were taken
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
  and the gate stays green while the bench shows it.
