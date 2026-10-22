# Page-cache advice for the CyberTiel GGUF (madvise / posix_fadvise)

Scope: **opt-in, default-off** page-cache advice for the 21.3 GiB CyberTiel
GGUF (`python/freetoken/models/gguf/`), plus the measurements that say where
advice helps, where it hurts, and which kernel facts any engine wiring must
respect. No engine or kernel files were edited; the call sites the engine
lane must land are documented in §6 only.

Knob: `FTV_GGUF_MADV` (default `0` = off, keep it off):

| value | behavior |
| --- | --- |
| `0` | off (default) - no advice is ever issued |
| `1` | `POSIX_FADV_WILLNEED` on expert-slab ranges at fetch |
| `2` | `1` + `POSIX_FADV_DONTNEED` on slab ranges the engine evicts |

## 1. The module

`python/freetoken/models/gguf/madvise.py`:

- `madvise_mode()` - reads `FTV_GGUF_MADV` (nonzero numeric string; `atoi`
  semantics like `FTV_NVFP4_M1`, so `"true"` stays off).
- `register_mapping(path, mmap_obj)` - called from `reader._reader()` on the
  reader's `np.memmap._mmap`; the live mapping is what makes real eviction
  possible (§2).
- `advise_willneed / advise_dontneed(path, offset, len)` - gated on the knob,
  never raise, memoize realpath + fd + size, thread-safe.
- `drop_file_pages(path)` - evict the whole file from *this process* (zap
  first, fadvise second - order matters, §2). Ungated by design: used by the
  bench cold-start and safe to call any time.
- `expert_slab_range(index, tensor_name, expert)` / `expert_ranges(...)`,
  `advise_expert_fetch(...)`, `advise_expert_evict(...)` - map one packed
  K-quant expert slab (gate/up/down, Q4_K or Q5_K) to file bytes.
- `ProcCounters` / `proc_counters()` / `.delta()` - honest fault/IO counters
  (§2); `advice_stats()` / `reset_advice_stats()` - cumulative call/byte/time
  counters; `close_advice_fds()`.

Tests: `tests/gguf_vulkan/test_page_cache_advice.py` (gating, no-op on absent
files, integrity through a live view after DONTNEED, range math for Q5_K
k=256 / row 210+144, counters, realpath registration).
Bench: `tests/gguf_vulkan/bench_page_cache.py` (`--mode micro|decode|all`).

## 2. Kernel facts measured on this box (7.2.4-zen2-1-zen, xfs, NVMe)

These are load-bearing; every one contradicts an intuitive assumption:

1. **`posix_fadvise(DONTNEED)` is a no-op for pages with PTEs in our
   process.** It reaches `invalidate_mapping_pages`, which skips
   `page_mapped` pages: 0 of 128 MiB were freed fadvise-only.
   **Fix: `mapping.madvise(mmap.MADV_DONTNEED, off, len)` on the reader's
   `np.memmap._mmap` first (zaps PTEs), then `posix_fadvise(DONTNEED)`.**
   Order matters. After that, 527/528 MiB were evicted and the re-read data
   is byte-identical (blake2b over the range: warm/control/after-zap/after-
   fadvise all `MATCH` - integrity holds; proven by
   `/tmp/opencode/debug_dontneed_integrity2.py`).
2. **`majflt` is ~0 for mmap file reads.** Refaults of file pages are counted
   as *minor* faults - majflt cannot be used as a cold signal here.
   Also `fault-around` maps ~16 pages per file fault, so `minflt` is not a
   page count either. **`/proc/self/io read_bytes` is the only honest disk
   signal.**
3. **`WILLNEED` does not prefetch on this kernel** for ranged advice into a
   clean cache: 64-128 MiB advised, ~0.5 MiB read during a 1-2 s sleep;
   whole-file WILLNEED trickles 231 MiB in 2 s (~115 MB/s) while demand
   paging sustains ~2.5 GB/s. It is *not* a cold-start accelerator.
   (Caveat: in-engine with active faults it still shaved ~9% off cold decode,
   §5 - readahead piggybacks on the demand faults.)
4. **Whole-file eviction is 242 ms one-off** (`drop_file_pages` on 21.3 GiB:
   one fadvise covering the file, no PTEs mapped yet at bench cold-start).

## 3. Advice costs (measured)

| call | cost |
| --- | --- |
| `WILLNEED`, micro (resident range) | **6.4 us/call** |
| `WILLNEED`, decode-path, warm | **7.5 us/call** (1.12 ms/tok at 150 calls/tok) |
| `WILLNEED`, decode-path, cold | **35.9 us/call** (5.40 ms/tok; does real readahead) |
| `DONTNEED`, micro (mostly-cold) | **10.1 us/call** (max 363 us) |
| `DONTNEED`, decode-path on live PTEs | **~59 us/call** (275 calls/tok) |
| whole-file evict | **242 ms** one-off |

Fetch-path advice count: **150 willneed calls/decode-token** (2405 per
6-token run); evict-path (mode 2): **275 dontneed calls/token**.

**Harness lesson for the engine lane:** the first bench wired the evict hook
as `set(cache)`-snapshot on *every* `_cache_put` - that alone cost
**+40 ms/tok** warm (isolated by the `seam0` arm: seams installed, knob
gated). Gate on the mode *first* and detect victims in O(1) (front-prefix of
an LRU), or the seam - not the kernel - is what you measure. After the fix
`seam0` is within noise of `off`.

## 4. Decode bench (final, interleaved arms)

`tests/gguf_vulkan/bench_page_cache.py --mode decode --tokens 6 --runs 3
--arms off,seam0,adv1,adv2`, `FTV_EXP_CACHE_MB=2048 FTV_HOST_PACK_MB=128`,
fresh process per run, arms interleaved per round so load drift hits all
arms, cold arms evict the file first (via `drop_file_pages`), advice seams
installed at the two documented call sites (§6). Machine load drifted
1.4 -> 5.3 avg during the run; medians of 3:

| arm | cold ms/tok | cold MiB/tok read | warm ms/tok | warm MiB/tok read | warm minflt/tok |
| --- | --- | --- | --- | --- | --- |
| `off` (before) | **223.2** (216.6/251.8) | 90.8 | **122.0** (121.2/122.3) | 0.0 | 1826 |
| `seam0` (harness only) | 215.6 (213.6/224.4) | 90.9 | 131.5 (120.7/152.7) | 1.7 | 1890 |
| `adv1` = willneed | **202.5** (193.4/206.0) | 83.6 | 129.6 (128.9/132.6) | 3.7 | 1540 |
| `adv2` = willneed+dontneed | 342.7 (337.8/379.2) | 204.1 | 283.4 (274.3/284.4) | 131.5 | 3892 |

Read as tok/s: off cold **4.48** / warm **8.20**; adv1 cold **4.94** /
warm **7.72**; adv2 cold **2.92** / warm **3.53**.

Verdicts:

- **Job-1 evidence (the thrash mechanism):** page-cache state alone swings
  this decode **1.83x** (223.2 -> 122.0 ms/tok median, 2.08x max) with *zero*
  change to engine, model or GPU. That is the quiet-box version of the
  reported 0.11 s/tok <-> 15+ s/tok swings - under heavier tenant pressure
  (other tenants evicting our cache) the same mechanism produces the
  pathological side. Earlier bench run caught it live: a "warm" arm spiked
  to 177.8 ms/tok with 38.6 MiB/tok disk reads when another tenant pushed.
- **adv1 (willneed at fetch): neutral-to-mildly-positive.** Cold -9%
  (202.5 vs 223.2, reads 83.6 vs 90.8 MiB/tok), warm +7.6 ms/tok which is
  ~1.1 ms/tok measured advice time + noise. It does not prefetch (§2.3) but
  costs almost nothing.
- **adv2 (donthneed at LRU eviction): harmful on the current engine.**
  Warm 283.4 vs 122.0 ms/tok (**2.3x worse**) with 131.5 MiB/tok
  *self-inflicted* disk reads; cold 1.5x worse. Only ~17-24 ms/tok of that
  is advice syscall time - the rest is policy: this LRU evicts at 44% miss
  churn (16-token window hit rate ~56%), so an evicted slab is usually
  re-fetched within a few tokens, and DONTNEED guarantees that re-fetch hits
  the NVMe instead of the page cache. It does visibly shrink resident cache
  (RSS 11.5 -> 9.9 GB) - the mechanism works, the trigger is wrong.
- adv2's variance is narrower (274-379 ms/tok spread vs off's 121-252) but
  that is "consistently disk-bound", not an improvement.

Micro bench (`--mode micro`, saved in `/tmp/opencode/page_cache_micro.txt`):
M1 overhead as §3; M2 DONTNEED-through-live-mapping 527/528 MiB freed,
intact=True; M3 2.75 GiB cold-start fill 1.14 s cold / 0.21 s warm,
willneed-then-fill 1.10 s (no prefetch benefit); M4 whole-file willneed
trickle (§2.3).

## 5. What to wire, and where (recommendation)

- Keep `FTV_GGUF_MADV=0` as shipped. adv1 is safe if fetch-side readahead is
  ever wanted; adv2 must not be wired at the *LRU* eviction seam.
- **DONTNEED belongs at true replacement points**: the fixed-slot expert
  **arena** design (28-32 slots/layer, ~73 MiB/slot, hot set pinned in VRAM)
  is the correct trigger - there, eviction means a slot is being *reused*,
  not merely LRU-cold. Wire it at the arena slot-replacement site and
  re-bench with this harness (`--arms off,adv2`).
- The arena cold-start fill (~2.7 GiB) does not need WILLNEED either: demand
  paging already runs it at ~2.5 GB/s (M3: 1.14 s); willneed measured 1.10 s
  and trickles at 115 MB/s when clean.

## 6. Exact engine call sites (document only - do not let advice land blind)

- **Fetch** (`advise_expert_fetch`, mode>=1): before the slab view is
  created in `gguf_batched.py::_gguf_gu_handle` (~line 298, gate+up) and
  `::_gguf_dn_handle` (~line 351, down), `gguf_resident.py:289`,
  `hotset_resident.py:269` - keyed by `(layer, proj, expert)`.
- **Evict** (`advise_expert_evict`, mode==2): the LRU victim loop inside
  `_cache_put` (`nvfp4_resident.py`, `hotset_resident.py:289` override) -
  **but see §5: only wire this at the arena replacement site.**
- The bench installs both seams itself
  (`bench_page_cache.py::_install_advice_seams`) by wrapping
  `packed.packed_expert` (module global - `gguf_provider` binds it by
  import, so it must be patched there too) and the instance `_cache_put`.

## 7. How to reproduce

```bash
cd FreeToken-Navi10-Cybertiel
V=/home/cus/.venv-vllm-vulkan/bin/python
export PYTHONPATH=python FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf

$V -m pytest tests/gguf_vulkan -q                      # gates (69 passed)
$V tests/gguf_vulkan/bench_page_cache.py --mode micro  # M1-M4
$V tests/gguf_vulkan/bench_page_cache.py --mode decode \
   --tokens 6 --runs 3 --arms off,seam0,adv1,adv2      # §4 table
```

Identity gate (must print `token_ids: [5844, 11, 13358, 11]`):

```bash
FTV_EXP_CACHE_MB=2048 FTV_HOST_PACK_MB=128 $V -m \
  freetoken.models.qwen3_5_moe.vulkan.cli \
  --gguf /tmp/opencode/CyberTiel.gguf \
  --prompt "Name three primary colors." \
  --max-tokens 4 --temperature 0 --seed 1234
```

Machine-hygiene reminders apply as always: check `uptime` first, arms are
interleaved for a reason, and do not compare runs across a load ramp.
