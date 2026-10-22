"""Page-cache bench for the memory-mapped GGUF expert bank.

Answers, with numbers instead of suspicion, what the page-cache advice in
``freetoken/models/gguf/madvise.py`` is worth on this box:

- ``--mode micro`` (CPU+disk only, no GPU): advice syscall overhead on the
  real file; what a whole-file WILLNEED does (the documented failure mode);
  what the arena's ~2.7 GB cold-start fill costs cold / with WILLNEED /
  warm; and whether DONTNEED through a live reader mapping really frees
  pages (read_bytes after the advise) without corrupting bytes.
- ``--mode decode`` (GPU): the same decode command run as fresh processes
  under 4 arms - advice off/on x cold/warm file cache - reporting s/tok,
  min/median/max over ``--runs``, /proc/self/io read_bytes per token and
  minor faults per token. Cold = the file's clean pages evicted with
  ``posix_fadvise(DONTNEED)`` right before the process starts (no root
  needed; requires no other process to hold the file mapped - it warns).
  The advice arms install the two call sites the ENGINE is supposed to
  land (fetch -> ``advise_expert_fetch`` before the slab view is created;
  evict -> ``advise_expert_evict`` in the LRU victim loop) by wrapping
  ``packed.packed_expert`` and the instance's ``_cache_put`` - the bench
  demonstrates the wiring; the engine lane lands it for real.

    FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \\
        python tests/gguf_vulkan/bench_page_cache.py --mode micro
    FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \\
        python tests/gguf_vulkan/bench_page_cache.py --mode decode --tokens 6 --runs 3

Every line prints the machine state (load average, Cached MiB, RSS) because
on a 31 GB box shared with other tenants, a number without it is noise.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import resource
import statistics
import subprocess
import sys
import time

GGUF = os.environ.get("FREETOKEN_TEST_MODEL_GGUF", "")
# Same prompt ids as tests/vulkan/bench_gguf_packed.py (GGUF-tokenized).
PROMPT = [14917, 557, 264, 1248, 1834, 13, 353, 14428, 2046, 13569, 321, 1801, 14799, 13]


def _machine() -> str:
    load = os.getloadavg()
    cached_mib = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Cached:"):
                    cached_mib = int(line.split()[1]) / 1024.0
                    break
    except OSError:
        pass
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return f"load={load[0]:.2f}/{load[1]:.2f}/{load[2]:.2f} cached={cached_mib:.0f}MiB rss_max={rss_mib:.0f}MiB"


def _others_map(path: str) -> list[int]:
    """PIDs other than ours that have ``path`` mapped (they block cold eviction)."""
    real = os.path.realpath(path)
    pids = []
    try:
        me = os.getpid()
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == me:
                continue
            try:
                with open(f"/proc/{pid}/maps") as f:
                    if real in f.read():
                        pids.append(int(pid))
            except OSError:
                continue
    except OSError:
        pass
    return pids


# -- micro mode (no GPU) ------------------------------------------------------
def _slab_spans(index, budget_bytes: int):
    """Whole expert-bank tensors, in file order, until ``budget_bytes``."""
    spans = []
    total = 0
    for name, info in index.items():
        if not name.endswith("_exps.weight") or len(info.ne) != 3:
            continue
        spans.append((name, info.path, info.offset, info.nbytes))
        total += info.nbytes
        if total >= budget_bytes:
            break
    return spans, total


def _read_spans(spans, label: str) -> tuple[float, float, int]:
    """Read every span through a fresh memmap (the packed path's access
    pattern); returns (seconds, read MiB from /proc/self/io, checksum)."""
    from freetoken.models.gguf.madvise import proc_counters

    before = proc_counters()
    t0 = time.perf_counter()
    ck = 0
    opened: dict[str, "object"] = {}
    for _name, path, off, nbytes in spans:
        mm = opened.get(path)
        if mm is None:
            mm = opened[path] = __import__("numpy").memmap(path, mode="r")
        view = mm[off : off + nbytes]
        for i in range(0, nbytes, 4096):
            ck ^= int(view[i])
        del view
    dt = time.perf_counter() - t0
    delta = proc_counters().delta(before)
    print(f"  [{label}] {dt:.2f}s  read_bytes +{delta.mib_read:.0f}MiB  "
          f"minflt +{delta.minflt}  {_machine()}", flush=True)
    return dt, delta.mib_read, ck


def micro(model_path: str, fill_gb: float) -> int:
    import numpy as np

    from freetoken.models.gguf import madvise
    from freetoken.models.gguf import reader as reader_mod

    print(f"== micro: {model_path} ({os.path.getsize(model_path) / 2**30:.1f} GiB file) ==")
    index = reader_mod.gguf_tensor_index(model_path)
    bank_bytes = sum(i.nbytes for n, i in index.items() if n.endswith("_exps.weight"))
    print(f"expert bank: {bank_bytes / 2**30:.2f} GiB in {len([n for n in index if n.endswith('_exps.weight')])} tensors")

    # M1 - advice overhead on a hot 512 KiB expert range, knob forced on
    os.environ[madvise.ENV_KNOB] = str(madvise.MODE_DONTNEED)
    name, path, off, _n = next(iter((n, i.path, i.offset, i.nbytes)
                                    for n, i in index.items() if n.endswith("_exps.weight")))
    view = np.memmap(path, mode="r")[off : off + (512 << 10)]
    for i in range(0, 512 << 10, 4096):
        _ = view[i]  # warm, so the advise measures the syscall, not a fault
    madvise.reset_advice_stats()
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        madvise.advise_willneed(path, off, 512 << 10)
    willneed_us = (time.perf_counter() - t0) * 1e6 / n
    t0 = time.perf_counter()
    for _ in range(n):
        madvise.advise_dontneed(path, off, 512 << 10)
    dontneed_us = (time.perf_counter() - t0) * 1e6 / n
    print(f"M1 advice overhead: willneed {willneed_us:.2f} us/call, "
          f"dontneed(zap+fadvise) {dontneed_us:.2f} us/call  stats={madvise.advice_stats()}")

    # M2 - DONTNEED actually frees pages through the live reader mapping
    probe = 512 << 20
    spans, _ = _slab_spans(index, probe)
    got = sum(s[3] for s in spans)
    print(f"M2 DONTNEED effectiveness over {got:.0f} B ({got / 2**20:.0f} MiB, {len(spans)} bank tensors)")
    _read_spans(spans, "warm fill")
    _ck0, _m0, ck_a = _read_spans(spans, "re-read (warm, control)")
    for _n, p, o, ln in spans:
        madvise.advise_dontneed(p, o, ln)   # knob is mode 2
    dt, mib, ck_b = _read_spans(spans, "after DONTNEED (must re-read from disk)")
    print(f"M2 verdict: {'EVICTED' if mib > got / 2**20 * 0.8 else 'NOT evicted'} "
          f"(read {mib:.0f} of {got / 2**20:.0f} MiB)  intact={ck_a == ck_b}")

    # M3 - the arena cold-start fill (~2.7 GB): cold vs WILLNEED vs warm
    fill = int(fill_gb * 2**30)
    spans, total = _slab_spans(index, fill)
    print(f"M3 cold-start fill proxy: {total / 2**30:.2f} GiB over {len(spans)} bank tensors")
    for _n, p, o, ln in spans:
        madvise.advise_dontneed(p, o, ln)
    cold_s, cold_mib, ck1 = _read_spans(spans, "fill COLD (demand fault)")
    for _n, p, o, ln in spans:
        madvise.advise_dontneed(p, o, ln)
    os.environ[madvise.ENV_KNOB] = str(madvise.MODE_WILLNEED)
    t0 = time.perf_counter()
    for _n, p, o, ln in spans:
        madvise.advise_willneed(p, o, ln)
    advise_s = time.perf_counter() - t0
    warm_mib_s, warm_mib, ck2 = _read_spans(spans, "fill with WILLNEED issued first")
    print(f"M3: cold {cold_s:.2f}s ({cold_s * 1000 / len(spans):.1f} ms/tensor), "
          f"willneed+fill {warm_mib_s:.2f}s (advise cost {advise_s:.2f}s), "
          f"intact={ck1 == ck2}")
    os.environ[madvise.ENV_KNOB] = str(madvise.MODE_DONTNEED)
    warm_s, warm_mib2, ck3 = _read_spans(spans, "fill WARM (no advise)")
    print(f"M3: warm {warm_s:.2f}s  intact={ck1 == ck3}   -> WILLNEED "
          f"{'helped' if warm_mib_s < cold_s * 0.9 else 'did NOT prefetch on this box'}")

    # M4 - the failure mode: whole-file WILLNEED (never do this in the engine)
    whole = [(n, i.path, i.offset, i.nbytes) for n, i in index.items()]
    whole_total = sum(s[3] for s in whole)
    for _n, p, o, ln in whole:
        madvise.advise_dontneed(p, o, ln)
    from freetoken.models.gguf.madvise import proc_counters
    io0 = proc_counters()
    os.environ[madvise.ENV_KNOB] = str(madvise.MODE_WILLNEED)
    t0 = time.perf_counter()
    for _n, p, o, ln in whole:
        madvise.advise_willneed(p, o, ln)
    adv_all = time.perf_counter() - t0
    time.sleep(2.0)
    read_during = proc_counters().delta(io0).mib_read
    print(f"M4 whole-file WILLNEED ({whole_total / 2**30:.1f} GiB): advise cost {adv_all:.2f}s, "
          f"read during 2s sleep +{read_during:.0f}MiB -> "
          f"{'prefetched (unexpected on this box)' if read_during > 100 else 'no async readahead (as measured)'}")
    print("M4 note: a whole-file WILLNEED/DONTNEED would evict or thrash the hot set; "
          "only per-slab ranges the engine picked are safe.")
    del view
    return 0


# -- decode mode (GPU) --------------------------------------------------------
def _worker(model_path: str, tokens: int, mode: int, advice: bool, label: str, cold: bool) -> int:
    """One fresh process per run: optional cold evict, load, prefill, decode."""
    import numpy as np

    from freetoken.models.gguf import madvise

    os.environ[madvise.ENV_KNOB] = str(mode)
    if cold:
        if not _others_map(model_path):
            # cold-cache arm: evict BEFORE anything in this process maps the file
            madvise.drop_file_pages(model_path)
            print(f"[{label}] cold-start: file pages evicted "
                  f"(cumulative read_bytes {madvise.proc_counters().mib_read:.0f}MiB)", flush=True)
        else:
            print(f"[{label}] WARNING: another process holds {model_path} mapped; "
                  f"cold eviction may be incomplete", flush=True)

    if advice:
        _install_advice_seams()

    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

    import torch
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    model = ResidentGGUF(model_path, None, max_len=2048)
    load_s = time.perf_counter() - t0
    for attach in _SEAMS:
        attach(model)

    from freetoken.models.gguf.madvise import proc_counters
    t0 = time.perf_counter()
    logits = model.prefill_chunked(PROMPT, chunk_size=64)
    prefill_s = time.perf_counter() - t0
    pos = model._seq_len

    c0 = proc_counters()
    s0 = madvise.advice_stats()  # decode-phase advice cost = end - s0
    per_tok = []
    ids: list[int] = []
    for _ in range(tokens):
        token = int(logits.argmax().item())
        ids.append(token)
        t1 = time.perf_counter()
        logits = model.decode_step_batched(model.embed_row(token), pos)
        pos += 1
        model._seq_len = pos
        per_tok.append(time.perf_counter() - t1)
    delta = proc_counters().delta(c0)
    calls = dict(model.stat_calls)
    payload = {
        "label": label, "mode": mode, "advice": advice, "cold": cold,
        "tokens": tokens,
        "load_s": load_s, "prefill_s": prefill_s,
        "s_per_tok": sum(per_tok) / len(per_tok),
        "per_tok": per_tok,
        "read_mib_per_tok": delta.mib_read / tokens,
        "minflt_per_tok": delta.minflt / tokens,
        "majflt_per_tok": delta.majflt / tokens,
        "rchar_per_tok": delta.rchar / tokens,
        "cache_hit": calls.get("cache_hit", 0), "cache_miss": calls.get("cache_miss", 0),
        "token_ids": ids,
        "advice_stats": madvise.advice_stats(),
        "advice_stats_after_prefill": s0,
        "rss_max_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "machine": _machine(),
    }
    print("BENCH_JSON " + json.dumps(payload), flush=True)
    return 0


def _install_advice_seams() -> None:
    """The two call sites the engine lane must land (documented in
    docs/vulkan/page-cache.md): fetch before the slab view is created, evict
    in the LRU victim loop. Installed here only so the bench can measure them.
    """
    from freetoken.models.gguf import madvise
    from freetoken.models.gguf import packed as packed_mod
    from freetoken.models.gguf.reader import gguf_tensor_index

    index = gguf_tensor_index(os.environ["FREETOKEN_TEST_MODEL_GGUF"])
    orig = packed_mod.packed_expert

    def fetch_seam(idx, tensor, expert, *, model_path=None):
        if madvise.madvise_mode() >= madvise.MODE_WILLNEED:
            ran = madvise.expert_slab_range(idx if idx is not None else index, tensor, expert)
            if ran is not None:
                madvise.advise_willneed(*ran)
        return orig(idx, tensor, expert, model_path=model_path)

    packed_mod.packed_expert = fetch_seam
    # gguf_provider binds packed_expert by import (the down-slab path goes
    # through it); packed_gate_up_expert needs no patch - it calls the
    # module-global packed_expert above.
    try:
        from freetoken.models.qwen3_5_moe.vulkan import gguf_provider as gprov
        gprov.packed_expert = fetch_seam
    except Exception as exc:  # engine not importable: decode mode will fail loudly anyway
        print(f"[advice] note: could not patch gguf_provider.packed_expert: {exc}", flush=True)

    def attach(model) -> None:
        id2bank = {
            id(b): b for b in getattr(model, "expert_mats", {}).values()
            if hasattr(b, "layer") and hasattr(b, "proj")
        }
        orig_put = model._cache_put

        _warned = []

        def evict_seam(key, entry):
            # gate FIRST: _cache_put is hot and even a set() copy of the key
            # space per put costs ~40 ms/tok (measured via the seam0 arm)
            if madvise.madvise_mode() != madvise.MODE_DONTNEED:
                return orig_put(key, entry)
            c = model._exp_cache
            had = key in c
            n0 = len(c)
            # LRU victims leave from the front; a 4-key prefix is O(1) and
            # covers any realistic eviction burst per put
            front = list(itertools.islice(c, 0, 4))
            orig_put(key, entry)
            k = (n0 - len(c)) if had else (n0 + 1 - len(c))
            victims = front[:max(k, 0)] if k > 0 else []
            if victims and any(v in c for v in victims):
                # the cache does not pop from the front (or bulk-evicts past
                # our prefix): advising the wrong ranges would poison the
                # measurement, so skip and say so once
                if not _warned:
                    _warned.append(1)
                    print("[evict-seam] WARNING: front-prefix victim assumption "
                          f"failed (k={k}); skipping advice for this path", flush=True)
                return
            for victim in victims:
                expert = victim[-1]
                for part in victim[:-1]:
                    bank = id2bank.get(part)
                    if bank is None:
                        continue
                    for nm in _proj_tensors(index, bank.layer, bank.proj):
                        ran = madvise.expert_slab_range(index, nm, expert)
                        if ran is not None:
                            madvise.advise_dontneed(*ran)

        model._cache_put = evict_seam  # type: ignore[method-assign]

    _SEAMS.append(attach)


_SEAMS: list = []


def _proj_tensors(index, layer: int, proj: str) -> list[str]:
    if proj == "down":
        return [f"blk.{layer}.ffn_down_exps.weight"]
    if "gate_up_exps" in "".join(index):
        return [f"blk.{layer}.ffn_gate_up_exps.weight"]
    return [f"blk.{layer}.ffn_gate_exps.weight", f"blk.{layer}.ffn_up_exps.weight"]


def _run_worker(model_path: str, tokens: int, mode: int, advice: bool, label: str,
                cold: bool) -> dict:
    from freetoken.models.gguf.madvise import ENV_KNOB

    env = dict(os.environ)
    env[ENV_KNOB] = str(mode)
    env["PYTHONPATH"] = "python" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
           "--worker", "--model", model_path, "--tokens", str(tokens), "--label", label]
    if advice:
        cmd.append("--advice")
    if cold:
        cmd.append("--cold")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    wall = time.perf_counter() - t0
    line = next((l for l in proc.stdout.splitlines() if l.startswith("BENCH_JSON ")), None)
    if line is None:
        print(f"[{label}] FAILED rc={proc.returncode} wall={wall:.0f}s\n"
              f"stdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}", flush=True)
        raise SystemExit(f"worker {label} failed")
    return json.loads(line[len("BENCH_JSON "):])


def decode(model_path: str, tokens: int, runs: int, arm_names: list[str]) -> int:
    # the same budgets the identity gate and the design lane use
    os.environ.setdefault("FTV_EXP_CACHE_MB", "2048")
    os.environ.setdefault("FTV_HOST_PACK_MB", "128")
    others = _others_map(model_path)
    arm_spec = {
        # name: (mode, install_seams).  seam0 = seams installed but the knob
        # gates every advise call - isolates the Python seam overhead itself.
        "off": (0, False), "seam0": (0, True), "adv1": (1, True), "adv2": (2, True),
    }
    for name in arm_names:
        if name not in arm_spec:
            raise SystemExit(f"--arms entries must be {sorted(arm_spec)}, got {name!r}")
    # each arm runs cold first (evicts for every arm now, not just advice
    # arms) then warm - cold/warm is independent of the advice level
    print(f"== decode: {tokens} tok x {runs} runs x {len(arm_names)} arms x cold/warm == "
          f"{'WARNING: other PIDs map the file: ' + str(others) if others else 'file exclusively ours'}")
    print(_machine())
    results: dict[str, list[dict]] = {}
    for name in arm_names:
        results[f"{name}-cold"] = []
        results[f"{name}-warm"] = []
    # cold block first, then warm; arms INTERLEAVED within each block so a
    # machine-load drift hits every arm equally
    for cold in (True, False):
        for r in range(runs):
            for name in arm_names:
                mode, advice = arm_spec[name]
                key = f"{name}-{'cold' if cold else 'warm'}"
                run_label = f"{key}-r{r}"
                res = _run_worker(model_path, tokens, mode, advice, run_label, cold)
                results[key].append(res)
                print(f"[{run_label}] {res['s_per_tok'] * 1000:.1f} ms/tok "
                      f"read={res['read_mib_per_tok']:.1f}MiB/tok minflt/tok={res['minflt_per_tok']:.0f} "
                      f"hit/miss={res['cache_hit']}/{res['cache_miss']} load={res['load_s']:.0f}s "
                      f"{res['machine']}", flush=True)
    print("\n== summary (s/tok, min/median/max over runs) ==")
    for key in results:
        times = sorted(r["s_per_tok"] for r in results[key])
        reads = statistics.median(r["read_mib_per_tok"] for r in results[key])
        flts = statistics.median(r["minflt_per_tok"] for r in results[key])
        print(f"{key:14s} {times[0] * 1000:.1f}/{statistics.median(times) * 1000:.1f}/"
              f"{times[-1] * 1000:.1f} ms/tok   read={reads:.1f}MiB/tok minflt/tok={flts:.0f}")
    out = pathlib.Path("/tmp/opencode/page_cache_decode.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("micro", "decode", "all"), default="micro")
    ap.add_argument("--model", default=GGUF)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--fill-gb", type=float, default=2.7,
                    help="cold-start fill proxy size (arena fill is ~2.7 GiB)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--advice", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cold", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--label", default="worker", help=argparse.SUPPRESS)
    ap.add_argument("--arms", default="off,adv1,adv2",
                    help="decode arms: comma list of off/adv1/adv2")
    args = ap.parse_args()
    if not args.model or not pathlib.Path(args.model).is_file():
        raise SystemExit("set --model or FREETOKEN_TEST_MODEL_GGUF")
    if args.worker:
        # keep the worker's stdout machine-readable; re-exec under the caller
        return _worker(args.model, args.tokens, int(os.environ.get(madvise_mode_env(), "0")),
                       args.advice, args.label, args.cold)
    if args.mode in ("micro", "all"):
        micro(args.model, args.fill_gb)
    if args.mode in ("decode", "all"):
        decode(args.model, args.tokens, args.runs,
               [a.strip() for a in args.arms.split(",") if a.strip()])
    return 0


def madvise_mode_env() -> str:
    from freetoken.models.gguf.madvise import ENV_KNOB
    return ENV_KNOB


if __name__ == "__main__":
    raise SystemExit(main())
