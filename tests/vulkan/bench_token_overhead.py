"""Host-interaction cost of ONE decode token, CPU idle vs CPU busy.

Answers the question the GPU-resident-token design has to answer first:
**how much of a decode token is spent waiting on the host, and how much of
that is the host's own CPU time?**

It changes NO engine or kernel behaviour. Every number comes from
wall-clock wrappers installed around the already-exported ``ext``
functions and around the decoder's Python methods (same technique as
``bench_token_breakdown.py``), plus the extension's own counters.

    FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \
        /home/cus/.venv-vllm-vulkan/bin/python tests/vulkan/bench_token_overhead.py \
        --tokens 16 --warmup 4 --runs 3

    ... --arm idle          # only the idle arm (no burner threads)
    ... --arm busy          # only the busy arm
    ... --busy-threads 7    # burner threads (default: one per core - 1)
    ... --json /tmp/out.json

Definitions used throughout (the same ones the design doc uses):

  drain        one point where the host BLOCKS on the GPU: a
               ``batch_submit`` fence wait or a synchronous
               submit-and-wait op.  This is the hard host<->GPU round trip.
  upload       one host->device data push (bytes staged for the GPU).
  download     one device->host data pull (bytes read back).
  host call    one Python-level call into the decoder (attention, GDN,
               router/MoE, rmsnorm, embed, expert fetch).
  gpu wait     wall time inside ``batch_submit`` = GPU execution + fence
               overhead + readback memcpy.
  host work    everything else in the token = record + host calls +
               python/torch glue NOT inside a wrapped function.

The idle/busy arms differ ONLY in how many CPU-burning threads run during
the timed window.  The burners are pinned to distinct CPUs and the decode
thread is pinned to its own CPU, so the busy arm measures "the same token
on a box whose cores are already busy" without stealing the decode
thread's core outright.

Also collected (free, from the same run): the per-layer expert routing
distribution, which sizes the fixed-slot expert arena the design needs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import sys
import threading
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Reuse the attribution wrappers + env snapshot of the sibling benchmark so
# the two scripts measure the same thing the same way.
from bench_token_breakdown import (  # noqa: E402
    PROMPT,
    SLICER,
    _delta,
    _env_line,
    _loadavg,
    _meminfo,
    _rss_mb,
    _unwrap_ext,
    _unwrap_model,
    _wrap_ext,
    _wrap_model,
    calibrate,
    snapshot,
)


# --------------------------------------------------------------------------
# CPU burner (GIL-free: hashlib releases the GIL for buffers > 2047 B)
# --------------------------------------------------------------------------
class CpuBurner:
    """Spin N threads on pinned CPUs to emulate a contended machine."""

    def __init__(self, nthreads: int, cpus: list[int]) -> None:
        self.nthreads = max(0, nthreads)
        self.cpus = list(cpus)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._err: list[BaseException] = []

    def _loop(self, data: bytes, cpu: int | None) -> None:
        try:
            if cpu is not None:
                os.sched_setaffinity(0, {cpu})
            while not self._stop.is_set():
                hashlib.sha256(data).digest()
        except BaseException as exc:  # noqa: BLE001 - reported, not raised
            self._err.append(exc)

    def __enter__(self) -> "CpuBurner":
        if self.nthreads == 0:
            return self
        data = b"x" * (1 << 20)
        for i in range(self.nthreads):
            cpu = self.cpus[i] if i < len(self.cpus) else None
            t = threading.Thread(target=self._loop, args=(data, cpu),
                                 daemon=True, name=f"burn-{i}")
            t.start()
            self._threads.append(t)
        time.sleep(0.2)  # let the burners reach the hot loop
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()


# --------------------------------------------------------------------------
# routing capture (sizes the fixed-slot expert arena)
# --------------------------------------------------------------------------
class RoutingCapture:
    """Record the host router's top-k expert ids, per layer, per token.

    The host-router path calls ``torch.topk(probs, k, dim=-1)`` once per
    layer with a [1, E] probability vector.  Wrapping ``torch.topk``
    captures the routed sets without touching the engine; the captured
    order is the layer order of the decode walk.
    """

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self.layers: list[list[int]] = []   # one entry per router call
        self._orig = torch.topk

    def _wrapped(self, *args, **kwargs):
        out = self._orig(*args, **kwargs)
        try:
            idx = out[1]
            src = args[0] if args else kwargs.get("input")
            # The router calls topk on a [.., num_experts] probability
            # vector; the OUTPUT's last dim is the top-k width, so the
            # input is what identifies a router call.
            if (src is not None and getattr(src, "ndim", 0) >= 1
                    and int(src.shape[-1]) == self.num_experts
                    and idx.numel()):
                self.layers.append([int(v) for v in idx.reshape(-1).tolist()])
        except (TypeError, ValueError, RuntimeError):
            pass
        return out

    def __enter__(self) -> "RoutingCapture":
        torch.topk = self._wrapped  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        torch.topk = self._orig  # type: ignore[assignment]


class _NullCapture:
    """No-op stand-in so the run loop has one code path."""

    layers: list = []

    def __enter__(self) -> "_NullCapture":
        return self

    def __exit__(self, *exc) -> None:
        return None


def arena_sizing(route: list[list[int]], num_layers: int, num_experts: int,
                 top_k: int) -> dict:
    """Arena sizing for the routed-expert working set.

    `route` is the router call log in CALL ORDER: one entry per
    (token, layer) router call, each holding that call's `top_k`
    expert ids.  The decode loop is layer-ordered per token, so call
    index i belongs to layer ``i % num_layers``.

    Three questions are answered:

    1. How wide is the working set?  (distinct experts touched per layer)
    2. What does a plain S-slot LRU arena miss?  (the FIFO sweep)
    3. What would a PREDICTIVE arena (prewarm the union of the last T
       tokens' routed sets) miss?  This is the regime the NVFP4 arena
       actually runs in, and it is the only one that can beat the
       working set of question 1.
    """
    per_layer: list[list[int]] = [[] for _ in range(num_layers)]
    for call_i, call in enumerate(route):
        per_layer[call_i % num_layers].extend(call[:top_k])

    out: dict = {"per_layer_distinct": [], "sweep": {}, "predictive": {},
                  "overlap": {}, "requests": 0}
    total_req = 0
    total_miss = {s: 0 for s in (8, 12, 16, 24, 32, 48, 64, 96, 128)}
    # Predictive arena: slots hold the union of the last T tokens.
    predict_miss = {t: 0 for t in (1, 2, 3, 4, 6)}
    overlap_sum = 0.0
    overlap_n = 0

    for li, seq in enumerate(per_layer):
        if not seq:
            continue
        out["per_layer_distinct"].append(len(set(seq)))
        total_req += len(seq)

        # (2) plain LRU arena with S slots
        for s in total_miss:
            ring: list[int] = []
            miss = 0
            for e in seq:
                if e in ring:
                    ring.remove(e)
                    ring.append(e)
                else:
                    miss += 1
                    ring.append(e)
                    if len(ring) > s:
                        ring.pop(0)
            total_miss[s] += miss

        # (3) predictive arena + (4) temporal overlap.
        # Per token t of this layer, `seq[t*top_k:(t+1)*top_k]` is the
        # routed set; the arena holds the union of the previous T sets.
        n_tok = len(seq) // top_k
        for t in range(n_tok):
            cur = set(seq[t * top_k:(t + 1) * top_k])
            prev_sets = [set(seq[(t - k) * top_k:(t - k + 1) * top_k])
                         for k in range(1, min(t, 6) + 1)]
            if prev_sets:
                inter = len(cur & prev_sets[0])
                overlap_sum += inter
                overlap_n += 1
            for T in predict_miss:
                union: set[int] = set()
                for ps in prev_sets[:T]:
                    union |= ps
                predict_miss[T] += len(cur - union)

    out["requests"] = total_req
    out["sweep"] = {
        str(s): {"misses": m,
                 "miss_rate": (m / total_req) if total_req else 0.0}
        for s, m in total_miss.items()
    }
    out["predictive"] = {
        str(t): {"slots_per_layer": t * top_k,
                 "misses": m,
                 "miss_rate": (m / total_req) if total_req else 0.0}
        for t, m in predict_miss.items()
    }
    out["overlap"] = {
        "mean_overlap_prev_token": (overlap_sum / overlap_n) if overlap_n else 0.0,
        "top_k": top_k,
    }
    return out


# --------------------------------------------------------------------------
# one measured run
# --------------------------------------------------------------------------
def run_once(model, ext, args, arm: str, batched: bool,
             prompt: list[int], capture: RoutingCapture | None) -> dict:
    model.reset()
    t0 = time.perf_counter()
    logits = model.prefill_chunked(prompt, chunk_size=64)
    prefill_s = time.perf_counter() - t0
    pos = model._seq_len

    step = model.decode_step_batched if batched else model.decode_step
    for _ in range(args.warmup):
        token = int(logits.argmax().item())
        logits = step(model.embed_row(token), pos)
        pos += 1
        model._seq_len = pos

    model.stats.clear()
    model.stat_calls.clear()
    SLICER.reset()
    if capture is not None:
        capture.layers.clear()
    before = snapshot(ext, model)

    per_token: list[float] = []
    ids: list[int] = []
    t_start = time.perf_counter()
    for _ in range(args.tokens):
        token = int(logits.argmax().item())
        ids.append(token)
        t0 = time.perf_counter()
        logits = step(model.embed_row(token), pos)
        per_token.append(time.perf_counter() - t0)
        pos += 1
        model._seq_len = pos
    total_s = time.perf_counter() - t_start
    after = snapshot(ext, model)
    fence_samples = list(SLICER.samples.get("fence", []))

    n = args.tokens
    buckets = {k: v / n for k, v in SLICER.buckets.items()}
    calls = {k: v / n for k, v in SLICER.calls.items()}
    total_ms = total_s / n * 1e3
    measured_ms = sum(buckets.values()) * 1e3
    gpu_wait_ms = buckets.get("fence", 0.0) * 1e3 + buckets.get("sync", 0.0) * 1e3
    host_work_ms = total_ms - gpu_wait_ms

    cd = {k: _delta(before, after, k) for k in after
          if k not in ("resident_mib", "stat_calls")}
    submits = cd.get("batch_submit_count", (0, 0))[0] / n if cd.get(
        "batch_submit_count") else 0.0
    barriers = cd.get("batch_barrier_stats", (0, 0))[0] / n if cd.get(
        "batch_barrier_stats") else 0.0
    q2 = cd.get("batch_upload_async_stats", (0, 0))[1] / n if cd.get(
        "batch_upload_async_stats") else 0.0
    cross = cd.get("batch_upload_async_stats", (0, 0))[0] / n if cd.get(
        "batch_upload_async_stats") else 0.0

    cal = args.calib
    empty_ms = cal.get("empty_submit_ms", 0.0)
    drains = calls.get("fence", 0.0) + calls.get("sync", 0.0)
    # Per-drain fixed cost = the empty submit; the rest of the fence time
    # is GPU execution (kernels + DMA) + the readback memcpy.
    drain_overhead_ms = drains * empty_ms
    host_calls = sum(calls.get(k, 0.0) for k in
                      ("host_attn", "host_gdn", "host_moe", "host_rms",
                       "host_embed", "host_fetch", "host_glue"))

    # Per-drain fence-wait profile: is the fence bucket LATENCY (many
    # small waits = one per drain, the cost of 91 pipeline drains) or
    # KERNEL time (a few big waits)?  The two need opposite fixes.
    fs = sorted(fence_samples)
    if fs:
        p = lambda q: fs[min(len(fs) - 1, int(q * len(fs)))]  # noqa: E731
        drain_profile = {
            "n": len(fs),
            "median_ms": statistics.median(fs),
            "p90_ms": p(0.90),
            "p99_ms": p(0.99),
            "max_ms": fs[-1],
            "sum_ms": sum(fs),
            "under_0p1ms": sum(1 for x in fs if x < 0.1),
            "under_0p5ms": sum(1 for x in fs if x < 0.5),
            "over_1ms": sum(1 for x in fs if x >= 1.0),
        }
    else:
        drain_profile = {}

    return {
        "arm": arm,
        "tokens": n,
        "prefill_s": prefill_s,
        "s_per_tok": total_s / n,
        "tok_per_s": n / total_s,
        "per_token_ms_median": statistics.median(x * 1e3 for x in per_token),
        "per_token_ms": [x * 1e3 for x in per_token],
        "total_ms": total_ms,
        "gpu_wait_ms": gpu_wait_ms,
        "host_work_ms": host_work_ms,
        "host_work_pct": 100.0 * host_work_ms / total_ms if total_ms else 0.0,
        "buckets_ms": {k: v * 1e3 for k, v in buckets.items()},
        "calls_per_tok": calls,
        "interactions_per_tok": {
            "drains": drains,
            "submits": submits,
            "sync_roundtrips": calls.get("sync", 0.0),
            # Every batch_* op except batch_submit lands in the "record"
            # bucket; the extension does not break it down per op class,
            # so the honest number is the record-call count.
            "record_calls": calls.get("record", 0.0),
            "host_calls": host_calls,
            "barriers": barriers,
            "queue2_submits": q2,
            "cross_queue_barriers": cross,
        },
        "bytes_up_mib": SLICER.bytes_up / n / 2**20,
        "bytes_down_mib": SLICER.bytes_down / n / 2**20,
        "drain_overhead_ms": drain_overhead_ms,
        "drain_overhead_pct": (100.0 * drain_overhead_ms / total_ms
                               if total_ms else 0.0),
        "drain_profile": drain_profile,
        "cache_hit": after["stat_calls"].get("cache_hit", 0) / n,
        "cache_miss": after["stat_calls"].get("cache_miss", 0) / n,
        "resident_mib": after["resident_mib"],
        "counters_delta": cd,
        "ids": ids,
        "env": _env_line(),
        "loadavg": _loadavg(),
        "rss_mb": _rss_mb(),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _fmt_counter_map(d: dict) -> str:
    parts = []
    for k in sorted(d):
        v = d[k]
        parts.append(f"{k}={v}")
    return " ".join(parts)


def print_arm(runs: list[dict], arm: str) -> dict:
    med = statistics.median(r["s_per_tok"] for r in runs)
    pick = min(runs, key=lambda r: abs(r["s_per_tok"] - med))
    total = pick["total_ms"]
    print(f"\n=== [{arm}] {len(runs)} runs, s/tok "
          f"{', '.join(f'{x:.3f}' for x in sorted(r['s_per_tok'] for r in runs))} "
          f"(median {med:.3f}) ===")
    print(f"    tok/s median {1 / med:.2f}  per-token ms "
          f"{pick['per_token_ms_median']:.1f} | {pick['env']}")
    print(f"    gpu_wait {pick['gpu_wait_ms']:.2f} ms/tok "
          f"({100 * pick['gpu_wait_ms'] / total:.1f}%)  "
          f"host_work {pick['host_work_ms']:.2f} ms/tok "
          f"({pick['host_work_pct']:.1f}%)")
    inter = pick["interactions_per_tok"]
    print(f"    interactions/tok: drains={inter['drains']:.1f} "
          f"(submits={inter['submits']:.1f} sync={inter['sync_roundtrips']:.1f}) "
          f"host_calls={inter['host_calls']:.1f} "
          f"barriers={inter['barriers']:.1f} "
          f"q2_submits={inter['queue2_submits']:.1f}")
    print(f"    bytes/tok: up {pick['bytes_up_mib']:.1f} MiB  "
          f"down {pick['bytes_down_mib']:.1f} MiB  "
          f"expert hit={pick['cache_hit']:.1f} miss={pick['cache_miss']:.1f}")
    print(f"    drain fixed overhead {pick['drain_overhead_ms']:.2f} ms/tok "
          f"({pick['drain_overhead_pct']:.1f}% of the token) "
          f"[empty_submit={args_calib_ms:.3f} ms]")
    dp = pick.get("drain_profile") or {}
    if dp:
        print(f"    fence-wait profile (n={dp['n']}): median "
              f"{dp['median_ms']:.3f} ms, p90 {dp['p90_ms']:.3f} ms, "
              f"p99 {dp['p99_ms']:.3f} ms, max {dp['max_ms']:.3f} ms; "
              f"{dp['under_0p1ms']} waits < 0.1 ms, "
              f"{dp['under_0p5ms']} < 0.5 ms, {dp['over_1ms']} >= 1 ms")
    for key, ms in sorted(pick["buckets_ms"].items(), key=lambda kv: -kv[1]):
        print(f"      {key:<16}{ms:>8.2f} ms/tok  "
              f"{pick['calls_per_tok'].get(key, 0):>7.1f} calls/tok")
    return pick


args_calib_ms = 0.0


def main() -> int:
    global args_calib_ms
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FREETOKEN_TEST_MODEL_GGUF", ""))
    ap.add_argument("--tokens", type=int, default=16,
                    help="timed decode tokens per run")
    ap.add_argument("--warmup", type=int, default=4,
                    help="untimed decode tokens (expert-LRU warmup)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--ctx", type=int, default=0,
                    help="prefill at least this many tokens (0 = PROMPT)")
    ap.add_argument("--arm", choices=("idle", "busy", "both"), default="both")
    ap.add_argument("--busy-threads", type=int, default=-1,
                    help="CPU burner threads (-1 = one per core - 1)")
    ap.add_argument("--sequential", action="store_true",
                    help="FTV_GGUF_BATCHED=0 (sequential decoder arm)")
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--no-routing", action="store_true",
                    help="skip the per-layer routing capture")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.model or not pathlib.Path(args.model).is_file():
        raise SystemExit("set --model or FREETOKEN_TEST_MODEL_GGUF to the GGUF file")

    if args.sequential:
        os.environ["FTV_GGUF_BATCHED"] = "0"
    batched = os.environ.get("FTV_GGUF_BATCHED", "1").strip() != "0"
    label = args.label or ("batched" if batched else "sequential")

    prompt = list(PROMPT)
    while len(prompt) < args.ctx:
        prompt += PROMPT
    prompt = prompt[: max(args.ctx, len(PROMPT))]

    from freetoken.kernel import vulkan as vk
    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

    torch.set_num_threads(8)
    print(f"[{label}] start {time.strftime('%H:%M:%S')} batched={batched} "
          f"prompt={len(prompt)} tok tokens={args.tokens} "
          f"warmup={args.warmup} runs={args.runs} env={_env_line()}",
          flush=True)

    ext = vk.load_ext()
    t0 = time.perf_counter()
    model = ResidentGGUF(args.model, None, max_len=args.max_len)
    print(f"[{label}] load {time.perf_counter() - t0:.1f}s {_env_line()}",
          flush=True)

    # CPU topology for the busy arm: the decode thread keeps CPU 0, the
    # burners take the other cores.
    avail = sorted(os.sched_getaffinity(0))
    main_cpu = avail[0]
    rest = avail[1:]
    n_burn = args.busy_threads
    if n_burn < 0:
        n_burn = max(0, (len(avail) // 2) - 1)  # one per physical core - 1
    burn_cpus = rest[:n_burn]
    print(f"[{label}] busy arm: {len(burn_cpus)} burner threads on CPUs "
          f"{burn_cpus}, decode thread pinned to CPU {main_cpu}", flush=True)

    saved_ext = _wrap_ext(ext)
    try:
        if args.no_calib:
            args.calib = {"empty_submit_ms": 0.0, "batch_h2d_gbs": 0.0,
                          "batch_d2h_gbs": 0.0, "sync_h2d_gbs": 0.0,
                          "sync_d2h_gbs": 0.0}
        else:
            args.calib = calibrate(ext)
            print(f"[{label}] calib {json.dumps(args.calib)}", flush=True)
        args_calib_ms = args.calib.get("empty_submit_ms", 0.0)

        cfg = model.config
        capture = None
        if not args.no_routing:
            capture = RoutingCapture(cfg.num_experts)

        saved_model = _wrap_model(model)
        SLICER.reset()
        results: dict[str, list[dict]] = {"idle": [], "busy": []}
        route: list[list[int]] = []
        try:
            os.sched_setaffinity(0, {main_cpu})
            cap = capture if capture is not None else _NullCapture()
            with cap:
                for i in range(args.runs):
                    for arm in (("idle", "busy") if args.arm == "both"
                                else (args.arm,)):
                        if arm == "busy":
                            with CpuBurner(len(burn_cpus), burn_cpus):
                                r = run_once(model, ext, args, arm, batched,
                                             prompt, capture)
                        else:
                            r = run_once(model, ext, args, arm, batched,
                                         prompt, capture)
                        results[arm].append(r)
                        if capture is not None:
                            route.extend(capture.layers)
                        print(f"[{label}] {arm} run {i + 1}/{args.runs}: "
                              f"{r['s_per_tok']:.3f}s/tok "
                              f"({r['tok_per_s']:.2f} tok/s) "
                              f"drains={r['interactions_per_tok']['drains']:.0f} "
                              f"gpu_wait={r['gpu_wait_ms']:.1f}ms "
                              f"host_work={r['host_work_ms']:.1f}ms "
                              f"{r['env']}", flush=True)
        finally:
            _unwrap_model(model, saved_model)
    finally:
        _unwrap_ext(ext, saved_ext)

    picks = {}
    for arm, runs in results.items():
        if runs:
            picks[arm] = print_arm(runs, arm)
    if "idle" in picks and "busy" in picks:
        d = picks["busy"]["total_ms"] / picks["idle"]["total_ms"]
        print(f"\n=== busy/idle slowdown: {d:.2f}x "
              f"({picks['idle']['total_ms']:.1f} -> "
              f"{picks['busy']['total_ms']:.1f} ms/tok) ===")

    sizing = None
    if capture is not None and route:
        sizing = arena_sizing(route, len(model.config.layer_types),
                              model.config.num_experts,
                              model.config.num_experts_per_tok)
        print(f"\n=== routing (arena sizing) ===")
        print(f"    router calls captured: {len(route)}  "
              f"distinct experts/layer (median): "
              f"{statistics.median(sizing['per_layer_distinct']) if sizing['per_layer_distinct'] else 0}")
        for s, d in sorted(sizing["sweep"].items(), key=lambda kv: int(kv[0])):
            print(f"    arena slots/layer {s:>4}: miss {d['misses']:>6} "
                  f"({100 * d['miss_rate']:.1f}%)")
        for t, d in sorted(sizing["predictive"].items(),
                           key=lambda kv: int(kv[0])):
            print(f"    predictive arena (union of last {t} tokens, "
                  f"{d['slots_per_layer']:>3} slots/layer): miss "
                  f"{d['misses']:>6} ({100 * d['miss_rate']:.1f}%)")
        print(f"    mean overlap with the previous token's routed set: "
              f"{sizing['overlap']['mean_overlap_prev_token']:.2f} / "
              f"{sizing['overlap']['top_k']}")

    payload = {
        "label": label,
        "batched": batched,
        "args": vars(args) | {"calib": args.calib},
        "results": results,
        "picks": picks,
        "arena_sizing": sizing,
        "route": route,
        "env_end": _env_line(),
    }
    # The default lands outside the repo: this bench must not add files to
    # the tree it measures.
    out = args.json or str(
        pathlib.Path("/tmp/opencode") / f"token_overhead_{label}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[{label}] wrote {out} | end {_env_line()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
