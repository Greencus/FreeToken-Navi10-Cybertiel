"""Per-token cost attribution for the CyberTiel GGUF Vulkan decoder.

Not collected by pytest (no ``test_`` prefix): run it directly to split ONE
decode token into (a) expert weight transfer, (b) GPU kernel/dispatch inside
fence waits, (c) host<->GPU round trips + fence waits, (d) host-side Python
math (router, attention, GDN recurrence) and (e) everything else.

It changes NO engine or kernel behaviour: every number comes from wall-clock
wrappers installed around the already-exported ``ext`` functions and around
the decoder's Python methods (the extension exposes ``batch_submit`` etc. as
module attributes, so plain module monkey-patching is enough).

    FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \
        /home/cus/.venv-vllm-vulkan/bin/python tests/vulkan/bench_token_breakdown.py \
        --tokens 16 --warmup 4 --runs 3

    ... --sequential            # FTV_GGUF_BATCHED=0 arm (sequential decoder)
    ... --json /tmp/out.json    # machine-readable dump (default: alongside)

Attribution method (exclusive wall-clock slices, no double counting):

  ext wrappers (leaves):
    fence     batch_submit            - queue submit + vkWaitForFences +
                                        readback memcpys = ALL GPU work of
                                        that batch (kernels + DMA + fence)
    sync      upload/update/download_*
              linear_*_resident_io    - one submit_and_wait round trip each
                                        (the sequential path's whole cost
                                        pattern)
    record    batch_* (except submit) - host-side vkCmd* recording only
    async     batch_upload_async*     - queue-2 upload fences
    non_gpu   allocate/free/stats     - pure host bookkeeping
  model wrappers (parents; their ext children are excluded from the slice):
    host_attn     _full_attn_host / _full_attn_step  (KV append + SDPA on CPU)
    host_gdn      _gdn_host / _gdn_step              (linear-attn recurrence)
    host_moe      _record_moe_in / _moe_step          (router + topk + python)
    host_fetch    _gguf_gu_handle / _gguf_dn_handle   (pack + LRU on miss)
    host_rms      _rms
    host_embed    embed_row
    host_glue     _pinned_affine* (torch.empty etc. around the sync ops)
  remainder = token wall time - every exclusive slice (python/torch glue
  that is not inside any wrapped function).

Transfer vs kernel inside ``fence``: calibrated separately - empty-submit
cost (batch_begin+batch_submit with no work) gives the fixed per-submit
overhead, and a measured batched/sync H2D+D2H loop gives GB/s, so
  transfer_est = bytes / measured_BW
  kernel_est  = fence_total - submits*empty_submit - transfer_est
Those two are DERIVED (labelled), everything else is measured directly.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import statistics
import time

import torch

# Same prompt set as tests/vulkan/test_mtp.py and bench_gguf_packed.py (GGUF ids).
PROMPT = [14917, 557, 264, 1248, 1834, 13, 353, 14428, 2046, 13569, 321, 1801, 14799, 13]


# --------------------------------------------------------------------------
# environment snapshot (page cache / load / RSS go with EVERY number)
# --------------------------------------------------------------------------
def _meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            if key in ("Cached", "MemFree", "MemAvailable", "SwapFree"):
                out[key] = int(rest.split()[0]) // 1024  # MiB
    return out


def _loadavg() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _env_line() -> str:
    mi = _meminfo()
    return (
        f"cached={mi.get('Cached', -1)}MiB mem_avail={mi.get('MemAvailable', -1)}MiB "
        f"load={_loadavg()} rss_max={_rss_mb():.0f}MiB"
    )


# --------------------------------------------------------------------------
# exclusive-time slicer (a stack: children subtract from their parents)
# --------------------------------------------------------------------------
class Slicer:
    def __init__(self) -> None:
        self.buckets: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.samples: dict[str, list[float]] = {}
        self.bytes_up = 0
        self.bytes_down = 0
        self._stack: list[list] = []  # [category, t0, child_seconds]

    def enter(self, cat: str) -> None:
        self._stack.append([cat, time.perf_counter(), 0.0])

    def exit(self) -> None:
        t1 = time.perf_counter()
        cat, t0, child = self._stack.pop()
        dt = (t1 - t0) - child
        self.buckets[cat] = self.buckets.get(cat, 0.0) + dt
        self.calls[cat] = self.calls.get(cat, 0) + 1
        if cat in ("fence", "sync"):
            self.samples.setdefault(cat, []).append(dt * 1e3)
        if self._stack:
            self._stack[-1][2] += t1 - t0

    def reset(self) -> None:
        self.buckets = {}
        self.calls = {}
        self.samples = {}
        self.bytes_up = 0
        self.bytes_down = 0


SLICER = Slicer()


def _tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size() if obj.numel() > 0 else 0
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    return 0


# --------------------------------------------------------------------------
# ext (pybind module) wrappers
# --------------------------------------------------------------------------
# Pure host bookkeeping: never a submit, never a fence.
_NON_GPU = {
    "allocate_resident", "free_resident", "resident_bytes_total",
    "desc_cache_stats", "batch_submit_count", "batch_barrier_stats",
    "batch_upload_async_stats", "batch_deferred_stats", "device_info",
    "memory_stats", "version", "set_desc_cache_cap",
}
_STATS_SUFFIX = ("_stats", "_count", "_total")

# Categories for non-"sync"/"record" batch ops.
_SPECIAL_BATCH = {
    "batch_submit": "fence",
    "batch_upload_async": "async_up",
    "batch_upload_async_flush": "async_up",
    "batch_upload_async_drain": "async_up",
}


def _classify(name: str) -> str:
    if name in _NON_GPU or name.endswith(_STATS_SUFFIX):
        return "non_gpu"
    if name in _SPECIAL_BATCH:
        return _SPECIAL_BATCH[name]
    if name.startswith("batch_"):
        return "record"          # host vkCmd* recording only
    return "sync"                # submit_and_wait round trip


def _is_upload(name: str) -> bool:
    return "upload" in name or name.startswith("update_resident")


def _is_download(name: str) -> bool:
    return "download" in name


def _wrap_ext(ext) -> dict:
    """Monkey-patch every callable on the ext module; return originals."""
    saved: dict = {}
    for name in sorted(dir(ext)):
        if name.startswith("_"):
            continue
        fn = getattr(ext, name)
        if not callable(fn):
            continue
        cat = _classify(name)
        if cat == "non_gpu":
            continue
        saved[name] = fn

        def make(cat=cat, name=name, fn=fn):
            def wrapper(*args, **kwargs):
                SLICER.enter(cat)
                try:
                    ret = fn(*args, **kwargs)
                finally:
                    SLICER.exit()
                # Python-level ext calls never nest inside each other (the
                # C++ batch_moe_record's internal batch_download is not
                # visible here), so count bytes on every wrapper call.
                if _is_upload(name):
                    SLICER.bytes_up += sum(_tensor_bytes(a) for a in args)
                    SLICER.bytes_up += sum(
                        _tensor_bytes(v) for v in kwargs.values())
                elif _is_download(name):
                    SLICER.bytes_down += sum(_tensor_bytes(a) for a in args)
                    SLICER.bytes_down += _tensor_bytes(ret)
                return ret
            return wrapper

        setattr(ext, name, make())
    return saved


def _unwrap_ext(ext, saved: dict) -> None:
    for name, fn in saved.items():
        setattr(ext, name, fn)


# --------------------------------------------------------------------------
# decoder host-method wrappers
# --------------------------------------------------------------------------
_HOST_MAP = {
    "_full_attn_host": "host_attn",
    "_full_attn_step": "host_attn",
    "_gdn_host": "host_gdn",
    "_gdn_step": "host_gdn",
    "_record_moe_in": "host_moe",
    "_moe_step": "host_moe",
    "_gguf_gu_handle": "host_fetch",
    "_gguf_dn_handle": "host_fetch",
    "_expert_handle": "host_fetch",
    "_rms": "host_rms",
    "embed_row": "host_embed",
    "_pinned_affine": "host_glue",
    "_pinned_affine_batch": "host_glue",
    "_sync_gdn_gpu": "host_gdn_sync",
}


def _wrap_model(model) -> dict:
    saved: dict = {}
    for attr, cat in _HOST_MAP.items():
        fn = getattr(model, attr, None)
        if fn is None:
            continue
        saved[attr] = fn

        def make(cat=cat, fn=fn):
            def wrapper(*args, **kwargs):
                SLICER.enter(cat)
                try:
                    return fn(*args, **kwargs)
                finally:
                    SLICER.exit()
            return wrapper

        setattr(model, attr, make())
    return saved


def _unwrap_model(model, saved: dict) -> None:
    for attr, fn in saved.items():
        setattr(model, attr, fn)


# --------------------------------------------------------------------------
# calibration: empty-submit cost and measured H2D/D2H GB/s
# --------------------------------------------------------------------------
def calibrate(ext, mib: int = 8, reps: int = 10) -> dict:
    """Empty-submit cost and measured H2D/D2H GB/s, warm driver state.

    Runs AFTER the model is loaded (pipelines/pools already exercised) and
    discards warm-up iterations, otherwise RADV's first-submit cold path
    pollutes the empty-submit number and the derived transfer/kernel split
    goes negative.
    """
    out: dict = {}
    for _ in range(10):  # warm-up (excluded)
        ext.batch_begin()
        ext.batch_submit()
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        ext.batch_begin()
        ext.batch_submit()
    out["empty_submit_ms"] = (time.perf_counter() - t0) * 1e3 / n
    empty_s = out["empty_submit_ms"] / 1e3

    nbytes = mib * 2**20
    src = torch.rand(nbytes // 4, dtype=torch.float32)
    dst = torch.empty(nbytes // 4, dtype=torch.float32)
    h = ext.allocate_resident(nbytes)

    def _rate(dt: float) -> float:
        adj = dt - empty_s  # DMA runs inside the same submit
        if adj <= 0:
            adj = dt
        return nbytes / adj / 1e9

    # batched H2D: record + DMA inside batch_submit
    for _ in range(3):
        ext.batch_begin(), ext.batch_upload(h, src), ext.batch_submit()
    t0 = time.perf_counter()
    for _ in range(reps):
        ext.batch_begin()
        ext.batch_upload(h, src)
        ext.batch_submit()
    out["batch_h2d_ms"] = (time.perf_counter() - t0) * 1e3 / reps
    out["batch_h2d_gbs"] = _rate(out["batch_h2d_ms"] / 1e3)

    # batched D2H: record + copy inside batch_submit
    for _ in range(3):
        ext.batch_begin(), ext.batch_download(h, dst), ext.batch_submit()
    t0 = time.perf_counter()
    for _ in range(reps):
        ext.batch_begin()
        ext.batch_download(h, dst)
        ext.batch_submit()
    out["batch_d2h_ms"] = (time.perf_counter() - t0) * 1e3 / reps
    out["batch_d2h_gbs"] = _rate(out["batch_d2h_ms"] / 1e3)

    # sync path (sequential decoder): update_resident / download_resident
    for _ in range(3):
        ext.update_resident(h, src)
    t0 = time.perf_counter()
    for _ in range(reps):
        ext.update_resident(h, src)
    out["sync_h2d_gbs"] = nbytes / ((time.perf_counter() - t0) / reps) / 1e9
    for _ in range(3):
        ext.download_resident(h, [nbytes // 4])
    t0 = time.perf_counter()
    for _ in range(reps):
        ext.download_resident(h, [nbytes // 4])
    out["sync_d2h_gbs"] = nbytes / ((time.perf_counter() - t0) / reps) / 1e9

    ext.free_resident(h)
    out["calib_bytes"] = nbytes
    return out


# --------------------------------------------------------------------------
# counters
# --------------------------------------------------------------------------
def snapshot(ext, model) -> dict:
    snap = {}
    for getter in (
        "batch_submit_count", "batch_barrier_stats", "desc_cache_stats",
        "batch_upload_async_stats", "batch_deferred_stats",
    ):
        try:
            snap[getter] = tuple(getattr(ext, getter)())
        except Exception as exc:  # noqa: BLE001 - counters are best-effort
            snap[getter] = f"err:{exc}"
    snap["resident_mib"] = ext.resident_bytes_total() / 2**20
    snap["stat_calls"] = dict(model.stat_calls)
    return snap


def _delta(before: dict, after: dict, key: str):
    a, b = after.get(key), before.get(key)
    if isinstance(a, tuple) and isinstance(b, tuple) and a and isinstance(a[0], int):
        return tuple(x - y for x, y in zip(a, b))
    return None


# --------------------------------------------------------------------------
# one measured run
# --------------------------------------------------------------------------
def run_once(model, ext, args, label: str, batched: bool,
             prompt: list[int]) -> dict:
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

    # zero the engine's own stage stats so report_stats() covers ONLY the
    # timed window, then snapshot the ext counters.
    model.stats.clear()
    model.stat_calls.clear()
    SLICER.reset()
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

    n = args.tokens
    buckets = {k: v / n for k, v in SLICER.buckets.items()}
    calls = {k: v / n for k, v in SLICER.calls.items()}
    samples = {}
    for key, vals in SLICER.samples.items():
        s = sorted(vals)
        samples[key] = {
            "n": len(s), "mean": sum(s) / len(s),
            "median": statistics.median(s),
            "p90": s[int(0.9 * (len(s) - 1))],
            "max": s[-1],
        }
    measured = sum(buckets.values())
    total_ms = total_s / n * 1e3
    remainder_ms = max(0.0, total_ms - measured * 1e3)

    subs = SLICER.calls.get("fence", 0)
    sync_calls = SLICER.calls.get("sync", 0)
    fence_per_tok_ms = SLICER.buckets.get("fence", 0.0) / n * 1e3
    sync_per_tok_ms = SLICER.buckets.get("sync", 0.0) / n * 1e3

    cal = args.calib
    h2d_gbs = cal.get("batch_h2d_gbs" if batched else "sync_h2d_gbs", 0.0)
    d2h_gbs = cal.get("batch_d2h_gbs" if batched else "sync_d2h_gbs", 0.0)
    up_bytes = SLICER.bytes_up / n          # per token
    down_bytes = SLICER.bytes_down / n      # per token
    transfer_ms = 0.0
    if h2d_gbs > 0:
        transfer_ms += up_bytes / (h2d_gbs * 1e9) * 1e3
    if d2h_gbs > 0:
        transfer_ms += down_bytes / (d2h_gbs * 1e9) * 1e3
    # all derived numbers are PER TOKEN, in ms
    fixed_submit_ms = (subs + sync_calls) / n * cal["empty_submit_ms"]
    kernel_est_ms = fence_per_tok_ms - subs / n * cal["empty_submit_ms"] - transfer_ms
    if not batched:
        # sync ops fold DMA+fence together; kernel = everything not bytes
        kernel_est_ms = sync_per_tok_ms - transfer_ms

    return {
        "label": label,
        "batched": batched,
        "tokens": n,
        "prefill_s": prefill_s,
        "s_per_tok": total_s / n,
        "tok_per_s": n / total_s,
        "per_token_ms": [x * 1e3 for x in per_token],
        "per_token_ms_median": statistics.median(x * 1e3 for x in per_token),
        "buckets_ms": {k: v * 1e3 for k, v in buckets.items()},
        "buckets_ms_total": {k: v for k, v in buckets.items()},  # seconds
        "calls_per_tok": calls,
        "samples_ms": samples,
        "remainder_ms": remainder_ms,
        "bytes_up_mib": up_bytes / 2**20,
        "bytes_down_mib": down_bytes / 2**20,
        "submits": subs / n,
        "sync_roundtrips": sync_calls / n,
        "fence_ms_total": fence_per_tok_ms,
        "derived": {
            "empty_submit_ms": cal["empty_submit_ms"],
            "transfer_est_ms": transfer_ms,
            "fixed_submit_est_ms": fixed_submit_ms,
            "kernel_est_ms": max(0.0, kernel_est_ms),
            "h2d_gbs": h2d_gbs,
            "d2h_gbs": d2h_gbs,
        },
        "engine_stats": {k: v / n for k, v in model.stats.items()},
        "engine_stat_calls": {k: v / n for k, v in model.stat_calls.items()},
        "counters_delta": {
            k: _delta(before, after, k) for k in after
            if k not in ("resident_mib", "stat_calls")
        },
        "resident_mib": after["resident_mib"],
        "cache_hit": after["stat_calls"].get("cache_hit", 0),
        "cache_miss": after["stat_calls"].get("cache_miss", 0),
        "ids": ids,
        "env": _env_line(),
        "rss_mb": _rss_mb(),
        "gpu_router": bool(getattr(model, "_gpu_router", False)),
        "exp_cache_budget_mib": getattr(model, "_exp_cache_budget", 0) / 2**20,
        "report_stats": model.report_stats(),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
_ORDER = [
    ("fence", "gpu fence (submit+wait+readback)"),
    ("sync", "sync round trips (submit_and_wait)"),
    ("record", "host record of batch commands"),
    ("async_up", "async queue-2 uploads"),
    ("host_attn", "host attention (KV+SDPA)"),
    ("host_gdn", "host GDN recurrence"),
    ("host_moe", "host router/topk/python MoE"),
    ("host_fetch", "host expert pack+LRU fetch"),
    ("host_rms", "host rmsnorm"),
    ("host_embed", "host embed row"),
    ("host_glue", "host affine glue"),
    ("host_gdn_sync", "host GDN->GPU sync"),
    ("non_gpu", "ext bookkeeping"),
]


def print_report(runs: list[dict], label: str) -> dict:
    med = statistics.median(r["s_per_tok"] for r in runs)
    pick = min(runs, key=lambda r: abs(r["s_per_tok"] - med))
    stoks = sorted(r["s_per_tok"] for r in runs)
    print(f"\n=== [{label}] {len(runs)} runs, s/tok "
          f"{', '.join(f'{x:.3f}' for x in stoks)} (median {med:.3f}) ===")
    print(f"    tok/s median {1 / med:.2f} "
          f"per-token ms (median run): {pick['per_token_ms_median']:.1f} "
          f"| {pick['env']}")
    total = pick["s_per_tok"] * 1e3
    print(f"    {'bucket':<40}{'ms/tok':>9}{'%':>7}{'calls/tok':>11}")
    acc = 0.0
    for key, title in _ORDER:
        ms = pick["buckets_ms"].get(key)
        if not ms:
            continue
        acc += ms
        c = pick["calls_per_tok"].get(key, 0)
        print(f"    {title:<40}{ms:>9.2f}{100 * ms / total:>6.1f}%{c:>11.1f}")
    print(f"    {'remainder (python/torch glue)':<40}"
          f"{pick['remainder_ms']:>9.2f}{100 * pick['remainder_ms'] / total:>6.1f}%")
    d = pick["derived"]
    print(f"    {'TOTAL':<40}{total:>9.2f}{100.0:>6.1f}%")
    print(f"    h2d={pick['bytes_up_mib']:.1f}MiB/tok "
          f"d2h={pick['bytes_down_mib']:.1f}MiB/tok "
          f"submits={pick['submits']} sync_rt={pick['sync_roundtrips']}")
    print(f"    derived: transfer_est={d['transfer_est_ms']:.1f}ms "
          f"fixed_submit_est={d['fixed_submit_est_ms']:.1f}ms "
          f"kernel_est={d['kernel_est_ms']:.1f}ms "
          f"empty_submit={d['empty_submit_ms']:.3f}ms "
          f"h2d={d['h2d_gbs']:.1f}GB/s d2h={d['d2h_gbs']:.1f}GB/s")
    for key, s in pick.get("samples_ms", {}).items():
        print(f"    per-{key}: mean={s['mean']:.3f}ms median={s['median']:.3f}ms "
              f"p90={s['p90']:.3f}ms max={s['max']:.3f}ms n={s['n']}")
    print(f"    engine: {pick['resident_mib']:.0f}MiB resident "
          f"hit={pick['cache_hit']} miss={pick['cache_miss']} "
          f"gpu_router={pick['gpu_router']} "
          f"exp_cache={pick['exp_cache_budget_mib']:.0f}MiB")
    print(f"    counters_delta: {pick['counters_delta']}")
    return pick


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FREETOKEN_TEST_MODEL_GGUF", ""))
    ap.add_argument("--tokens", type=int, default=32,
                    help="timed decode tokens per run")
    ap.add_argument("--warmup", type=int, default=8,
                    help="untimed decode tokens (expert-LRU warmup)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--ctx", type=int, default=0,
                    help="prefill at least this many tokens (0 = the 14-token "
                         "PROMPT); raises context so the host attention path "
                         "is measured at a realistic length")
    ap.add_argument("--sequential", action="store_true",
                    help="FTV_GGUF_BATCHED=0 (sequential decoder arm)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None, help="output path")
    ap.add_argument("--no-calib", action="store_true")
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
    print(f"[{label}] start {time.strftime('%H:%M:%S')} "
          f"batched={batched} prompt={len(prompt)} tok "
          f"tokens={args.tokens} warmup={args.warmup} runs={args.runs} "
          f"env={_env_line()}", flush=True)

    ext = vk.load_ext()
    t0 = time.perf_counter()
    model = ResidentGGUF(args.model, None, max_len=args.max_len)
    print(f"[{label}] load {time.perf_counter() - t0:.1f}s {_env_line()}",
          flush=True)

    saved_ext = _wrap_ext(ext)
    try:
        # Calibration AFTER load: pipelines/pools warm, driver cold path gone.
        if args.no_calib:
            args.calib = {"empty_submit_ms": 0.0, "batch_h2d_gbs": 0.0,
                          "batch_d2h_gbs": 0.0, "sync_h2d_gbs": 0.0,
                          "sync_d2h_gbs": 0.0}
        else:
            args.calib = calibrate(ext)
            print(f"[{label}] calib {json.dumps(args.calib, indent=2)}",
                  flush=True)
        saved_model = _wrap_model(model)
        SLICER.reset()

        runs = []
        try:
            for i in range(args.runs):
                runs.append(run_once(model, ext, args, label, batched, prompt))
                print(f"[{label}] run {i + 1}/{args.runs}: "
                      f"{runs[-1]['s_per_tok']:.3f}s/tok "
                      f"({runs[-1]['tok_per_s']:.2f} tok/s) "
                      f"hit={runs[-1]['cache_hit']} "
                      f"miss={runs[-1]['cache_miss']} {runs[-1]['env']}",
                      flush=True)
        finally:
            _unwrap_model(model, saved_model)
    finally:
        _unwrap_ext(ext, saved_ext)

    pick = print_report(runs, label)

    payload = {
        "label": label,
        "batched": batched,
        "args": vars(args) | {"calib": args.calib},
        "runs": runs,
        "median_run": pick,
        "env_end": _env_line(),
    }
    out = args.json or str(
        pathlib.Path(__file__).with_name(f"token_breakdown_{label}.json"))
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[{label}] wrote {out} | end { _env_line() }", flush=True)
    print(f"[{label}] engine stage stats (median run, s over "
          f"{pick['tokens']} tok):\n{pick['report_stats']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
