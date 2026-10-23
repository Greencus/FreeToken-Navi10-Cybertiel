"""Per-stage cost attribution for chunked PREFILL (the decode lane's
``bench_token_overhead.py`` measures one token; this measures the whole
prompt pass).

It changes NO engine or kernel behaviour: wall-clock wrappers around the
already-exported ``ext`` functions (reused from ``bench_token_breakdown``)
plus per-name call counts, and model-method wrappers around the three
chunk stages (MoE / GDN / full attention) plus the shared helpers.

    FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \
        /home/cus/.venv-vllm-vulkan/bin/python \
        tests/vulkan/bench_prefill_attribution.py --ctx 512 --chunk 64

What it reports:

  sync       one synchronous submit-and-wait round trip (upload / kernel /
             download) - the prefill's launch-bound cost
  record     a batch_* call recorded into an open command buffer (host
             vkCmd* only; no drain)
  fence      batch_submit: one drain of the whole open batch
  pre_moe / pre_gdn / pre_attn  EXCLUSIVE python time inside the three
             chunk stages (their ext/child time is subtracted)
  host_rms / host_glue / host_fetch / host_embed  the shared helpers

and, from the per-name counters: exact op counts, so "round trips per
chunk / per layer" and "dispatches per chunk" are counted, not estimated.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bench_token_breakdown import (  # noqa: E402
    PROMPT,
    SLICER,
    _delta,
    _env_line,
    _loadavg,
    _unwrap_ext,
    _wrap_ext,
    calibrate,
    snapshot,
)

# Ops that record a COMPUTE dispatch into an open batch (vs a copy).
_DISPATCH_OPS = {
    "batch_kquant_io", "batch_nvfp4_io", "batch_bf16_io", "batch_fp32_io",
    "batch_swiglu_io", "batch_vecadd_io", "batch_rms1p_io",
    "batch_topk256_io", "batch_gdn_conv_io", "batch_gdn_recur_io",
    "batch_attn_prep_qwen35_io", "batch_attn_fwd_qwen35_io",
    "batch_moe_record", "batch_moe_fused_record", "batch_moe_kquant_record",
    "batch_moe_kquant_fused_record", "batch_kquant_gateup8_io",
    "batch_kquant_acc8_io", "batch_nvfp4_gateup8_io", "batch_nvfp4_acc8_io",
    "batch_fill",
}
_COPY_OPS = {"batch_upload", "batch_upload_multi", "batch_download",
             "batch_upload_async", "batch_upload_async_flush"}
_SYNC_DISPATCH_SUFFIX = ("_resident_io",)

NAME_CALLS: dict[str, int] = {}

# model methods, category (children of these are subtracted)
_STAGE_MAP = {
    "_moe_chunk": "pre_moe",
    "_gdn_chunk": "pre_gdn",
    "_full_attn_chunk": "pre_attn",
    "_rms": "host_rms",
    "_pinned_affine_batch": "host_glue",
    "_pinned_affine": "host_glue",
    "_expert_affine_batch": "host_glue",
    "_expert_handle": "host_fetch",
    "_gguf_gu_handle": "host_fetch",
    "_gguf_dn_handle": "host_fetch",
    "embed_row": "host_embed",
}


def _wrap_counts(ext) -> dict:
    saved: dict = {}
    for name in sorted(dir(ext)):
        if name.startswith("_"):
            continue
        fn = getattr(ext, name)
        if not callable(fn):
            continue
        saved[name] = fn

        def make(name=name, fn=fn):
            def wrapper(*args, **kwargs):
                NAME_CALLS[name] = NAME_CALLS.get(name, 0) + 1
                return fn(*args, **kwargs)
            return wrapper

        setattr(ext, name, make())
    return saved


def _unwrap_counts(ext, saved: dict) -> None:
    for name, fn in saved.items():
        setattr(ext, name, fn)


def _wrap_stages(model) -> dict:
    saved: dict = {}
    for attr, cat in _STAGE_MAP.items():
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


def _unwrap_stages(model, saved: dict) -> None:
    for attr, fn in saved.items():
        setattr(model, attr, fn)


def _counts_summary() -> dict:
    sync = {k: v for k, v in NAME_CALLS.items()
            if not k.startswith("batch_")}
    dispatch = {k: v for k, v in NAME_CALLS.items() if k in _DISPATCH_OPS}
    copies = {k: v for k, v in NAME_CALLS.items() if k in _COPY_OPS}
    return {
        "sync_ops": sync,
        "recorded_dispatches": dispatch,
        "recorded_copies": copies,
        "n_sync_roundtrips": sum(sync.values()),
        "n_recorded_dispatches": sum(dispatch.values()),
        "n_recorded_copies": sum(copies.values()),
        "n_sync_dispatches": sum(v for k, v in sync.items()
                                 if k.endswith(_SYNC_DISPATCH_SUFFIX)),
        "all": dict(sorted(NAME_CALLS.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FREETOKEN_TEST_MODEL_GGUF", ""))
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.model or not pathlib.Path(args.model).is_file():
        raise SystemExit("set --model or FREETOKEN_TEST_MODEL_GGUF")

    prompt = list(PROMPT)
    while len(prompt) < args.ctx:
        prompt += PROMPT
    prompt = prompt[: max(args.ctx, len(PROMPT))]
    chunks = (len(prompt) + args.chunk - 1) // args.chunk

    from freetoken.kernel import vulkan as vk
    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

    torch.set_num_threads(8)
    label = args.label or "prefill"
    print(f"[{label}] start {time.strftime('%H:%M:%S')} ctx={len(prompt)} "
          f"chunk={args.chunk} chunks={chunks} env={_env_line()}", flush=True)

    ext = vk.load_ext()
    t0 = time.perf_counter()
    model = ResidentGGUF(args.model, None, max_len=4096)
    print(f"[{label}] load {time.perf_counter() - t0:.1f}s {_env_line()}",
          flush=True)

    saved_ext = _wrap_ext(ext)
    saved_cnt = _wrap_counts(ext)
    try:
        if args.no_calib:
            calib = {"empty_submit_ms": 0.0}
        else:
            calib = calibrate(ext)
            print(f"[{label}] calib {json.dumps(calib)}", flush=True)
        saved_model = _wrap_stages(model)
        runs = []
        try:
            for _ in range(args.runs):
                model.reset()
                model.stats.clear()
                model.stat_calls.clear()
                SLICER.reset()
                before = snapshot(ext, model)
                SLICER.reset()
                NAME_CALLS.clear()
                t0 = time.perf_counter()
                logits = model.prefill_chunked(prompt, chunk_size=args.chunk)
                dt = time.perf_counter() - t0
                after = snapshot(ext, model)
                runs.append({
                    "prefill_s": dt,
                    "tok_per_s": len(prompt) / dt,
                    "buckets_s": dict(SLICER.buckets),
                    "calls": dict(SLICER.calls),
                    "counts": _counts_summary(),
                    "bump_s": dict(model.stats),
                    "stat_calls": dict(model.stat_calls),
                    "counters": {k: _delta(before, after, k) for k in after
                                 if k not in ("resident_mib", "stat_calls")},
                    "resident_mib": after["resident_mib"],
                    "loadavg": _loadavg(),
                    "ids": [int(logits.argmax().item())],
                })
                print(f"[{label}] run {len(runs)}: prefill {dt:.2f}s "
                      f"({len(prompt) / dt:.2f} tok/s) "
                      f"sync={runs[-1]['counts']['n_sync_roundtrips']} "
                      f"rec_disp={runs[-1]['counts']['n_recorded_dispatches']} "
                      f"load={_loadavg()}", flush=True)
        finally:
            _unwrap_stages(model, saved_model)
    finally:
        _unwrap_counts(ext, saved_cnt)
        _unwrap_ext(ext, saved_ext)

    best = min(runs, key=lambda r: r["prefill_s"])
    n_layers = len(model.config.layer_types)
    print(f"\n=== [{label}] best {best['prefill_s']:.2f}s "
          f"= {best['tok_per_s']:.2f} tok/s | load {best['loadavg']} ===")
    print(f"    chunks={chunks} layers={n_layers} ids={best['ids']}")
    print(f"    sync round trips: {best['counts']['n_sync_roundtrips']} "
          f"total | {best['counts']['n_sync_roundtrips'] / chunks:.0f}/chunk "
          f"| {best['counts']['n_sync_roundtrips'] / chunks / n_layers:.1f}/layer")
    print(f"    recorded dispatches: {best['counts']['n_recorded_dispatches']} "
          f"total | {best['counts']['n_recorded_dispatches'] / chunks:.0f}/chunk")
    print(f"    recorded copies: {best['counts']['n_recorded_copies']} total")
    print(f"    sync dispatches (kernels): {best['counts']['n_sync_dispatches']}")
    print("    exclusive buckets (s):")
    for k, v in sorted(best["buckets_s"].items(), key=lambda kv: -kv[1]):
        print(f"      {k:>14} {v:8.3f}  ({100 * v / best['prefill_s']:5.1f}%)"
              f"  calls={best['calls'].get(k, 0)}")
    print("    per-name counts:")
    for k, v in list(best["counts"]["all"].items())[:30]:
        print(f"      {k:>34} {v}")
    print("    model _bump stats (s):")
    for k, v in sorted(best["bump_s"].items(), key=lambda kv: -kv[1]):
        print(f"      {k:>20} {v:8.3f}")
    print(f"    counters: {best['counters']}")
    print(f"    resident_mib {best['resident_mib']:.0f}")

    out = args.json or str(pathlib.Path("/tmp/opencode")
                           / f"prefill_attr_{label}.json")
    with open(out, "w") as fh:
        json.dump({"label": label, "args": vars(args), "runs": runs,
                   "calib": calib, "env_end": _env_line()}, fh, indent=1)
    print(f"[{label}] wrote {out} | end {_env_line()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
