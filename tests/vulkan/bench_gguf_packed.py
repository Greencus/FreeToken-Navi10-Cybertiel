"""Decode-only bench for the packed K-quant GGUF path (Vulkan).

Not collected by pytest (no ``test_`` prefix): run it directly to measure the
DECODE-ONLY cost of the CyberTiel GGUF with the in-shader K-quant lane, and to
A/B it against the host-dequant fallback. ``bench_mtp.py`` times whole
``generate()`` calls (prefill included); this one prefills once and then times
only the ``decode_step`` loop, which is the number that has to be compared
against the 29.7 s/token host-dequant baseline and llama.cpp's 16-18 tok/s.

    FREETOKEN_TEST_MODEL_GGUF=<file> PYTHONPATH=python \
        python tests/vulkan/bench_gguf_packed.py --tokens 8 --warmup 4

    FTV_GGUF_PACKED=0 ... bench_gguf_packed.py   # host-dequant fallback arm

Reports tok/s, s/tok, VRAM (resident bytes), RSS, the expert-cache hit/miss
counters and the per-stage timer split (``report_stats``), whose
``exp_dequant`` / ``exp_packed`` keys say outright how much host dequant the
run did (packed arm: zero).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import resource
import time

import torch

# Same prompt set as tests/vulkan/test_mtp.py and bench_mtp.py (GGUF ids).
PROMPT = [14917, 557, 264, 1248, 1834, 13, 353, 14428, 2046, 13569, 321, 1801, 14799, 13]


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# stat_calls keys whose every call is one synchronous submit+fence round
# trip (update/upload/download_resident in the C++).  Deltas of these over
# the timed loop = the sequential path's round-trip count.
_SYNC_KEYS = (
    "pin_up", "pin_kernel", "pin_down",
    "pin_batch_kernel", "pin_batch_down",
    "exp_x", "exp_kernel", "exp_down", "exp_up",
    "exp_batch_kernel", "exp_batch_down",
)


def _round_trips(calls: dict) -> int:
    return sum(calls.get(k, 0) for k in _SYNC_KEYS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FREETOKEN_TEST_MODEL_GGUF", ""))
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--label", default=None,
                    help="tag for the printed line (default: packed/fallback from env)")
    ap.add_argument("--batched", action="store_true",
                    help="decode via decode_step_batched (gguf_batched.py "
                         "K-quant recorders) instead of decode_step")
    args = ap.parse_args()
    if not args.model or not pathlib.Path(args.model).is_file():
        raise SystemExit("set --model or FREETOKEN_TEST_MODEL_GGUF to the GGUF file")

    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

    torch.set_num_threads(8)
    packed = os.environ.get("FTV_GGUF_PACKED", "1").strip() != "0"
    label = args.label or (
        ("packed-batched" if args.batched else "packed") if packed else "fallback"
    )
    t0 = time.perf_counter()
    model = ResidentGGUF(args.model, None, max_len=args.max_len)
    print(f"[{label}] load {time.perf_counter() - t0:.1f}s rss={_rss_mb():.0f}MB "
          f"resident={model.ext.resident_bytes_total() / 2**20:.0f}MiB", flush=True)

    t0 = time.perf_counter()
    logits = model.prefill_chunked(PROMPT, chunk_size=64)
    prefill_s = time.perf_counter() - t0
    pos = model._seq_len
    step = model.decode_step_batched if args.batched else model.decode_step
    for _ in range(args.warmup):
        token = int(logits.argmax().item())
        logits = step(model.embed_row(token), pos)
        pos += 1
        model._seq_len = pos

    rt0 = _round_trips(model.stat_calls)
    sb0 = model.ext.batch_submit_count()[0]
    t0 = time.perf_counter()
    ids: list[int] = []
    for _ in range(args.tokens):
        token = int(logits.argmax().item())
        ids.append(token)
        logits = step(model.embed_row(token), pos)
        pos += 1
        model._seq_len = pos
    dt = time.perf_counter() - t0
    n_rt = _round_trips(model.stat_calls) - rt0
    n_sub = model.ext.batch_submit_count()[0] - sb0

    calls = model.stat_calls
    print(
        f"[{label}] prefill {len(PROMPT)} tok in {prefill_s:.1f}s; "
        f"decode {args.tokens} tok in {dt:.1f}s = {dt / args.tokens:.3f}s/tok "
        f"({args.tokens / dt:.3f} tok/s)",
        flush=True,
    )
    print(
        f"[{label}] vram={model.ext.resident_bytes_total() / 2**20:.0f}MiB "
        f"rss_max={_rss_mb() / 1024:.2f}GiB "
        f"experts cache_hit={calls.get('cache_hit', 0)} "
        f"cache_miss={calls.get('cache_miss', 0)}",
        flush=True,
    )
    print(
        f"[{label}] round_trips={n_rt} "
        f"({n_rt / args.tokens:.0f}/tok) batch_submits={n_sub} "
        f"({n_sub / args.tokens:.1f}/tok)",
        flush=True,
    )
    print(f"[{label}] token_ids: {ids}", flush=True)
    print(f"[{label}] stats:\n{model.report_stats()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
