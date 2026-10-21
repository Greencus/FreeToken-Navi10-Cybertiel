"""Decode-only A/B bench for static hot-set expert residency (Vulkan GGUF).

Not collected by pytest (no ``test_`` prefix): run it directly to measure
the packed-path decode cost with and without the pinned hot set on the
CyberTiel GGUF. Mirrors ``bench_gguf_packed.py``'s timed section (prefill
once, warmup, then only the ``decode_step`` loop) and adds:

- ``--hotset``: build ``HotsetResidentGGUF`` instead of ``ResidentGGUF``;
- exact per-token expert BYTES: the three handle fetchers are wrapped to
  sum the packed bytes of every cache miss (misses are the only uploads);
- ``--identity``: greedy generate of "Name three primary colors."
  (max_new=4, temperature=0) and a check against the known-good ids.

    FREETOKEN_TEST_MODEL_GGUF=<file> PYTHONPATH=python \
        python tests/vulkan/bench_gguf_hotset.py --tokens 8 --warmup 4
    FREETOKEN_TEST_MODEL_GGUF=<file> PYTHONPATH=python \
        python tests/vulkan/bench_gguf_hotset.py --hotset --tokens 8 --warmup 4

Hot-set arm env defaults (override via env): FTV_GGUF_HOTSET_MB=4096,
FTV_GGUF_HOTSET_K=0 (table's 24), FTV_GGUF_HOTSET_KEYS=both.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import resource
import time

import torch

PROMPT = [14917, 557, 264, 1248, 1834, 13, 353, 14428, 2046, 13569, 321, 1801, 14799, 13]
IDENTITY_PROMPT = "Name three primary colors."
IDENTITY_IDS = [5844, 11, 13358, 11]


def identity_prompt_ids(tokenizer) -> list[int]:
    """cli.py build_prompt: chat template + generation prompt (no thinking).

    The reference ids were measured through the CLI, which renders the
    chat template by default; the RAW text greedily continues differently
    ([271, 248068, ...]) on both the hot-set and the plain path.
    """
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": IDENTITY_PROMPT}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
    )
    return [int(i) for i in out["input_ids"]]


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _install_byte_counters(model) -> dict:
    """Wrap the expert handle fetchers; return a one-key dict of miss bytes."""
    missed: dict[str, float] = {"bytes": 0.0}

    def wrap(orig, nbytes_of):
        def inner(*args):
            before = model.stat_calls.get("cache_miss", 0)
            out = orig(*args)
            if model.stat_calls.get("cache_miss", 0) > before:
                missed["bytes"] += nbytes_of(*args)
            return out
        return inner

    model._expert_handle = wrap(
        model._expert_handle, lambda matrix, e: float(matrix.packed_bytes))
    if hasattr(model, "_gguf_gu_handle"):
        model._gguf_gu_handle = wrap(
            model._gguf_gu_handle,
            lambda g, u, e: float(g.packed_bytes + u.packed_bytes))
        model._gguf_dn_handle = wrap(
            model._gguf_dn_handle, lambda d, e: float(d.packed_bytes))
    return missed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FREETOKEN_TEST_MODEL_GGUF", ""))
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--hotset", action="store_true",
                    help="build HotsetResidentGGUF (pinned hot set) instead "
                         "of plain ResidentGGUF")
    ap.add_argument("--identity", action="store_true",
                    help="greedy-generate the identity prompt first and "
                         "check the known-good ids")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    if not args.model or not pathlib.Path(args.model).is_file():
        raise SystemExit("set --model or FREETOKEN_TEST_MODEL_GGUF to the GGUF file")

    torch.set_num_threads(8)
    label = args.label or ("hotset" if args.hotset else "baseline")

    if args.hotset:
        os.environ.setdefault("FTV_GGUF_HOTSET", "1")
        from freetoken.models.qwen3_5_moe.vulkan.hotset_resident import (
            HotsetResidentGGUF,
        )
        cls = HotsetResidentGGUF
    else:
        from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF
        cls = ResidentGGUF

    t0 = time.perf_counter()
    model = cls(args.model, None, max_len=args.max_len)
    missed = _install_byte_counters(model)
    print(f"[{label}] load {time.perf_counter() - t0:.1f}s rss={_rss_mb():.0f}MB "
          f"resident={model.ext.resident_bytes_total() / 2**20:.0f}MiB", flush=True)

    if args.identity:
        from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

        tok = load_gguf_tokenizer(args.model)
        ids = identity_prompt_ids(tok)
        out = model.generate(ids, max_new=4, temperature=0.0, seed=1234)
        ok = out == IDENTITY_IDS
        print(f"[{label}] identity: {out} expected {IDENTITY_IDS} "
              f"{'MATCH' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            return 2
        # generate() reset the sequence; counters keep cumulative state

    # Snapshot after warmup so prefill/warmup traffic is excluded from the
    # decode-only numbers below.
    vram0 = model.ext.resident_bytes_total()

    t0 = time.perf_counter()
    logits = model.prefill_chunked(PROMPT, chunk_size=64)
    prefill_s = time.perf_counter() - t0
    pos = model._seq_len
    for _ in range(args.warmup):
        token = int(logits.argmax().item())
        logits = model.decode_step(model.embed_row(token), pos)
        pos += 1
        model._seq_len = pos

    miss1 = model.stat_calls.get("cache_miss", 0)
    hit1 = model.stat_calls.get("cache_hit", 0)
    bytes1 = missed["bytes"]

    t0 = time.perf_counter()
    ids = []
    for _ in range(args.tokens):
        token = int(logits.argmax().item())
        ids.append(token)
        logits = model.decode_step(model.embed_row(token), pos)
        pos += 1
        model._seq_len = pos
    dt = time.perf_counter() - t0

    misses = model.stat_calls.get("cache_miss", 0) - miss1
    hits = model.stat_calls.get("cache_hit", 0) - hit1
    bts = missed["bytes"] - bytes1
    print(
        f"[{label}] prefill {len(PROMPT)} tok in {prefill_s:.1f}s; "
        f"decode {args.tokens} tok in {dt:.1f}s = {dt / args.tokens:.3f}s/tok "
        f"({args.tokens / dt:.3f} tok/s)",
        flush=True,
    )
    print(
        f"[{label}] decode misses={misses} ({misses / args.tokens:.1f}/tok) "
        f"hits={hits} ({hits / max(1, hits + misses) * 100:.1f}% "
        f"of requests) bytes={bts / 2**20:.1f}MiB "
        f"({bts / args.tokens / 2**20:.1f} MiB/tok)",
        flush=True,
    )
    print(
        f"[{label}] vram={model.ext.resident_bytes_total() / 2**20:.0f}MiB "
        f"(was {vram0 / 2**20:.0f} at prefill start) "
        f"rss_max={_rss_mb() / 1024:.2f}GiB",
        flush=True,
    )
    print(f"[{label}] token_ids: {ids}", flush=True)
    if hasattr(model, "hotset_info"):
        info = model.hotset_info()
        print(
            f"[{label}] hotset: pinned {info['pinned_experts']} experts / "
            f"{info['pinned_entries']} entries / "
            f"{info['pinned_bytes'] / 2**20:.0f}MiB "
            f"(pin {info['pin_seconds']:.1f}s, mode={info['mode']}, "
            f"lru_region={info['lru_region'] / 2**20:.0f}MiB)",
            flush=True,
        )
    print(f"[{label}] stats:\n{model.report_stats()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
