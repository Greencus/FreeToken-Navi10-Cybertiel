"""A/B bench for MTP speculative decoding on the Vulkan GGUF decoder.

Not collected by pytest (no ``test_`` prefix): run it directly to measure the
speculative path against the non-speculative one on the same prompt, and to
answer the question the identity gate cannot - whether drafting pays.

    FREETOKEN_TEST_MODEL_GGUF=<file> PYTHONPATH=python \
        python tests/vulkan/bench_mtp.py --tokens 16 --n-max 3 --p-min 0,0.25

Reported per configuration:

- tok/s and s/tok over the timed generation (steady state: the expert LRU is
  warmed by a discarded warm-up generation first, which is what makes cold
  K-quant dequant and VRAM uploads stop dominating);
- the MTP accept rate and per-phase ms (draft / verify / catch-up / rollback);
- identity: the timed tokens must equal the baseline's - a bench that is fast
  because it invented tokens is worse than no bench;
- RSS and pinned resident bytes (VRAM) at the end of the run.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import resource
import time

import torch

from freetoken.models.qwen3_5_moe.vulkan.mtp import generate_mtp

# Same prompt set as tests/vulkan/test_mtp.py (GGUF tokenizer ids).
PROMPT = [14917, 557, 264, 1248, 1834, 13, 353, 14428, 2046, 13569, 321, 1801, 14799, 13]


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _timed_generate(model, prompt, tokens, warmup):
    if warmup:
        model.generate(prompt, max_new=warmup, temperature=0.0)
    t0 = time.perf_counter()
    out = model.generate(prompt, max_new=tokens, temperature=0.0)
    return out, time.perf_counter() - t0


def _timed_mtp(model, prompt, tokens, warmup, n_max, p_min):
    if warmup:
        generate_mtp(model, prompt, max_new=warmup, n_max=n_max, p_min=p_min)
    t0 = time.perf_counter()
    out = generate_mtp(model, prompt, max_new=tokens, n_max=n_max, p_min=p_min)
    return out, time.perf_counter() - t0


def _batching_probe(model, head, prompt, tokens):
    """Sequential vs batched cost of the same trunk rows (the MTP verify).

    One snapshot of the post-prefill state, three arms that each replay the
    same ``tokens`` rows from it: the production sequential ``decode_step``,
    then the batched ``verify`` once with one row and once with all rows.
    If batching does not amortize dispatch, MTP cannot win.
    """
    model.reset()
    head.reset()
    logits = model.prefill_chunked(prompt, chunk_size=64)
    pos = model._seq_len
    snap = {i: t.clone() for i, t in model.gdn_state.items()}
    conv = {i: t.clone() for i, t in model.conv_buf.items()}
    kvc = {i: t.clone() for i, t in model.k_cache.items()}
    vcv = {i: t.clone() for i, t in model.v_cache.items()}

    # One greedy row stream to replay in every arm (advanced from the
    # snapshot, then rewound by the resets below).
    ids = [int(logits.argmax())]
    while len(ids) < tokens:
        logits = model.decode_step(model.embed_row(ids[-1]), pos + len(ids) - 1)
        ids.append(int(logits.argmax()))

    def _rewind():
        for i, t in snap.items():
            model.gdn_state[i].copy_(t)
        for i, t in conv.items():
            model.conv_buf[i].copy_(t)
        for i, t in kvc.items():
            model.k_cache[i].copy_(t)
        for i, t in vcv.items():
            model.v_cache[i].copy_(t)
        model._gdn_gpu_dirty = True

    _rewind()
    t0 = time.perf_counter()
    for off, token in enumerate(ids):
        model.decode_step(model.embed_row(token), pos + off)
    seq_s = time.perf_counter() - t0

    _rewind()
    t0 = time.perf_counter()
    head.verify(pos, ids[:1])
    bat1_s = time.perf_counter() - t0

    _rewind()
    t0 = time.perf_counter()
    head.verify(pos, ids)
    batn_s = time.perf_counter() - t0
    _rewind()
    return {
        "rows": tokens,
        "sequential_ms": seq_s * 1e3,
        "batched_1row_ms": bat1_s * 1e3,
        "batched_nrow_ms": batn_s * 1e3,
        "seq_per_row_ms": seq_s * 1e3 / tokens,
        "batch_per_row_ms": batn_s * 1e3 / tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FREETOKEN_TEST_MODEL_GGUF", ""))
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--n-max", type=int, default=3)
    ap.add_argument("--p-min", default="0.0", help="comma-separated p_min values")
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()
    if not args.model or not pathlib.Path(args.model).is_file():
        raise SystemExit("set --model or FREETOKEN_TEST_MODEL_GGUF to the GGUF file")

    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF
    from freetoken.models.qwen3_5_moe.vulkan.mtp import enable_mtp

    torch.set_num_threads(8)
    t0 = time.perf_counter()
    model = ResidentGGUF(args.model, None, max_len=args.max_len)
    print(f"load {time.perf_counter() - t0:.1f}s rss={_rss_mb():.0f}MB", flush=True)

    base_tokens, base_s = _timed_generate(model, PROMPT, args.tokens, args.warmup)
    print(
        f"baseline  {args.tokens} tok in {base_s:.1f}s = {base_s / args.tokens:.3f}s/tok "
        f"({args.tokens / base_s:.2f} tok/s)",
        flush=True,
    )

    head = enable_mtp(model, n_max=args.n_max)
    print(_batching_probe(model, head, PROMPT, min(8, args.tokens)), flush=True)

    for p_min in [float(x) for x in args.p_min.split(",")]:
        mtp_tokens, mtp_s = _timed_mtp(
            model, PROMPT, args.tokens, args.warmup, args.n_max, p_min
        )
        st = dict(model.mtp.stats)
        drafted, accepted = st["drafted"], st["accepted"]
        rate = accepted / drafted if drafted else 0.0
        ident = "IDENTICAL" if mtp_tokens == base_tokens else "DIVERGED"
        rounds = st["rounds"]
        passes = rounds + st["rollbacks"] + (drafted + rounds) / 40
        print(
            f"mtp n_max={args.n_max} p_min={p_min:g}  {args.tokens} tok in {mtp_s:.1f}s "
            f"= {mtp_s / args.tokens:.3f}s/tok ({args.tokens / mtp_s:.2f} tok/s)  "
            f"accept {accepted:.0f}/{drafted:.0f}={rate:.2f}  "
            f"rounds {rounds:.0f} rollbacks {st['rollbacks']:.0f}  "
            f"passes/token {passes / args.tokens:.3f}  {ident}",
            flush=True,
        )
        print(
            f"    prefill {st['ms_prefill']:.0f}ms draft {st['ms_draft']:.0f}ms "
            f"verify {st['ms_verify']:.0f}ms catchup {st['ms_catchup']:.0f}ms "
            f"rollback {st['ms_rollback']:.0f}ms",
            flush=True,
        )
        if mtp_tokens != base_tokens:
            for i, (a, b) in enumerate(zip(base_tokens, mtp_tokens)):
                if a != b:
                    print(f"    first mismatch at {i}: baseline {a} mtp {b}", flush=True)
                    break
    print(
        f"rss_max={_rss_mb():.0f}MB resident={model.ext.resident_bytes_total() / 2**20:.0f}MiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
