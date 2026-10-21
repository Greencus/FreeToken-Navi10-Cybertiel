"""A/B bench for MTP speculative decoding on the Vulkan GGUF decoder.

Not collected by pytest (no ``test_`` prefix): run it directly to measure the
speculative path against the non-speculative one on the same prompt, and to
answer the question the identity gate cannot - whether drafting pays.

    FREETOKEN_TEST_MODEL_GGUF=<file> PYTHONPATH=python \
        python tests/vulkan/bench_mtp.py --tokens 32 --rounds 3 \
            --n-max 3 --p-min 0.25

Measurement rules (benchmarks lie on a loaded box):

- ``--rounds`` interleaves the arms (base, then each p_min, per round) so
  both see the same thermal/cache drift; the verdict is the MEDIAN s/tok
  per arm, and every round's MTP tokens must equal that round's baseline
  tokens (identity every round, not once);
- a discarded warm-up generation warms the expert LRU first, so cold
  K-quant dequant and VRAM uploads stop dominating the timed windows;
- decode-only s/tok is reported next to wall s/tok: prefill is timed
  separately (baseline ``prefill_chunked``; MTP's own ms_prefill stats
  include its prefill_with_h + draft catch-up) and subtracted per arm;
- ``/proc/meminfo`` (Cached, swap-in-use), VRAM used and loadavg are
  printed at start and end - a run with swap climbing or a loaded box is
  flagged, not averaged over;
- native counter deltas per token (submits, record barriers, queue-2
  submits) separate structural cost from wall-clock noise.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import resource
import statistics
import time

import torch

from freetoken.models.qwen3_5_moe.vulkan.mtp import generate_mtp

# Same prompt set as tests/vulkan/test_mtp.py (GGUF tokenizer ids).
PROMPT = [14917, 557, 264, 1248, 1834, 13, 353, 14428, 2046, 13569, 321, 1801, 14799, 13]


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            parts = v.split()
            if parts and parts[0].isdigit():
                out[k] = int(parts[0])  # kB
    return out


def _vram_used_mb() -> int | None:
    for p in sorted(pathlib.Path("/sys/class/drm").glob(
            "card*/device/mem_info_vram_used")):
        try:
            return int(p.read_text().strip()) // 2**20
        except OSError:
            continue
    return None


def _env_line(tag: str) -> str:
    m = _meminfo()
    swap_used = max(0, m.get("SwapTotal", 0) - m.get("SwapFree", 0)) // 1024
    load = os.getloadavg()[0]
    vram = _vram_used_mb()
    flag = " SWAP-RISING" if swap_used > 8 * 1024 else ""
    flag += " LOADED" if load > 4.0 else ""
    return (
        f"env {tag}: avail={m.get('MemAvailable', 0) // 1024}MiB "
        f"cached={m.get('Cached', 0) // 1024}MiB swap_used={swap_used}MiB "
        f"vram={vram if vram is not None else '?'}MiB load={load:.2f}{flag}"
    )


def _counters(model) -> dict[str, int]:
    ext = model.ext
    return {
        "submits": ext.batch_submit_count()[0],
        "barriers": ext.batch_barrier_stats()[0],
        "q2": ext.batch_upload_async_stats()[0],
    }


def _submits_delta(c0: dict, c1: dict) -> int:
    return c1["submits"] - c0["submits"]


def _delta_str(c0: dict, c1: dict, tokens: int) -> str:
    d = lambda k: (c1[k] - c0[k]) / tokens
    return f"submits/tok {d('submits'):.1f} barriers/tok {d('barriers'):.1f} q2/tok {d('q2'):.1f}"


def _timed_generate(model, tokens, warmup, batched=True):
    if warmup:
        model.generate(PROMPT, max_new=warmup, temperature=0.0,
                       prefill_chunk=64, batched=batched)
    c0 = _counters(model)
    t0 = time.perf_counter()
    out = model.generate(PROMPT, max_new=tokens, temperature=0.0,
                         prefill_chunk=64, batched=batched)
    submits = _submits_delta(c0, _counters(model))
    return out, time.perf_counter() - t0, _delta_str(c0, _counters(model), tokens), submits


def _timed_mtp(model, tokens, warmup, n_max, p_min):
    if warmup:
        generate_mtp(model, PROMPT, max_new=warmup, n_max=n_max, p_min=p_min)
    c0 = _counters(model)
    t0 = time.perf_counter()
    out = generate_mtp(model, PROMPT, max_new=tokens, n_max=n_max, p_min=p_min)
    wall = time.perf_counter() - t0
    st = dict(model.mtp.stats)
    submits = _submits_delta(c0, _counters(model))
    return out, wall, st, _delta_str(c0, _counters(model), tokens), submits


def _timed_prefill(model) -> float:
    """Baseline prefill wall time, for the decode-only split."""
    model.reset()
    t0 = time.perf_counter()
    model.prefill_chunked(PROMPT, chunk_size=64)
    return time.perf_counter() - t0


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
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--n-max", type=int, default=3)
    ap.add_argument("--p-min", default="0.25", help="comma-separated p_min values")
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()
    if not args.model or not pathlib.Path(args.model).is_file():
        raise SystemExit("set --model or FREETOKEN_TEST_MODEL_GGUF to the GGUF file")

    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF
    from freetoken.models.qwen3_5_moe.vulkan.mtp import enable_mtp

    torch.set_num_threads(8)
    print(_env_line("start"), flush=True)
    print(
        f"mtp knobs: n_max={args.n_max} p_min={args.p_min} "
        f"skip_kv_snapshot={os.environ.get('FTV_MTP_SKIP_KV_SNAPSHOT', '0')}",
        flush=True,
    )
    t0 = time.perf_counter()
    model = ResidentGGUF(args.model, None, max_len=args.max_len)
    print(f"load {time.perf_counter() - t0:.1f}s rss={_rss_mb():.0f}MB", flush=True)

    p_mins = [float(x) for x in args.p_min.split(",")]

    # Discarded warm-up for both arms, then a separate timed prefill so the
    # decode-only split has the baseline's own number to subtract.
    _timed_generate(model, args.warmup, 0)
    base_prefill_s = _timed_prefill(model)
    print(f"baseline prefill {base_prefill_s:.2f}s", flush=True)

    head = enable_mtp(model, n_max=args.n_max)
    generate_mtp(model, PROMPT, max_new=args.warmup, n_max=args.n_max,
                 p_min=p_mins[0])
    print(_batching_probe(model, head, PROMPT, min(8, args.tokens)), flush=True)

    base_wall: list[float] = []
    base_dec: list[float] = []
    base_tokens: list[list[int]] = []
    base_submits: list[int] = []
    mtp_wall: dict[float, list[float]] = {p: [] for p in p_mins}
    mtp_dec: dict[float, list[float]] = {p: [] for p in p_mins}
    mtp_ident: dict[float, int] = {p: 0 for p in p_mins}
    mtp_last_st: dict[float, dict] = {p: {} for p in p_mins}
    mtp_submits: dict[float, list[int]] = {p: [] for p in p_mins}

    for rnd in range(1, args.rounds + 1):
        bt, b_wall, b_cnt, b_sub = _timed_generate(model, args.tokens, 0)
        base_wall.append(b_wall)
        base_dec.append(max(0.0, b_wall - base_prefill_s))
        base_tokens.append(bt)
        base_submits.append(b_sub)
        print(
            f"round {rnd} base   {args.tokens}tok {b_wall:.2f}s "
            f"{b_wall / args.tokens * 1e3:.1f}ms/tok "
            f"decode-only {base_dec[-1] / args.tokens * 1e3:.1f}ms/tok "
            f"submits {b_sub}  {b_cnt}",
            flush=True,
        )
        for p in p_mins:
            mt, m_wall, st, m_cnt, m_sub = _timed_mtp(
                model, args.tokens, 0, args.n_max, p)
            ident = mt == bt
            mtp_ident[p] += int(ident)
            mtp_wall[p].append(m_wall)
            mtp_submits[p].append(m_sub)
            dec = max(0.0, m_wall - st.get("ms_prefill", 0.0) / 1e3)
            mtp_dec[p].append(dec)
            mtp_last_st[p] = st
            drafted, accepted = st["drafted"], st["accepted"]
            rate = accepted / drafted if drafted else 0.0
            rows = drafted + st.get("rounds", 0.0)
            print(
                f"round {rnd} mtp p_min={p:g} {args.tokens}tok {m_wall:.2f}s "
                f"{m_wall / args.tokens * 1e3:.1f}ms/tok "
                f"decode-only {dec / args.tokens * 1e3:.1f}ms/tok  "
                f"accept {accepted}/{drafted}={rate:.2f}  "
                f"rows/tok {rows / args.tokens:.3f} submits {m_sub}  {m_cnt}  "
                f"{'IDENTICAL' if ident else 'DIVERGED'}",
                flush=True,
            )

    med_base_w = statistics.median(base_wall)
    med_base_d = statistics.median(base_dec)
    print(
        f"\nmedian base:   wall {med_base_w / args.tokens * 1e3:.1f}ms/tok "
        f"({args.tokens / med_base_w:.2f} tok/s)  decode-only "
        f"{med_base_d / args.tokens * 1e3:.1f}ms/tok "
        f"({args.tokens / med_base_d:.2f} tok/s)  "
        f"submits/round {statistics.median(base_submits):.0f}",
        flush=True,
    )
    for p in p_mins:
        med_w = statistics.median(mtp_wall[p])
        med_d = statistics.median(mtp_dec[p])
        st = mtp_last_st[p]
        rounds = st.get("rounds", 0)
        drafted = st.get("drafted", 0)
        rows = drafted + rounds
        passes = rounds + st.get("rollbacks", 0) + rows / 40
        speed_w = med_base_w / med_w
        speed_d = med_base_d / med_d
        print(
            f"median mtp p_min={p:g}: wall {med_w / args.tokens * 1e3:.1f}ms/tok "
            f"({args.tokens / med_w:.2f} tok/s)  decode-only "
            f"{med_d / args.tokens * 1e3:.1f}ms/tok "
            f"({args.tokens / med_d:.2f} tok/s)  "
            f"x{speed_w:.2f} wall / x{speed_d:.2f} decode  "
            f"identity {mtp_ident[p]}/{args.rounds}  "
            f"passes/tok {passes / args.tokens:.3f}  "
            f"rows/tok {rows / args.tokens:.3f}  "
            f"submits/round {statistics.median(mtp_submits[p]):.0f}",
            flush=True,
        )
        print(
            f"    last round: prefill {st.get('ms_prefill', 0):.0f}ms "
            f"draft {st.get('ms_draft', 0):.0f}ms verify {st.get('ms_verify', 0):.0f}ms "
            f"catchup {st.get('ms_catchup', 0):.0f}ms "
            f"rollback {st.get('ms_rollback', 0):.0f}ms "
            f"snapshot {st.get('ms_snapshot', 0):.0f}ms "
            f"({st.get('ms_snapshot', 0) / max(rounds, 1):.0f}ms/round)",
            flush=True,
        )
    print(
        f"rss_max={_rss_mb():.0f}MB "
        f"resident={model.ext.resident_bytes_total() / 2**20:.0f}MiB",
        flush=True,
    )
    print(_env_line("end"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
