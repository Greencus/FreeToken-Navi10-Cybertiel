"""STEP-0 falsifier: one command buffer for a WHOLE decode token.

The GPU-resident-token design (docs/vulkan/gpu-resident-token.md, section
10) claims the 45.35 ms/tok fence bucket over 91 drains is mostly
pipeline-drain latency, not GPU work.  The cheapest test of that claim:

- ``control`` arm: the normal decode_step_batched walk (one command
  buffer submit per batch: ~91 per token); records the fence time of
  every batch_submit.
- ``onebuf`` arm: batch_begin/batch_submit are patched so the WHOLE
  token records into ONE command buffer and the host submits ONCE at the
  end; the single fence wait is then the GPU's real per-token work plus
  the transfers (the host walk still runs, but its mid-token readbacks
  only fill at the final submit, so the OUTPUT of this arm is garbage -
  only the fence number is meaningful).

If onebuf's single fence collapses toward ~20 ms, the drains were
latency and the projection holds; if it stays near the control arm's
fence SUM, the GPU work really is ~45 ms and the plan must re-scope.

    FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \
        /home/cus/.venv-vllm-vulkan/bin/python tests/vulkan/bench_onebuf.py \
        --tokens 4

Changes NO engine or kernel behaviour outside the onebuf arm's patched
batch_begin/batch_submit (restored after each token).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bench_token_breakdown import PROMPT, _env_line  # noqa: E402


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _prefill(model, prompt):
    model.reset()
    logits = model.prefill_chunked(list(prompt), chunk_size=64)
    return logits, model._seq_len


def run_control(model, ext, tokens: int) -> dict:
    """Normal walk; per-token wall + fence sum + submit count."""
    real_submit = ext.batch_submit
    fences: list[float] = []
    submits: list[int] = []
    cur = {"t": 0.0, "n": 0}

    def timed_submit(*a, **k):
        t0 = time.perf_counter()
        try:
            return real_submit(*a, **k)
        finally:
            cur["t"] += time.perf_counter() - t0
            cur["n"] += 1

    ext.batch_submit = timed_submit
    walls: list[float] = []
    try:
        for _ in range(tokens):
            token = int(_LOGITS["v"].argmax().item())
            cur["t"], cur["n"] = 0.0, 0
            t0 = time.perf_counter()
            _LOGITS["v"] = model.decode_step_batched(
                model.embed_row(token), _POS["v"])
            walls.append(time.perf_counter() - t0)
            fences.append(cur["t"])
            submits.append(cur["n"])
            _POS["v"] += 1
            model._seq_len = _POS["v"]
    finally:
        ext.batch_submit = real_submit
    return {
        "arm": "control",
        "wall_ms_median": _median([w * 1e3 for w in walls]),
        "fence_ms_median": _median([f * 1e3 for f in fences]),
        "fence_ms_sum": sum(f * 1e3 for f in fences),
        "submits_median": _median([float(n) for n in submits]),
        "per_token": [
            {"wall_ms": w * 1e3, "fence_ms": f * 1e3, "submits": n}
            for w, f, n in zip(walls, fences, submits)
        ],
    }


def run_onebuf(model, ext, tokens: int) -> dict:
    """Whole token in ONE command buffer; ONE fence at the end."""
    real_begin, real_submit = ext.batch_begin, ext.batch_submit
    state = {"open": False}
    swallowed = {"begin": 0, "submit": 0}

    def begin():
        if not state["open"]:
            real_begin()
            state["open"] = True
        else:
            swallowed["begin"] += 1

    def submit(*_a, **_k):
        swallowed["submit"] += 1  # keep recording; never end the buffer

    walls: list[float] = []
    fences: list[float] = []
    ext.batch_begin, ext.batch_submit = begin, submit
    try:
        for _ in range(tokens):
            token = int(_LOGITS["v"].argmax().item())
            swallowed["begin"] = swallowed["submit"] = 0
            state["open"] = False          # each token = one fresh buffer
            t0 = time.perf_counter()
            _LOGITS["v"] = model.decode_step_batched(
                model.embed_row(token), _POS["v"])
            walls.append(time.perf_counter() - t0)
            _POS["v"] += 1
            model._seq_len = _POS["v"]
            # close THIS token's buffer and time its single fence while
            # the patch is still installed (so a nested begin/submit can
            # never slip through between walk and submit)
            if state["open"]:
                t0 = time.perf_counter()
                real_submit()
                fences.append(time.perf_counter() - t0)
    finally:
        ext.batch_begin, ext.batch_submit = real_begin, real_submit
    return {
        "arm": "onebuf",
        "wall_ms_median": _median([w * 1e3 for w in walls]),
        "fence_ms_median": _median([f * 1e3 for f in fences]),
        "fence_ms_sum": sum(f * 1e3 for f in fences),
        "submits_median": 1.0,
        "swallowed_begin_median": _median([float(swallowed["begin"])]),
        "per_token": [
            {"wall_ms": w * 1e3, "fence_ms": f * 1e3, "submits": 1}
            for w, f in zip(walls, fences)
        ],
    }


_LOGITS: dict = {}
_POS: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=os.environ.get("FREETOKEN_TEST_MODEL_GGUF", ""))
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--json", default=None)
    ap.add_argument("--prefetch", action="store_true",
                    help="keep FTV_GGUF_PREFETCH=1 (default off here: "
                         "queue-2 uploads require a closed main batch, "
                         "which the onebuf arm keeps open)")
    args = ap.parse_args()
    if not args.model or not pathlib.Path(args.model).is_file():
        raise SystemExit("set --model or FREETOKEN_TEST_MODEL_GGUF")
    if not args.prefetch:
        os.environ["FTV_GGUF_PREFETCH"] = "0"

    from freetoken.kernel import vulkan as vk
    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

    torch.set_num_threads(8)
    print(f"[onebuf] start {time.strftime('%H:%M:%S')} "
          f"tokens={args.tokens} env={_env_line()}", flush=True)
    ext = vk.load_ext()
    model = ResidentGGUF(args.model, None, max_len=args.max_len)

    _LOGITS["v"], _POS["v"] = _prefill(model, PROMPT)
    ctrl = run_control(model, ext, args.tokens)
    print(f"[onebuf] control: wall {ctrl['wall_ms_median']:.2f} ms "
          f"fence {ctrl['fence_ms_median']:.2f} ms "
          f"submits {ctrl['submits_median']:.0f}", flush=True)

    _LOGITS["v"], _POS["v"] = _prefill(model, PROMPT)
    one = run_onebuf(model, ext, args.tokens)
    print(f"[onebuf] onebuf:  wall {one['wall_ms_median']:.2f} ms "
          f"fence {one['fence_ms_median']:.2f} ms "
          f"(first token fence {one['per_token'][0]['fence_ms']:.2f} ms)",
          flush=True)

    payload = {"control": ctrl, "onebuf": one, "env_end": _env_line()}
    out = args.json or "/tmp/opencode/step0_onebuf.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[onebuf] wrote {out} | end {_env_line()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
