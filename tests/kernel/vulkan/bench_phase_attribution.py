"""GPU timeline micro-benchmark: per-barrier cost + per-phase attribution.

Lane-B Phase 1/2 companion to the timestamp instrumentation added to
``ext_module.cpp`` (``batch_ts_enable`` / ``batch_ts_read`` /
``batch_barrier_probe``).  Answers, with hardware timestamps:

1. BARRIER COST (always runs, no model): N ``vector_add`` dispatches on
   DISTINCT output buffers (so the hazard logic stays silent - the only
   barriers are the probe's) separated by
   ``batch_barrier_probe(mode)``:

     mode 0 = today's device-scope barrier (COMPUTE|TRANSFER stages,
              MEMORY_WRITE -> MEMORY_READ|WRITE)
     mode 1 = shader-scoped barrier (COMPUTE stage, SHADER_WRITE ->
              SHADER_READ)
     mode 2 = no barrier

   per-barrier ns = sum of the ``barrier_probe`` timestamp deltas / N,
   cross-checked with the diff-of-totals (arm total - mode-2 total)/N.
   The per-barrier cost times the engine's ~211 barriers/token is the
   suspected ~14 ms/token of non-kernel GPU time.

2. lm_head GEMV (always runs, no model): synthetic 248320 x 2048 BF16
   weight (the real head shape: 1017 MB of weights per token read),
   one ``batch_bf16_io`` dispatch per round, GPU ns from timestamps.

3. TOKEN TIMELINE (needs FREETOKEN_TEST_MODEL_GGUF): a full decode token
   with the timeline enabled; every ``batch_submit`` is wrapped so its
   per-op GPU deltas are captured per submit.  Reports GPU ns per label
   per token (gdn_conv, gdn_recur, the fused K-quant MoE shaders, the
   router tail, barriers, ...) and isolates lm_head as the last submit's
   ``linear_bf16_resident_f32`` dispatch (that batch holds exactly the
   head GEMV).

4. GDN ONE-LAYER REPLAY (needs the model): replays batch A of one
   linear-attention layer - in-projs, conv, recur, out_proj, vecadd -
   with per-piece timestamps, so each piece of a GDN layer is priced on
   its own.

Run (quiet box, one GPU job at a time):

    FREETOKEN_TEST_MODEL_GGUF=/tmp/opencode/CyberTiel.gguf PYTHONPATH=python \
        /home/cus/.venv-vllm-vulkan/bin/python \
        tests/kernel/vulkan/bench_phase_attribution.py [--parts 1,2,3,4]

The timeline marks add a few extra commands per op (the timestamp writes
themselves); they never reorder or alter the recorded dispatches, so the
fork2 gate stays bit-exact with the instrumentation on and off.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent
                       / "vulkan"))

from bench_token_breakdown import PROMPT, _env_line  # noqa: E402


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _skip_unless_vulkan(ext) -> None:
    import freetoken.kernel.vulkan as vk

    if not vk.available():
        raise SystemExit("no usable Vulkan device on this machine")


# ------------------------------------------------------------------
# Part 1: barrier cost (device-scope vs shader-scoped vs none).
# ------------------------------------------------------------------
def part1_barrier_cost(ext, rounds: int = 7, n: int = 64) -> dict:
    n_el = 32768
    g = torch.Generator().manual_seed(1234)
    h_a = ext.upload_resident(
        (0.1 * torch.randn(n_el, generator=g)).contiguous())
    h_b = ext.upload_resident(
        (0.1 * torch.randn(n_el, generator=g)).contiguous())
    # Distinct y per dispatch: no two dispatches share a buffer, so
    # batch_ensure_ordered never fires; the probe barriers are the only
    # ones in the command buffer.
    ys = [ext.allocate_resident(n_el * 4) for _ in range(n)]
    results: dict[str, dict] = {}
    try:
        for mode, name in ((0, "device_scope"), (1, "shader_scope"),
                           (2, "none")):
            barrier_ns, total_ns, wall_ms = [], [], []
            for _ in range(rounds):
                ext.batch_ts_enable(True)
                ext.batch_begin()
                for i in range(n):
                    ext.batch_vecadd_io(h_a, h_b, ys[i], n_el)
                    ext.batch_barrier_probe(mode)
                wall_ms.append(ext.batch_submit())
                pairs, trunc = ext.batch_ts_read()
                assert not trunc, "timestamp pool overflowed"
                barrier_ns.append(
                    sum(ns for lab, ns in pairs
                        if lab == "barrier_probe") / n)
                total_ns.append(sum(ns for _lab, ns in pairs))
            ext.batch_ts_enable(False)
            results[name] = {
                "per_barrier_mark_ns_median": _median(barrier_ns),
                "total_gpu_ms_median": _median(total_ns) * 1e-6,
                "wall_ms_median": _median(wall_ms),
            }
    finally:
        for h in [h_a, h_b, *ys]:
            ext.free_resident(h)
    # Diff-of-totals cross-check: (arm total - none total) / N.
    for name in ("device_scope", "shader_scope"):
        results[name]["per_barrier_diff_ns"] = (
            (results[name]["total_gpu_ms_median"]
             - results["none"]["total_gpu_ms_median"]) * 1e6 / n)
    return results


# ------------------------------------------------------------------
# Part 2: the lm_head GEMV alone (synthetic weights, real head shape).
# ------------------------------------------------------------------
def part2_lm_head(ext, rounds: int = 5) -> dict:
    rows, cols = 248320, 2048
    t0 = time.perf_counter()
    w = torch.randn(rows, cols, dtype=torch.bfloat16)
    h_w = ext.upload_resident(w.contiguous())
    del w
    h_x = ext.upload_resident(
        torch.randn(1, cols, dtype=torch.float32).contiguous())
    h_y = ext.allocate_resident(rows * 4)
    setup_s = time.perf_counter() - t0
    gpu_ns, walls = [], []
    try:
        for _ in range(rounds):
            ext.batch_ts_enable(True)
            ext.batch_begin()
            ext.batch_bf16_io(h_x, h_w, h_w, h_y, 1, rows, cols, False)
            walls.append(ext.batch_submit())
            pairs, trunc = ext.batch_ts_read()
            assert not trunc
            gpu_ns.append(sum(ns for lab, ns in pairs
                              if lab == "linear_bf16_resident_f32"))
    finally:
        ext.batch_ts_enable(False)
        for h in (h_x, h_w, h_y):
            ext.free_resident(h)
    gpu_ms = [x * 1e-6 for x in gpu_ns]
    return {
        "shape": [rows, cols],
        "weights_mb": rows * cols * 2 / 2**20,
        "gpu_ms_median": _median(gpu_ms),
        "gpu_ms_all": gpu_ms,
        "wall_ms_median": _median(walls),
        "setup_s": setup_s,
    }


# ------------------------------------------------------------------
# Part 3: full-token GPU timeline (model-gated).
# ------------------------------------------------------------------
def part3_token_timeline(model, ext, tokens: int = 6) -> dict:
    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF  # noqa

    model.reset()
    logits = model.prefill_chunked(list(PROMPT), chunk_size=64)
    pos = model._seq_len

    real_submit = ext.batch_submit
    cur: dict = {"label_ns": {}, "submits": []}

    def timed_submit(*a, **k):
        wall = real_submit(*a, **k)
        pairs, trunc = ext.batch_ts_read()
        assert not trunc, "timestamp pool overflowed"
        seg: dict[str, float] = {}
        for lab, ns in pairs:
            seg[lab] = seg.get(lab, 0.0) + ns
        cur["label_ns"] = {k2: cur["label_ns"].get(k2, 0.0) + seg.get(k2, 0.0)
                           for k2 in set(cur["label_ns"]) | set(seg)}
        cur["submits"].append(seg)
        return wall

    per_token: list[dict] = []
    ext.batch_ts_enable(True)
    ext.batch_submit = timed_submit
    try:
        for _ in range(tokens):
            token = int(logits.argmax().item())
            cur["label_ns"], cur["submits"] = {}, []
            t0 = time.perf_counter()
            logits = model.decode_step_batched(model.embed_row(token), pos)
            wall_ms = (time.perf_counter() - t0) * 1e3
            pos += 1
            model._seq_len = pos
            per_token.append({
                "wall_ms": wall_ms,
                "label_ns": dict(cur["label_ns"]),
                "n_submits": len(cur["submits"]),
                # lm_head = the token's LAST submit (norm is host; that
                # batch holds upload + the head GEMV + download only).
                # The head GEMV's label is linear_bf16_resident_f32 or,
                # with FTV_DENSE_Q80, one of the linear_q80gguf_* variants.
                "lm_head_gpu_ms": sum(ns for lab, ns
                                      in cur["submits"][-1].items()
                                      if lab.startswith(
                                          ("linear_bf16_resident",
                                           "linear_q80gguf"))) * 1e-6,
            })
    finally:
        ext.batch_submit = real_submit
        ext.batch_ts_enable(False)

    labels = sorted({k2 for t in per_token for k2 in t["label_ns"]})
    med: dict[str, float] = {}
    for lab in labels:
        med[lab] = _median([t["label_ns"].get(lab, 0.0) * 1e-6
                            for t in per_token])
    gpu_tok_ms = _median([sum(t["label_ns"].values()) * 1e-6
                          for t in per_token])
    return {
        "wall_ms_median": _median([t["wall_ms"] for t in per_token]),
        "gpu_token_ms_median": gpu_tok_ms,
        "submits_median": _median([float(t["n_submits"]) for t in per_token]),
        "lm_head_gpu_ms_median": _median([t["lm_head_gpu_ms"]
                                          for t in per_token]),
        "per_label_gpu_ms_median": med,
        "per_token": per_token,
    }


# ------------------------------------------------------------------
# Part 4: one GDN layer replay, per-piece (model-gated).
# ------------------------------------------------------------------
def part4_gdn_layer(model, ext, layer: int = 0) -> dict:
    from freetoken.models.qwen3_5_moe.vulkan.nvfp4_resident import _translate

    cfg = model.config
    g = torch.Generator().manual_seed(7)
    h = (0.1 * torch.randn(cfg.hidden_size, generator=g)).contiguous().float()
    x = (0.1 * torch.randn(cfg.hidden_size, generator=g)).contiguous().float()
    base = f"language_model.model.layers.{layer}"
    opt = model.pinned[_translate(f"{base}.linear_attn.out_proj")]
    out = torch.empty(cfg.hidden_size)
    gpu_ns: list[dict] = []
    walls: list[float] = []
    for _ in range(5):
        ext.batch_ts_enable(True)
        ext.batch_begin()
        ext.batch_upload_multi([model.h_x_fused, model.h_x_resid],
                               [h, x], [0, 0])
        model._record_gdn_in(layer, ext, download=False)
        ext.batch_gdn_conv_io(
            model.h_qkv, model.conv_buf_gpu[layer], model.gdn_taps[layer],
            model.h_mixed, model._gdn_C)
        ext.batch_gdn_recur_io(
            model.h_mixed, model.h_z, model.h_a, model.h_b,
            model.gdn_alog[layer], model.gdn_dtb[layer],
            model.gdn_normw[layer], model.gdn_state_gpu[layer], model.h_x2,
            model._gdn_C, model._gdn_key_dim,
            cfg.linear_num_value_heads, cfg.linear_key_head_dim,
            cfg.linear_value_head_dim, model._gdn_rep,
            model._gdn_scale, 1e-6, cfg.rms_norm_eps)
        ext.batch_bf16_io(
            model.h_x2, opt.handle, opt.handle, model.h_ao,
            1, opt.rows, opt.cols, False)
        ext.batch_vecadd_io(model.h_x_resid, model.h_ao, model.h_x1,
                            cfg.hidden_size)
        ext.batch_download(model.h_x1, out)
        walls.append(ext.batch_submit())
        pairs, trunc = ext.batch_ts_read()
        assert not trunc
        seg: dict[str, float] = {}
        for lab, ns in pairs:
            seg[lab] = seg.get(lab, 0.0) + ns
        gpu_ns.append(seg)
    ext.batch_ts_enable(False)
    labels = sorted({k2 for seg in gpu_ns for k2 in seg})
    med = {lab: _median([seg.get(lab, 0.0) * 1e-6 for seg in gpu_ns])
           for lab in labels}
    return {
        "layer": layer,
        "per_label_gpu_ms_median": med,
        "layer_gpu_ms_median": _median([sum(seg.values()) * 1e-6
                                        for seg in gpu_ns]),
        "wall_ms_median": _median(walls),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=os.environ.get("FREETOKEN_TEST_MODEL_GGUF", ""))
    ap.add_argument("--parts", default="1,2",
                    help="comma list: 1=barrier 2=lm_head 3=timeline "
                         "4=gdn_layer (3,4 need the model)")
    ap.add_argument("--json", default="/tmp/opencode/phase_attribution.json")
    ap.add_argument("--rounds", type=int, default=0,
                    help="override per-part round counts")
    args = ap.parse_args()
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    need_model = bool(parts & {"3", "4"})
    if need_model and (not args.model or not pathlib.Path(args.model).is_file()):
        raise SystemExit("parts 3/4 need --model or FREETOKEN_TEST_MODEL_GGUF")

    from freetoken.kernel import vulkan as vk

    print(f"[phase] start {time.strftime('%H:%M:%S')} parts={sorted(parts)} "
          f"env={_env_line()}", flush=True)
    ext = vk.load_ext()
    _skip_unless_vulkan(ext)
    payload: dict = {"parts": {}}

    if "1" in parts:
        r = part1_barrier_cost(ext)
        payload["parts"]["barrier"] = r
        print(f"[phase] barrier/device_scope: "
              f"{r['device_scope']['per_barrier_mark_ns_median']:.0f} ns/mark "
              f"{r['device_scope']['per_barrier_diff_ns']:.0f} ns/diff",
              flush=True)
        print(f"[phase] barrier/shader_scope: "
              f"{r['shader_scope']['per_barrier_mark_ns_median']:.0f} ns/mark "
              f"{r['shader_scope']['per_barrier_diff_ns']:.0f} ns/diff",
              flush=True)
        print(f"[phase] barrier/none:          "
              f"{r['none']['per_barrier_mark_ns_median']:.0f} ns/mark "
              f"(total {r['none']['total_gpu_ms_median']:.3f} ms vs "
              f"{r['device_scope']['total_gpu_ms_median']:.3f} device-scope)",
              flush=True)

    if "2" in parts:
        r = part2_lm_head(ext)
        payload["parts"]["lm_head"] = r
        print(f"[phase] lm_head synthetic {r['shape']}: "
              f"{r['gpu_ms_median']:.3f} ms GPU "
              f"({r['weights_mb']:.0f} MB weights)", flush=True)

    if need_model:
        from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

        model = ResidentGGUF(args.model, None, max_len=4096)
        if "3" in parts:
            r = part3_token_timeline(model, ext)
            payload["parts"]["timeline"] = r
            print(f"[phase] token timeline: wall {r['wall_ms_median']:.2f} ms "
                  f"gpu {r['gpu_token_ms_median']:.2f} ms "
                  f"submits {r['submits_median']:.0f} "
                  f"lm_head {r['lm_head_gpu_ms_median']:.3f} ms", flush=True)
            for lab, ms in sorted(r["per_label_gpu_ms_median"].items(),
                                  key=lambda kv: -kv[1]):
                print(f"[phase]   {lab:36s} {ms:8.3f} ms/tok", flush=True)
        if "4" in parts:
            r = part4_gdn_layer(model, ext)
            payload["parts"]["gdn_layer"] = r
            print(f"[phase] GDN layer {r['layer']}: "
                  f"{r['layer_gpu_ms_median']:.3f} ms GPU "
                  f"(wall {r['wall_ms_median']:.2f} ms)", flush=True)
            for lab, ms in sorted(r["per_label_gpu_ms_median"].items(),
                                  key=lambda kv: -kv[1]):
                print(f"[phase]   {lab:36s} {ms:8.4f} ms/layer", flush=True)

    payload["env_end"] = _env_line()
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[phase] wrote {args.json} | end {_env_line()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
