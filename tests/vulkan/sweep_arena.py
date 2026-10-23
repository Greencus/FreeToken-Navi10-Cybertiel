"""Arena-size sweep: slots/layer -> steady-state ms/tok, upload traffic, VRAM peak.

The arena is VRAM-limited (28 slots/layer = 2047 MiB of an 8176 MiB card).
The design doc's own table says 48 slots/layer drops the steady-state
uploads from 67/token to 6/token (178 -> 16 MiB/token of PCIe + host
staging copy).  This script measures that on THIS box, past the cold-start
fill, and watches the driver's VRAM so a GTT spill shows up as a cliff.

    FTV_GGUF_ARENA_SLOTS=40 PYTHONPATH=python python sweep_arena.py --tokens 96
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bench_token_breakdown import PROMPT  # noqa: E402

MODEL = os.environ.get("FREETOKEN_TEST_MODEL_GGUF", "/tmp/opencode/CyberTiel.gguf")
VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"


def vram_mib() -> float:
    try:
        with open(VRAM) as fh:
            return int(fh.read().strip()) / 2**20
    except OSError:
        return -1.0


def disk_read_mib() -> float:
    """Bytes actually read from the block device (not the page cache)."""
    try:
        with open("/proc/self/io") as fh:
            for line in fh:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return -1.0


def env_line() -> str:
    out = subprocess.run(
        ["bash", "-lc",
         "echo \"cached=$(awk '/MemCached/{print $2}' /proc/meminfo)kB "
         "mem_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB "
         "load=$(cut -d' ' -f1-3 /proc/loadavg)\""],
        capture_output=True, text=True).stdout.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    slots = os.environ.get("FTV_GGUF_ARENA_SLOTS", "28")
    print(f"[sweep] slots={slots} tokens={args.tokens} env={env_line()}",
          flush=True)

    from freetoken.kernel import vulkan as vk
    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

    torch.set_num_threads(8)
    ext = vk.load_ext()
    model = ResidentGGUF(args.model, None, max_len=args.max_len)

    model.reset()
    logits = model.prefill_chunked(list(PROMPT), chunk_size=64)
    pos = model._seq_len
    print(f"[sweep] prefilled seq_len={pos} vram={vram_mib():.0f}MiB", flush=True)

    walls: list[float] = []
    disk: list[float] = []
    vram_peak = vram_mib()
    up0 = None
    cnt0 = None
    t_start = time.perf_counter()
    disk_prev = disk_read_mib()
    for i in range(args.tokens + args.warmup):
        tok = int(logits.argmax().item())
        t0 = time.perf_counter()
        logits = model.decode_step_batched(model.embed_row(tok), pos)
        dt = time.perf_counter() - t0
        pos += 1
        model._seq_len = pos
        vram_peak = max(vram_peak, vram_mib())
        if i >= args.warmup:
            walls.append(dt * 1e3)
            now = disk_read_mib()
            disk.append(now - disk_prev)
            disk_prev = now
        if i == args.warmup - 1:
            up0 = ext.batch_upload_async_stats()
            cnt0 = dict(model.stat_calls)
    total = time.perf_counter() - t_start
    up1 = ext.batch_upload_async_stats()

    # steady state: the last half of the measured window
    half = len(walls) // 2
    steady = walls[half:]
    early = walls[:half]
    med = statistics.median
    payload = {
        "slots": slots,
        "tokens": len(walls),
        "median_ms": med(walls),
        "early_median_ms": med(early),
        "steady_median_ms": med(steady),
        "steady_tok_s": 1.0 / (med(steady) / 1e3),
        "all_tok_s": len(walls) / total,
        "vram_peak_mib": vram_peak,
        "resident_mib": ext.resident_bytes_total() / 2**20,
        "q2_submits_delta": (up1[0] - up0[0]) if up0 else -1,
        "q2_uploads_delta": (up1[1] - up0[1]) if up0 else -1,
        "stat_calls": {k: v for k, v in model.stat_calls.items()
                       if isinstance(v, (int, float))},
        "env": env_line(),
        "disk_read_mib_per_tok": disk,
        "disk_read_mib_total": sum(disk),
        "per_token_ms": walls,
    }
    print(f"[sweep] slots={slots} median {med(walls):.1f} ms "
          f"early {med(early):.1f} steady {med(steady):.1f} ms "
          f"({payload['steady_tok_s']:.2f} tok/s steady) "
          f"vram_peak {vram_peak:.0f}MiB resident {payload['resident_mib']:.0f}MiB "
          f"q2_uploads/token {(up1[1]-up0[1])/max(1,len(walls)):.1f} "
          f"DISK {sum(disk)/max(1,len(walls)):.1f}MiB/token "
          f"(first8 {sum(disk[:8]):.0f} last8 {sum(disk[-8:]):.0f} MiB)",
          flush=True)
    out = args.json or f"/tmp/opencode/sweep_arena_{slots}.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[sweep] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
