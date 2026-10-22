"""llama.cpp side of the perf-gap measurement (same file, same box).

Runs ``llama-cli`` (GGML_VULKAN=ON build) back to back over a fixed set of
arms, 3 runs each (median reported), and parses its own log lines:
load time, offloaded-layer count, prompt-eval rate and eval (decode) rate.
Every run also records Cached/MemAvailable/loadavg from /proc and the
process's peak RSS (VmHWM, polled from /proc/<pid>/status).

    /home/cus/.venv-vllm-vulkan/bin/python tests/vulkan/bench_llamacpp_gap.py \
        --runs 3 --tokens 48 --out /tmp/opencode/llamacpp_gap_raw.json

Arms (edit ARM_EXTRA / arms() to add more):
  ncmoe30      -ngl 999 -ncmoe 30 -t 8     (llama.cpp's fastest shape)
  ncmoe41      -ngl 999 -ncmoe 41 -t 8     (ALL experts on the CPU)
  fit          -fit on -t 8                (default fit-on behaviour)
  ncmoe0       -ngl 999 -ncmoe 0  -t 8     (experts on the GPU; may OOM)
  ncmoe41_t16  -ngl 999 -ncmoe 41 -t 16    (thread collapse check)
  ncmoe30_fa0  -ngl 999 -ncmoe 30 -t 8 -fa off
Raw logs go to --log-dir (outside the repo by default).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import time

CLI = "/home/cus/working/llama.cpp/build/bin/llama-cli"
MODEL = "/tmp/opencode/CyberTiel.gguf"
PROMPT = (
    "In 1969, humanity took its first steps on the Moon. The Apollo 11 "
    "mission landed Neil Armstrong and Buzz Aldrin near the Sea of "
    "Tranquility, while Michael Collins orbited above. The crew returned "
    "samples, planted a flag, and came home after eight days."
)


def arms() -> list[tuple[str, list[str]]]:
    return [
        ("ncmoe30", ["-ngl", "999", "-ncmoe", "30", "-t", "8"]),
        ("ncmoe41", ["-ngl", "999", "-ncmoe", "41", "-t", "8"]),
        ("fit", ["-fit", "on", "-t", "8"]),
        ("ncmoe0", ["-ngl", "999", "-ncmoe", "0", "-t", "8"]),
        ("ncmoe41_t16", ["-ngl", "999", "-ncmoe", "41", "-t", "16"]),
        ("ncmoe30_fa0", ["-ngl", "999", "-ncmoe", "30", "-t", "8", "-fa", "off"]),
    ]


def _meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            if key in ("Cached", "MemFree", "MemAvailable"):
                out[key] = int(rest.split()[0]) // 1024
    return out


def _env() -> dict:
    mi = _meminfo()
    return {
        "cached_mib": mi.get("Cached", -1),
        "mem_avail_mib": mi.get("MemAvailable", -1),
        "load": [round(x, 2) for x in os.getloadavg()],
    }


def _vmhwm_kib(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


_RE = {
    "load_ms": re.compile(r"load time\s*=\s*([\d.]+)\s*ms"),
    "offload": re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers"),
    "bufsize": re.compile(r"model buffer size\s*=\s*(.+)"),
    "prompt_eval": re.compile(
        r"prompt eval time\s*=\s*([\d.]+)\s*/\s*(\d+)\s*runs\s*\(\s*"
        r"([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)"),
    "eval": re.compile(
        r"\beval time\s*=\s*([\d.]+)\s*/\s*(\d+)\s*runs\s*\(\s*"
        r"([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)"),
}


def parse_log(text: str) -> dict:
    out: dict = {}
    for key in ("load_ms", "offload", "bufsize"):
        m = _RE[key].search(text)
        if m:
            out[key] = m.group(1) if key != "offload" else (
                int(m.group(1)), int(m.group(2)))
    for key in ("prompt_eval", "eval"):
        m = _RE[key].search(text)
        if m:
            out[f"{key}_total_ms"] = float(m.group(1))
            out[f"{key}_runs"] = int(m.group(2))
            out[f"{key}_ms_per_tok"] = float(m.group(3))
            out[f"{key}_tps"] = float(m.group(4))
    out["oom"] = bool(re.search(r"out of memory|VK_ERROR_OUT_OF_DEVICE_MEMORY", text))
    out["error"] = bool(re.search(r"error:|assert", text, re.I))
    return out


def run_once(extra: list[str], tokens: int, log_path: pathlib.Path,
             timeout: int = 1800) -> dict:
    cmd = [CLI, "-m", MODEL, "-p", PROMPT, "-n", str(tokens),
           "--temp", "0", "--seed", "42", "--perf", "--no-display-prompt",
           "--simple-io", "-c", "2048"] + extra
    env = dict(os.environ)
    rec: dict = {"cmd": " ".join(cmd), "env_before": _env(), "t0": time.time()}
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env=env, stdin=subprocess.DEVNULL)
    except OSError as exc:
        rec["spawn_error"] = str(exc)
        return rec
    peak = 0
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            peak = max(peak, _vmhwm_kib(proc.pid))
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        rec["timeout"] = True
    except Exception as exc:  # noqa: BLE001
        rec["run_error"] = str(exc)
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    rec["wall_s"] = time.perf_counter() - t0
    rec["rss_peak_mib"] = peak / 1024.0
    rec["rc"] = proc.returncode
    text = "".join(lines)
    log_path.write_text(text)
    rec["parsed"] = parse_log(text)
    rec["env_after"] = _env()
    return rec


def median_run(runs: list[dict], key: str) -> dict:
    vals = [r for r in runs if key in r.get("parsed", {})]
    if not vals:
        return {}
    med = statistics.median(r["parsed"][key] for r in vals)
    return min(vals, key=lambda r: abs(r["parsed"][key] - med))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--log-dir", default="/tmp/opencode/llamacpp_logs")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name(
        "llamacpp_gap_raw.json")))
    ap.add_argument("--only", default=None,
                    help="comma-separated arm names to run (default: all)")
    args = ap.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    print(f"start {time.strftime('%H:%M:%S')} {_env()}", flush=True)

    results = {}
    for name, extra in arms():
        if only and name not in only:
            continue
        print(f"\n=== arm {name}: {' '.join(extra)} ===", flush=True)
        runs = []
        for i in range(args.runs):
            rec = run_once(extra, args.tokens, log_dir / f"{name}_{i}.log")
            runs.append(rec)
            p = rec.get("parsed", {})
            line = (f"  run {i + 1}: tps={p.get('eval_tps', float('nan')):.2f} "
                    f"ms/tok={p.get('eval_ms_per_tok', float('nan')):.1f} "
                    f"pp_tps={p.get('prompt_eval_tps', float('nan')):.1f} "
                    f"load={p.get('load_ms', float('nan')):.0f}ms "
                    f"offload={p.get('offload')} "
                    f"rss={rec.get('rss_peak_mib', 0):.0f}MiB "
                    f"wall={rec.get('wall_s', 0):.0f}s "
                    f"oom={p.get('oom')} err={p.get('error')} "
                    f"| {rec.get('env_after', {})}")
            print(line, flush=True)
            if p.get("oom") or rec.get("timeout"):
                print("  arm looks broken, stopping this arm", flush=True)
                break
        results[name] = {"extra": extra, "runs": runs}
        if runs:
            med = median_run(runs, "eval_tps")
            if med:
                results[name]["median_eval_tps"] = med["parsed"]["eval_tps"]
                results[name]["median"] = med["parsed"]

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(
        {"env": _env(), "tokens": args.tokens, "results": results}, indent=1))
    print(f"\nwrote {out} | end {_env()}", flush=True)
    print("\nsummary (median eval tok/s):")
    for name, res in results.items():
        print(f"  {name:<14} {res.get('median_eval_tps', float('nan')):.2f} tok/s "
              f"ms/tok={res.get('median', {}).get('eval_ms_per_tok', float('nan')):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
