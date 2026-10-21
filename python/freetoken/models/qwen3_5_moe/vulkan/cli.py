"""Chat/generation CLI for the Vulkan Qwen3.5-MoE engine.

Port of the fork's ``python/chat_qwen35.py`` + ``python/chat_ressl.py``:
loads a checkpoint through ``engine.load_engine`` (NVFP4 auto-detected),
prefills a prompt (chat template unless --raw-prompt), streams/samples
tokens with the fork's sampling defaults, and prints timing.

Usage::

    PYTHONPATH=python python -m freetoken.models.qwen3_5_moe.vulkan.cli \
        --model /path/to/checkpoint --prompt "The capital of France is" \
        --max-tokens 16

Dependency-light: numpy + torch + the Vulkan extension for the engine;
``transformers`` is imported only to render the chat template (falls back
to raw ``tokenizers`` encoding without it).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import List

import torch

from . import engine
from .config import load_qwen35_config
from .nvfp4 import MODEL_DIR


def load_tokenizer(model_dir: pathlib.Path):
    """Fork behavior: transformers AutoTokenizer (chat template support).

    Falls back to a bare ``tokenizers.Tokenizer`` (raw encode only) when
    transformers is unavailable; chat-template prompts then need
    --raw-prompt.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return _load_tokenizers_fallback(model_dir)
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"cannot load tokenizer at {model_dir}: {exc}") from exc


def _load_tokenizers_fallback(model_dir: pathlib.Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "tokenizer requires transformers (or tokenizers) in this environment"
        ) from exc
    tok_file = pathlib.Path(model_dir) / "tokenizer.json"
    if not tok_file.exists():
        raise RuntimeError(f"no tokenizer.json at {model_dir}")
    return Tokenizer.from_file(str(tok_file))


def _as_ids(out) -> List[int]:
    """Normalize tokenizer outputs (list / dict / encoding) to list[int]."""
    if isinstance(out, str):
        raise RuntimeError("chat template returned a string; encode failed")
    try:
        ids = out["input_ids"]
    except (KeyError, TypeError, IndexError):
        ids = out.ids if hasattr(out, "ids") else list(out)
    try:
        return [int(i) for i in ids]
    except (ValueError, TypeError) as exc:
        raise RuntimeError("tokenizer produced non-integer token ids") from exc


def build_prompt(tokenizer, text: str, thinking: bool = False) -> List[int]:
    """Fork chat_qwen35.build_prompt: chat template with generation prompt."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "tokenizer cannot render a chat template (no transformers); "
            "use --raw-prompt"
        )
    try:
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=thinking,
            tokenize=True,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("chat template render failed") from exc
    if isinstance(out, str):
        out = tokenizer.encode(out)
    return _as_ids(out)


def encode_raw(tokenizer, text: str) -> List[int]:
    out = tokenizer.encode(text)
    if not isinstance(out, list):
        try:
            return _as_ids(out)
        except RuntimeError as exc:
            raise RuntimeError("tokenizer.encode produced bad ids") from exc
    return [int(i) for i in out]


def smoke(model, tokenizer) -> None:
    """Fork smoke(): one decode_step on the template'd 'Hi' prompt, top-5."""
    ids = build_prompt(tokenizer, "Hi")
    print(f"smoke prompt ids: {ids[:8]}...", flush=True)
    model.reset()
    started = time.time()
    logits = model.decode_step(model.embed_row(ids[0]), 0)
    ms = (time.time() - started) * 1000.0
    top5 = torch.topk(logits, 5)
    print(f"decode_step ms={ms:.0f} finite={torch.isfinite(logits).all().item()}",
          flush=True)
    print(f"top5 ids={top5.indices.tolist()}", flush=True)
    print(f"top5 toks={[tokenizer.decode([i]) for i in top5.indices.tolist()]}",
          flush=True)
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite smoke logits")
    print("SMOKE_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m freetoken.models.qwen3_5_moe.vulkan.cli",
        description="Vulkan Qwen3.5-MoE chat/generation CLI",
    )
    parser.add_argument(
        "--model", default=os.environ.get("FTV_MODEL_DIR", str(MODEL_DIR)),
        help="checkpoint directory (NVFP4 auto-detected via hf_quant_config.json)",
    )
    parser.add_argument("--gguf", default=None,
                        help="single qwen35moe GGUF file (CyberTiel); overrides --model")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--raw-prompt", action="store_true",
                        help="skip the chat template; encode prompt text directly")
    parser.add_argument("--thinking", action="store_true",
                        help="enable thinking mode in the chat template")
    parser.add_argument("--max-tokens", "--max-new", dest="max_tokens",
                        type=int, default=32, help="max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="top-k filter (0 = off)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="nucleus filter (1.0 = off)")
    parser.add_argument("--cache-mb", type=int, default=None,
                        help="FTV_EXP_CACHE_MB: VRAM budget for expert cache")
    parser.add_argument("--host-pack-mb", type=int, default=None,
                        help="FTV_HOST_PACK_MB: host packed-expert cache budget")
    parser.add_argument("--dequant-cache-mb", type=int, default=None,
                        help="FTV_GGUF_DEQUANT_CACHE_MB: host K-quant dequant cache")
    parser.add_argument("--emit-json", default=None,
                        help="write {prompt: {text, token_ids, ...}} JSON here "
                             "(merges if the file exists)")
    parser.add_argument("--gpu-nvfp4", type=int, choices=(0, 1), default=None,
                        help="FTV_NVFP4_GPU: 1 = GPU NVFP4 kernel, 0 = host-dequant BF16")
    parser.add_argument("--prefill-chunk", type=int, default=64,
                        help="batched prefill chunk size (fork: 64 fastest)")
    parser.add_argument("--max-len", type=int, default=4096,
                        help="KV/RoPE capacity (model max 262144)")
    parser.add_argument("--kv-dtype", default="fp32", choices=("fp32", "bf16"),
                        help="KV cache dtype (bf16 = capacity option only)")
    parser.add_argument("--batch", action="store_true",
                        help="NVFP4 batched decode path (task #12)")
    parser.add_argument("--no-batch", dest="batch", action="store_false")
    parser.set_defaults(batch=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mtp", action="store_true",
                        help="MTP (nextn) speculative drafting; greedy only, "
                             "needs --gguf (blk.40 nextn block)")
    parser.add_argument("--mtp-n-max", type=int, default=3,
                        help="MTP drafts per round (llama.cpp default 3)")
    parser.add_argument("--mtp-n-min", type=int, default=0,
                        help="discard a draft shorter than this many tokens")
    parser.add_argument("--mtp-p-min", type=float, default=0.0,
                        help="MTP draft top-1 probability gate (0.25 recommended "
                             "for this model)")
    parser.add_argument("--smoke", action="store_true",
                        help="wiring check: 1 decode_step, logits stats")
    parser.add_argument("--device", default="vulkan",
                        help="accepted for compatibility; Vulkan is the only backend here")
    args = parser.parse_args()
    if args.device != "vulkan":
        raise SystemExit(f"unsupported --device {args.device!r} (only 'vulkan')")
    # Fail these before pinning weights: both need the model built to matter,
    # and neither should cost a load to discover.
    if args.mtp and args.gguf is None:
        raise SystemExit(
            "--mtp drafts from the GGUF nextn block (blk.40); pass --gguf"
        )
    if args.mtp and args.temperature > 0.0:
        raise SystemExit(
            "--mtp is greedy-only (its accept rule compares drafted tokens "
            "against greedy samples, llama.cpp's server path); "
            "pass --temperature 0"
        )

    # Env knobs MUST be applied before model construction (the engine and
    # the native extension read them at init/dispatch time).
    if args.cache_mb is not None:
        os.environ["FTV_EXP_CACHE_MB"] = str(args.cache_mb)
    if args.host_pack_mb is not None:
        os.environ["FTV_HOST_PACK_MB"] = str(args.host_pack_mb)
    if args.dequant_cache_mb is not None:
        os.environ["FTV_GGUF_DEQUANT_CACHE_MB"] = str(args.dequant_cache_mb)
    if args.gpu_nvfp4 is not None:
        os.environ["FTV_NVFP4_GPU"] = str(args.gpu_nvfp4)

    if args.gguf is not None:
        gguf_path = pathlib.Path(args.gguf)
        if not gguf_path.is_file():
            raise SystemExit(f"no GGUF file at {gguf_path}")
        from freetoken.models.gguf.reader import load_gguf_metadata
        from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

        tokenizer = load_gguf_tokenizer(str(gguf_path))
        from .gguf_provider import gguf_config

        config = gguf_config(gguf_path)
        meta = load_gguf_metadata(str(gguf_path))
        eos = int(meta.get("tokenizer.ggml.eos_token_id", engine.EOS_IDS[0]))
        print(f"gguf: {gguf_path}\n"
              f"layers={config.num_hidden_layers} "
              f"experts={config.num_experts} top{config.num_experts_per_tok} "
              f"eos={eos}", flush=True)
        print("loading + pinning weights...", flush=True)
        started = time.time()
        model = engine.load_engine(
            gguf=gguf_path, max_len=args.max_len, kv_dtype=args.kv_dtype
        )
        source = str(gguf_path)
        nvfp4 = False
    else:
        model_dir = pathlib.Path(args.model)
        if not (model_dir / "config.json").exists():
            raise SystemExit(f"no config.json at {model_dir}")
        nvfp4 = engine.is_nvfp4_checkpoint(model_dir)
        eos = engine.EOS_IDS[0]

        tokenizer = load_tokenizer(model_dir)
        config = load_qwen35_config(model_dir)
        print(f"model: {model_dir}\n"
              f"nvfp4={nvfp4} layers={config.num_hidden_layers} "
              f"experts={config.num_experts} top{config.num_experts_per_tok}",
              flush=True)
        print("loading + pinning weights...", flush=True)
        started = time.time()
        model = engine.load_engine(
            model_dir, nvfp4=nvfp4, max_len=args.max_len, kv_dtype=args.kv_dtype
        )
        source = str(model_dir)
    print(f"resident ready in {time.time() - started:.1f}s "
          f"pinned={len(model.pinned)} mats max_len={args.max_len} "
          f"kv={args.kv_dtype}", flush=True)

    if args.smoke:
        smoke(model, tokenizer)
        return

    if args.raw_prompt:
        ids = encode_raw(tokenizer, args.prompt)
    else:
        ids = build_prompt(tokenizer, args.prompt, thinking=args.thinking)
    print(f"prefill {len(ids)} tokens...", flush=True)
    started = time.time()
    kwargs = dict(
        max_new=args.max_tokens,
        eos=eos,
        prefill_chunk=args.prefill_chunk,
        log_progress=len(ids) >= args.prefill_chunk,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    # ResidentGGUF inherits ResidentNVFP4's batched decode (the K-quant
    # record overrides live in gguf_batched.py); FTV_GGUF_BATCHED=0 makes
    # its generate() downgrade to the sequential path with a warning.
    if isinstance(model, engine.ResidentNVFP4):
        kwargs["batched"] = args.batch
    mtp_json = None
    if args.mtp:
        from .mtp import generate_mtp

        try:
            out = generate_mtp(
                model, ids,
                max_new=args.max_tokens, eos=eos,
                prefill_chunk=args.prefill_chunk,
                log_progress=len(ids) >= args.prefill_chunk,
                n_max=args.mtp_n_max, n_min=args.mtp_n_min,
                p_min=args.mtp_p_min,
            )
        except ValueError as exc:  # no nextn block in this GGUF
            raise SystemExit(f"--mtp unavailable: {exc}") from exc
        mtp_json = dict(model.mtp.stats)
        mtp_json.update(
            n_max=args.mtp_n_max, n_min=args.mtp_n_min, p_min=args.mtp_p_min,
            tokens=len(out),
        )
    else:
        out = model.generate(ids, **kwargs)
    wall = time.time() - started
    text = tokenizer.decode(out)
    n = max(len(out), 1)
    print(f"--- ({wall:.1f}s, {len(out)} tokens, {wall / n:.2f}s/tok, "
          f"{n / wall:.2f} tok/s) ---")
    if mtp_json:
        drafted, accepted = mtp_json["drafted"], mtp_json["accepted"]
        rate = accepted / drafted if drafted else 0.0
        passes = (mtp_json["rounds"] + mtp_json["rollbacks"]
                  + (drafted + mtp_json["rounds"]) / config.num_hidden_layers)
        print(f"mtp: drafted {drafted:.0f} accepted {accepted:.0f} "
              f"rate {rate:.2f} rounds {mtp_json['rounds']:.0f} "
              f"rollbacks {mtp_json['rollbacks']:.0f} "
              f"passes/tok {passes / n:.3f} "
              f"(greedy baseline 1.000, "
              f"{(1 - passes / n) * 100:.1f}% fewer full passes)")
        print(f"mtp ms: draft {mtp_json['ms_draft']:.0f} "
              f"verify {mtp_json['ms_verify']:.0f} "
              f"catchup {mtp_json['ms_catchup']:.0f} "
              f"rollback {mtp_json['ms_rollback']:.0f}")
    print(text)
    print(f"token_ids: {out}")
    if args.emit_json:
        emit_result(args.emit_json, args.prompt, ids, out, text, args, source,
                    extra={"mtp": mtp_json} if mtp_json else None)
    print("GGUF_CHAT_OK" if args.gguf else ("RESSL_CHAT_OK" if nvfp4
                                            else "QWEN35_CHAT_OK"))


def emit_result(path_str: str, prompt: str, prompt_ids: List[int],
                token_ids: List[int], text: str, args, source: str,
                *, extra: dict | None = None) -> None:
    """Merge {prompt: {text, token_ids, ...}} into the JSON at ``path_str``.

    Merging keeps one file for several reference prompts run separately.
    ``extra`` adds run-specific fields (the MTP accept counters).
    """
    import json

    path = pathlib.Path(path_str)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
    data[prompt] = {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "text": text,
        "token_ids": token_ids,
        "greedy": args.temperature == 0.0,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_new": args.max_tokens,
        "raw_prompt": bool(args.raw_prompt),
        "source": source,
    }
    if extra:
        data[prompt].update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"emit-json -> {path}")


if __name__ == "__main__":
    main()
