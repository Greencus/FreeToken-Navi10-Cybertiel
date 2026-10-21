"""Top-level Vulkan inference engine for the Qwen3.5-MoE family.

This module is the package's model/engine API: it re-exports the two
resident decoders (``ResidentQwen35`` for TorchAO-affine checkpoints,
``ResidentNVFP4`` for the Ressl ModelOpt NVFP4 checkpoint) and provides
``load_engine``/``generate`` one-call helpers on top of them.

Port lineage note: the fork's ``python/freetoken_vulkan.py`` - the file
this module replaces - is the early subprocess shim (one ``*.exe`` spawn
per kernel, Windows-only paths, unusable for inference). The fork's real
top-level engine API lived in its chat CLIs, which wire the extension to
``ResidentNVFP4``/``ResidentQwen35`` and call ``generate``; that behavior
is what ``load_engine`` + ``cli.py`` reproduce here.

Usage::

    from freetoken.models.qwen3_5_moe.vulkan import engine

    model = engine.load_engine(model_dir)          # NVFP4 auto-detected
    out_ids = engine.generate(model, prompt_ids, max_new=16, temperature=0.6)
"""
from __future__ import annotations

import pathlib
from typing import Any, List

from ._runtime import ext_or_none, get_ext, runtime_info, shaders_dir
from .config import Qwen35MoeConfig, load_qwen35_config
from .resident import ResidentQwen35
from .nvfp4_resident import ResidentNVFP4
from .gguf_resident import ResidentGGUF
# Drop-in subclass: pins the static expert hot set at init (env-gated,
# FTV_GGUF_HOTSET=0 restores plain ResidentGGUF behavior exactly).
from .hotset_resident import HotsetResidentGGUF

# Fork defaults (chat_ressl.py / chat_qwen35.py): EOS ids of the
# Qwen3.5 tokenizer (248046 eos, 248044 pad/bos).
EOS_IDS = (248046, 248044)


def is_nvfp4_checkpoint(model_dir: str | pathlib.Path) -> bool:
    """NVFP4 (Ressl ModelOpt) checkpoints carry hf_quant_config.json and no
    TorchAO quantization_config; everything else goes through the affine path."""
    model_dir = pathlib.Path(model_dir)
    return (model_dir / "hf_quant_config.json").exists()


def load_engine(
    model_dir: str | pathlib.Path | None = None,
    *,
    gguf: str | pathlib.Path | None = None,
    nvfp4: bool | None = None,
    max_len: int = 4096,
    kv_dtype: str = "fp32",
    ext: Any = None,
) -> ResidentQwen35 | ResidentNVFP4 | ResidentGGUF:
    """Construct the resident decoder for ``model_dir`` (or a single GGUF).

    ``gguf=<file>`` builds a ``HotsetResidentGGUF`` (``ResidentGGUF`` with
    the static expert hot-set pin; FTV_GGUF_HOTSET=0 disables the pin and
    it behaves exactly like ``ResidentGGUF``) for ``qwen35moe`` GGUFs,
    e.g. CyberTiel); ``model_dir=None`` is then the
    source of config/tokenizer only if gguf is also None (it is unused for
    GGUF construction). ``nvfp4=None`` auto-detects the checkpoint flavor of
    ``model_dir``; pass True/False to force a path. ``ext=None`` resolves the
    pybind module lazily through ``_runtime``.
    """
    if gguf is not None:
        return HotsetResidentGGUF(gguf, ext, max_len=max_len, kv_dtype=kv_dtype)
    if model_dir is None:
        raise ValueError("load_engine needs model_dir or gguf=")
    if nvfp4 is None:
        nvfp4 = is_nvfp4_checkpoint(model_dir)
    cls = ResidentNVFP4 if nvfp4 else ResidentQwen35
    return cls(model_dir, ext, max_len=max_len, kv_dtype=kv_dtype)


def generate(
    model: ResidentQwen35 | ResidentNVFP4 | ResidentGGUF,
    prompt_ids: List[int],
    **kwargs: Any,
) -> List[int]:
    """Prefill + generate on ``model``; kwargs mirror ``generate()`` on the
    decoders (max_new, eos, prefill_chunk, temperature, top_k, top_p,
    seed, ...). ``batched`` is accepted for the NVFP4 batched decode path
    (``ResidentGGUF`` rejects it: its decode path is the sequential one).
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if isinstance(model, ResidentNVFP4):
        allowed = ("max_new", "eos", "prefill_chunk", "log_progress",
                   "temperature", "top_k", "top_p", "seed", "batched")
    else:
        allowed = ("max_new", "eos", "prefill_chunk", "log_progress",
                   "temperature", "top_k", "top_p", "seed")
    unknown = sorted(set(kwargs) - set(allowed))
    if unknown:
        raise TypeError(f"unsupported generate kwargs for {type(model).__name__}: {unknown}")
    return model.generate(prompt_ids, **kwargs)


__all__ = [
    "EOS_IDS",
    "Qwen35MoeConfig",
    "ResidentGGUF",
    "ResidentNVFP4",
    "ResidentQwen35",
    "ext_or_none",
    "generate",
    "get_ext",
    "is_nvfp4_checkpoint",
    "load_engine",
    "load_qwen35_config",
    "runtime_info",
    "shaders_dir",
]
