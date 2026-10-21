"""Shared fixtures for tests/vulkan.

Skip rules:
- ``vulkan_ext`` skips when the native extension or a usable Vulkan device
  is unavailable (ext resolution fails, or a trivial allocate/free raises).
- ``nvfp4_model_dir`` / ``qwen35_model_dir`` / ``gptoss_model_dir`` skip
  unless the matching env var points at a real checkpoint, so the default
  pytest run never touches a 23 GB download.
- ``gguf_model_path`` skips unless ``FREETOKEN_TEST_MODEL_GGUF`` points at an
  existing GGUF file (the CyberTiel checkpoint; it may be a symlink or a
  HF cache blob without a .gguf suffix, so any existing file counts).
- ``resident_model`` prefers the GGUF and falls back to the NVFP4 dir, so
  weight-source-agnostic gates (GDN shader, layer parity) run against
  whichever checkpoint the operator shipped.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from freetoken.models.qwen3_5_moe.vulkan._runtime import ext_or_none, runtime_info


def _probe_ext():
    ext = ext_or_none()
    if ext is None:
        return None, f"extension unresolved: {runtime_info()}"
    try:
        handle = ext.allocate_resident(256)
        ext.free_resident(handle)
    except Exception as exc:  # no device / driver failure
        return None, f"allocate/free failed: {type(exc).__name__}: {exc}"
    return ext, None


@pytest.fixture(scope="session")
def vulkan_ext():
    ext, err = _probe_ext()
    if ext is None:
        pytest.skip(f"no usable Vulkan device ({err})")
    return ext


def _model_dir_from_env(var: str) -> pathlib.Path:
    raw = os.environ.get(var, "").strip()
    if not raw:
        pytest.skip(f"set {var} to a checkpoint directory to enable this gate")
    path = pathlib.Path(raw)
    if not (path / "config.json").exists():
        pytest.skip(f"{var}={path} has no config.json")
    return path


@pytest.fixture(scope="session")
def nvfp4_model_dir() -> pathlib.Path:
    return _model_dir_from_env("FREETOKEN_TEST_MODEL")


@pytest.fixture(scope="session")
def qwen35_model_dir() -> pathlib.Path:
    return _model_dir_from_env("FREETOKEN_TEST_MODEL_QWEN35")


@pytest.fixture(scope="session")
def gptoss_model_dir() -> pathlib.Path:
    return _model_dir_from_env("FREETOKEN_TEST_MODEL_GPT_OSS")


@pytest.fixture(scope="session")
def gguf_model_path() -> pathlib.Path:
    raw = os.environ.get("FREETOKEN_TEST_MODEL_GGUF", "").strip()
    if not raw:
        pytest.skip(
            "set FREETOKEN_TEST_MODEL_GGUF to the CyberTiel GGUF file to "
            "enable this gate"
        )
    path = pathlib.Path(raw)
    if not path.is_file():
        pytest.skip(f"FREETOKEN_TEST_MODEL_GGUF={path} is not an existing file")
    return path


@pytest.fixture(scope="session")
def nvfp4_model(vulkan_ext, nvfp4_model_dir):
    """One shared ResidentNVFP4 for the model-gated gates: constructing it
    pins ~1-2 GB of VRAM, so tests share the instance and each resets."""
    from freetoken.models.qwen3_5_moe.vulkan import ResidentNVFP4

    model = ResidentNVFP4(nvfp4_model_dir, vulkan_ext)
    return model


@pytest.fixture(scope="session")
def gguf_model(vulkan_ext, gguf_model_path):
    """One shared ResidentGGUF (host-dequant K-quant path); same VRAM-sharing
    rationale as nvfp4_model."""
    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF

    model = ResidentGGUF(gguf_model_path, vulkan_ext)
    return model


@pytest.fixture(scope="session")
def resident_model(vulkan_ext):
    """Checkpoint-agnostic resident decoder: GGUF preferred, NVFP4 fallback.

    Lets weight-source-neutral gates (GDN shader vs host, layer parity) run
    against whichever checkpoint is on disk; both flavors expose config,
    _dense and the host references the gates compare against. Reads the env
    directly (not via gguf_model_path/nvfp4_model_dir) so a missing GGUF
    falls through to the NVFP4 dir instead of skipping at the fixture.
    """
    from freetoken.models.qwen3_5_moe.vulkan import ResidentGGUF, ResidentNVFP4

    gguf = os.environ.get("FREETOKEN_TEST_MODEL_GGUF", "").strip()
    if gguf and pathlib.Path(gguf).is_file():
        return ResidentGGUF(gguf, vulkan_ext)
    nv = _model_dir_from_env_optional("FREETOKEN_TEST_MODEL")
    if nv is not None:
        return ResidentNVFP4(nv, vulkan_ext)
    pytest.skip(
        "no checkpoint: set FREETOKEN_TEST_MODEL_GGUF (preferred) or "
        "FREETOKEN_TEST_MODEL"
    )


def _model_dir_from_env_optional(var: str) -> pathlib.Path | None:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    path = pathlib.Path(raw)
    if not (path / "config.json").exists():
        return None
    return path
