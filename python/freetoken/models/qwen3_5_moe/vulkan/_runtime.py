"""Resolve the Vulkan pybind extension and the compiled shader directory.

Every module in this package reaches the native extension through
``get_ext()`` (lazy) and the SPIR-V directory through ``shaders_dir()``.
Nothing here runs at import time, so ``import freetoken.models.qwen3_5_moe.
vulkan`` never fails when Vulkan or the extension is missing.

Resolution order for the extension:

1. ``from freetoken.kernel.vulkan import ext, shaders_dir`` - the
   production path (the new backend package this engine will ship with).
2. Fallback: ``import freetoken_vulkan_ext`` after appending
   ``$FREETOKEN_VULKAN_EXT_DIR`` (default: the staged ``.build`` dir) to
   ``sys.path`` - the standalone staged extension used for interim testing.

``FREETOKEN_VULKAN_SHADERS`` is honored when set (the C++ side reads it
via getenv); when unset it is populated with the resolved shader dir
before the extension is first imported, so the native side never falls
back to its Windows placeholder path.

``torch`` is imported before the extension because the extension links
against ``libc10.so`` and only resolves it when torch has already been
loaded into the process.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

# Staged build (fallback): freetoken_vulkan_ext.so + sibling shaders/.
_DEFAULT_EXT_DIR = pathlib.Path(
    os.environ.get(
        "FREETOKEN_VULKAN_EXT_DIR",
        "/home/cus/working/FreeTokenVulkan5700xt/.build",
    )
)

_ext: Any | None = None
_ext_error: str | None = None
_preferred_error: str | None = None


def _preferred_shaders_dir() -> pathlib.Path | None:
    """Shader dir published by freetoken.kernel.vulkan, if that package is importable."""
    try:
        from freetoken.kernel.vulkan import shaders_dir as published  # type: ignore
    except Exception:
        return None
    try:
        if callable(published):
            published = published()
        return pathlib.Path(published)
    except Exception:
        return None


def shaders_dir() -> pathlib.Path:
    """Directory holding the compiled ``*.spv`` kernels.

    Honors ``FREETOKEN_VULKAN_SHADERS`` when set; otherwise resolves from
    ``freetoken.kernel.vulkan`` or the staged build dir.
    """
    env = os.environ.get("FREETOKEN_VULKAN_SHADERS")
    if env:
        return pathlib.Path(env)
    published = _preferred_shaders_dir()
    if published is not None:
        return published
    return _DEFAULT_EXT_DIR / "shaders"


def _ensure_shader_env() -> pathlib.Path:
    resolved = shaders_dir()
    # The native side getenv()s this per shader-path lookup; populate it
    # only when the user has not set it (env always wins above).
    os.environ.setdefault("FREETOKEN_VULKAN_SHADERS", str(resolved))
    return resolved


def get_ext() -> Any:
    """Return the pybind extension module, importing it on first use.

    Raises RuntimeError (chaining the underlying error) when neither the
    package path nor the staged fallback can provide the extension.
    """
    global _ext, _ext_error, _preferred_error
    if _ext is not None:
        return _ext
    _ensure_shader_env()

    # Preferred production path.
    try:
        from freetoken.kernel.vulkan import ext as pkg_ext  # type: ignore

        # torch first: the extension links libc10.so, which only resolves
        # once torch has been loaded into the process.
        import torch  # noqa: F401

        _ext = pkg_ext
        return _ext
    except Exception as exc:  # module missing OR half-built package
        _preferred_error = f"{type(exc).__name__}: {exc}"

    # Staged fallback: FREETOKEN_VULKAN_EXT_DIR on sys.path.
    try:
        ext_dir = pathlib.Path(
            os.environ.get("FREETOKEN_VULKAN_EXT_DIR", str(_DEFAULT_EXT_DIR))
        )
        ext_dir_str = str(ext_dir)
        if ext_dir_str not in sys.path:
            sys.path.insert(0, ext_dir_str)
        import torch  # noqa: F401

        import freetoken_vulkan_ext as staged_ext  # type: ignore

        _ext = staged_ext
        return _ext
    except Exception as exc:
        _ext_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "Vulkan extension unavailable: preferred path "
            f"(freetoken.kernel.vulkan) failed with [{_preferred_error}]; "
            f"staged fallback ({_DEFAULT_EXT_DIR}) failed with [{_ext_error}]"
        ) from exc


def ext_or_none() -> Any | None:
    """The extension module, or None when it cannot be resolved.

    Test helper: lets callers pytest.skip() instead of failing when the
    machine has no Vulkan device / no built extension.
    """
    try:
        return get_ext()
    except Exception:
        return None


def runtime_info() -> dict:
    """Diagnostics: which resolution path was used and why not the others."""
    return {
        "ext_loaded": _ext is not None,
        "preferred_error": _preferred_error,
        "fallback_error": _ext_error,
        "shaders_dir": str(shaders_dir()),
        "shader_env": os.environ.get("FREETOKEN_VULKAN_SHADERS"),
        "ext_dir_env": os.environ.get(
            "FREETOKEN_VULKAN_EXT_DIR", str(_DEFAULT_EXT_DIR)
        ),
    }
