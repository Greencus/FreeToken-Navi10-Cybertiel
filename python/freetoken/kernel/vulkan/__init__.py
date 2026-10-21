"""FreeToken Vulkan compute backend (native runtime, ported from Navi10).

This package is a faithful port of the from-scratch Vulkan backend from the
``freetoken-vulkan`` fork: the unmodified pybind11 extension
(``ext_module.cpp`` + ``vk_util.hpp``, ~72 ops: rmsnorm, resident linears,
batched MoE dispatch, flash attention, GPU router, ...) plus the GLSL compute
shaders it loads. It is the Vulkan sibling of ``freetoken.kernel.triton`` /
``freetoken.kernel.ggml_cpu``.

Public API
----------
``available()``            True when a Vulkan device + the extension are usable.
``ext`` / ``ext_module``   the raw pybind module (lazy; also importable as
                           ``freetoken.kernel.vulkan.ext`` /
                           ``freetoken.kernel.vulkan.ext_module``).
``load_ext()``             load (or JIT-build) the extension now.
``require_ext()``          same, but raises an actionable RuntimeError.
``shaders_dir()``          directory holding the compiled ``.spv`` files
                           (compiles them lazily on first use).
``compile_shaders(force)`` compile every ``shaders/*.comp`` with glslc.
``device_info()``          name / driver / API version / VRAM / shader dir.

Environment
-----------
``FREETOKEN_VULKAN_SHADERS``     directory with compiled ``.spv`` files; this
                                package sets it to the compiled cache on
                                ``load_ext()`` if the user has not set it
                                (the C++ side reads it at pipeline-load time).
``FREETOKEN_VULKAN_SHADER_CACHE``where ``compile_shaders()`` writes its output
                                (default ``~/.cache/freetoken/vulkan/shaders``).
``FREETOKEN_VULKAN_BUILD_VERBOSE``  set to a non-empty string for chatty JIT
                                builds.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
from typing import Any

from . import _build as _build_mod
from . import _probe

__all__ = [
    "available",
    "compile_shaders",
    "device_info",
    "ext",
    "ext_module",
    "load_ext",
    "require_ext",
    "shaders_dir",
]

_PKG_DIR = pathlib.Path(__file__).resolve().parent
_PKG_NAME = __name__  # "freetoken.kernel.vulkan"
_ALIAS_NAMES = (_PKG_NAME + ".ext", _PKG_NAME + ".ext_module")

_EXT: Any = None  # loaded pybind module, cached on success


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------
def _find_glslc() -> str:
    """Locate glslc (shaderc); raise a clear error when missing."""
    glslc = shutil.which("glslc")
    if glslc:
        return glslc
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for sub in ("bin", "Bin"):
            cand = pathlib.Path(sdk) / sub / "glslc"
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    raise RuntimeError(
        "glslc (the GLSL→SPIR-V compiler from shaderc) was not found on PATH "
        "and VULKAN_SDK is not set. Install it (e.g. `pacman -S shaderc`, "
        "`apt install glslc`, or the LunarG Vulkan SDK) and retry; you can "
        "also point VULKAN_SDK at an SDK containing bin/glslc."
    )


def _shader_cache_dir() -> pathlib.Path:
    env = os.environ.get("FREETOKEN_VULKAN_SHADER_CACHE")
    if env:
        return pathlib.Path(env).expanduser()
    return pathlib.Path.home() / ".cache" / "freetoken" / "vulkan" / "shaders"


def compile_shaders(force: bool = False) -> pathlib.Path:
    """Compile every ``shaders/*.comp`` in this package to ``.spv``.

    Output goes to ``$FREETOKEN_VULKAN_SHADER_CACHE`` (or
    ``~/.cache/freetoken/vulkan/shaders``). Files whose ``.spv`` is newer
    than the ``.comp`` are skipped unless ``force``. Returns the cache dir.
    Raises RuntimeError when glslc is missing or a shader fails to compile.
    """
    glslc = _find_glslc()  # clear error before doing any work
    src_dir = _PKG_DIR / "shaders"
    comps = sorted(src_dir.glob("*.comp"))
    if not comps:
        raise RuntimeError(f"no .comp shaders found in {src_dir}")

    cache = _shader_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    import subprocess

    for comp in comps:
        spv = cache / (comp.name + ".spv")
        if (
            not force
            and spv.is_file()
            and spv.stat().st_mtime >= comp.stat().st_mtime
        ):
            continue
        tmp = spv.with_name(spv.name + ".tmp")
        proc = subprocess.run(
            [
                glslc,
                "--target-env=vulkan1.2",
                "-fshader-stage=compute",
                str(comp),
                "-o",
                str(tmp),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"glslc failed for {comp.name} (exit {proc.returncode}):\n"
                f"{proc.stderr or proc.stdout}"
            )
        os.replace(tmp, spv)
    return cache


def shaders_dir() -> pathlib.Path:
    """Directory holding the compiled ``.spv`` files the C++ side loads.

    If ``$FREETOKEN_VULKAN_SHADERS`` is set (user override), that directory
    is used as-is. Otherwise this is the compiled cache, populated lazily on
    first call via ``compile_shaders()``.
    """
    env = os.environ.get("FREETOKEN_VULKAN_SHADERS")
    if env:
        return pathlib.Path(env).expanduser()
    return compile_shaders()


# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------
def load_ext() -> Any:
    """Load (or build) and return the raw pybind module.

    Ensures the shader cache exists first and defaults
    ``FREETOKEN_VULKAN_SHADERS`` to it when the user has not set it (the C++
    extension resolves ``.spv`` paths through that variable). The module is
    also exposed as ``freetoken.kernel.vulkan.ext`` /
    ``freetoken.kernel.vulkan.ext_module``.
    """
    global _EXT
    if _EXT is not None:
        return _EXT
    # Lazy shader compile + env defaulting (see module docstring).
    os.environ.setdefault("FREETOKEN_VULKAN_SHADERS", str(shaders_dir()))
    mod = _build_mod.load_extension()
    _EXT = mod
    # Stable alias names: both import styles resolve to the raw module.
    sys.modules.setdefault(_PKG_NAME + ".ext", mod)
    sys.modules.setdefault(_PKG_NAME + ".ext_module", mod)
    globals()["ext"] = mod
    globals()["ext_module"] = mod
    return mod


def require_ext() -> Any:
    """Like ``load_ext()`` but raises a clear, actionable error on failure."""
    try:
        return load_ext()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RuntimeError(
            f"failed to load the native Vulkan extension ({_PKG_NAME}): "
            f"{exc}\n"
            "Checklist:\n"
            "  * a Vulkan loader + driver and at least one usable GPU\n"
            "  * glslc (shaderc) available for shader compilation\n"
            "  * torch and a C++20 compiler (g++) for the JIT build\n"
            "Prebuild everything with: python -m freetoken.kernel.vulkan.build"
        ) from exc


def available() -> bool:
    """True when a usable Vulkan device AND the extension are present.

    Never raises. The device check uses the safe ctypes probe (the C++
    extension aborts the process on Vulkan errors, so it must not be used
    for probing); the extension check loads/imports it (building if needed).
    """
    try:
        if _probe.probe() is None:
            return False
        load_ext()
        return True
    except Exception:  # noqa: BLE001 - availability is best-effort
        return False


def _driver_string(p: dict[str, Any]) -> str | None:
    """Human driver string, e.g. 'radv (Mesa 26.2.2-arch1.1)'.

    Best effort: ``vulkaninfo --summary`` has the driver name/info (the
    ctypes probe only exposes the raw driverVersion int, which it returns as
    a fallback).
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            name = info = None
            for line in proc.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("driverName") and name is None:
                    name = stripped.split("=", 1)[1].strip()
                elif stripped.startswith("driverInfo") and info is None:
                    info = stripped.split("=", 1)[1].strip()
                if name and info:
                    break
            if name and info:
                return f"{name} ({info})"
            if name:
                return name
    except Exception:  # noqa: BLE001 - vulkaninfo is optional
        pass
    if p.get("driver_version"):
        return f"driverVersion {p['driver_version']}"
    return None


def device_info() -> dict[str, Any]:
    """Describe the primary Vulkan device and the shader dir actually used.

    Keys: name, driver, api_version, driver_version, vendor_id, device_id,
    vram_bytes, vram_gib, device_count, shaders_dir, shaders_compiled.
    """
    p = _probe.probe() or {}
    shaders = shaders_dir()  # lazily compiles when no user override
    compiled = len(list(shaders.glob("*.spv"))) if shaders.is_dir() else 0
    vram = p.get("vram_bytes")
    return {
        "name": p.get("name"),
        "driver": _driver_string(p),
        "api_version": p.get("api_version"),
        "driver_version": p.get("driver_version"),
        "vendor_id": p.get("vendor_id"),
        "device_id": p.get("device_id"),
        "device_count": p.get("device_count"),
        "vram_bytes": vram,
        "vram_gib": round(vram / (1 << 30), 2) if vram else None,
        "shaders_dir": str(shaders),
        "shaders_compiled": compiled,
    }


# ---------------------------------------------------------------------------
# Lazy attribute access: ``freetoken.kernel.vulkan.ext`` before/without the
# submodule shim files being imported, plus clear errors on failure.
# ---------------------------------------------------------------------------
def __getattr__(name: str) -> Any:
    if name in ("ext", "ext_module"):
        return require_ext()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
