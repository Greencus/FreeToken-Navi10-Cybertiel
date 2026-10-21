"""Build / locate the native ``freetoken_vulkan_ext`` pybind extension.

Ported from the Navi10 Vulkan fork (``freetoken-vulkan/python``). The C++ is
unchanged (``ext_module.cpp`` + sibling ``vk_util.hpp``; the quoted include
already resolves inside this package). Three load strategies, in order:

  (a) an already-built in-tree shared object matching this interpreter —
      searched next to this package and in the torch cpp_extension build dir;
  (b) JIT build via ``torch.utils.cpp_extension.load``;
  (c) direct ``c++`` invocation (the known-good manual recipe from the fork,
      with include/lib/python paths derived from the RUNNING interpreter).

The module's pybind name stays ``freetoken_vulkan_ext`` (TORCH_EXTENSION_NAME)
— the package re-exports it under the stable names
``freetoken.kernel.vulkan.ext`` / ``.ext_module``.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from types import ModuleType

PKG_DIR = pathlib.Path(__file__).resolve().parent
EXT_NAME = "freetoken_vulkan_ext"
SOURCE = PKG_DIR / "ext_module.cpp"

# Build scratch (object files). /tmp/opencode is the sanctioned temp area on
# this machine; fall back to the system temp dir elsewhere.
_WORK_ROOT = pathlib.Path("/tmp/opencode")


def _work_dir() -> pathlib.Path:
    try:
        _WORK_ROOT.mkdir(parents=True, exist_ok=True)
        return _WORK_ROOT
    except OSError:
        return pathlib.Path(tempfile.gettempdir())


def _soabi() -> str:
    return sysconfig.get_config_var("SOABI") or ""


def _so_names() -> list[str]:
    """SOABI-tagged first (matches this interpreter), then untagged."""
    soabi = _soabi()
    names = []
    if soabi:
        names.append(f"{EXT_NAME}.{soabi}.so")
    names.append(f"{EXT_NAME}.so")
    return names


def _torch_build_dir() -> pathlib.Path | None:
    try:
        from torch.utils.cpp_extension import _get_build_directory

        return pathlib.Path(_get_build_directory(EXT_NAME, verbose=False))
    except Exception:  # noqa: BLE001 - private API, optional location
        return None


def prebuilt_candidates() -> list[pathlib.Path]:
    """Existing .so files to try, best match for this interpreter first."""
    found: list[pathlib.Path] = []
    for name in _so_names():
        found.append(PKG_DIR / name)
    tbd = _torch_build_dir()
    if tbd is not None:
        for name in _so_names():
            found.append(tbd / name)
    return [p for p in found if p.is_file()]


def _import_so(path: pathlib.Path) -> ModuleType:
    """dlopen a prebuilt extension whose init symbol is PyInit_freetoken_vulkan_ext."""
    spec = importlib.util.spec_from_file_location(EXT_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    prev = sys.modules.get(EXT_NAME)
    sys.modules[EXT_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(EXT_NAME, None)
        if prev is not None:
            sys.modules[EXT_NAME] = prev
        raise
    return module


def _jit_build() -> ModuleType:
    """Strategy (b): torch.utils.cpp_extension.load."""
    import torch
    from torch.utils.cpp_extension import load

    # torch.utils.cpp_extension.load() has no `libraries=` kwarg (that only
    # exists on the CppExtension/setup path), so -lvulkan comes via ldflags —
    # equivalent flags to the fork's validated recipe.
    return load(
        name=EXT_NAME,
        sources=[str(SOURCE)],
        extra_include_paths=[str(PKG_DIR)],
        extra_cflags=["-std=c++20", "-O2"],
        extra_ldflags=["-lvulkan"],
        verbose=os.environ.get("FREETOKEN_VULKAN_BUILD_VERBOSE", "") != "",
    )


def _manual_build(out_dir: pathlib.Path) -> pathlib.Path:
    """Strategy (c): direct c++ invocation, flags equivalent to the fork's
    validated Linux recipe (python/build_ext.sh), with all paths derived from
    the RUNNING interpreter instead of hardcoded."""
    import torch
    from torch.utils.cpp_extension import include_paths as torch_include_paths

    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or "c++"
    py_inc = sysconfig.get_paths()["include"]
    torch_inc = list(torch_include_paths())
    torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
    abi = int(bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True)))

    out_dir.mkdir(parents=True, exist_ok=True)
    soabi = _soabi()
    out_so = out_dir / (f"{EXT_NAME}.{soabi}.so" if soabi else f"{EXT_NAME}.so")

    work = _work_dir() / "freetoken-vulkan-ext-build"
    work.mkdir(parents=True, exist_ok=True)
    obj = work / "ext.o"

    compile_cmd = [
        cxx, "-std=c++20", "-O2", "-fPIC",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        f"-DTORCH_EXTENSION_NAME={EXT_NAME}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-I/usr/include",
        f"-I{PKG_DIR}",
        *(f"-I{inc}" for inc in torch_inc),
        f"-I{py_inc}",
        "-c", str(SOURCE),
        "-o", str(obj),
    ]
    link_cmd = [
        cxx, "-shared",
        "-o", str(out_so),
        str(obj),
        "-lvulkan",
        f"-L{torch_lib}",
        f"-Wl,-rpath,{torch_lib}",
        "-lc10", "-ltorch_cpu", "-ltorch", "-ltorch_python",
    ]

    for cmd in (compile_cmd, link_cmd):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "manual extension build failed "
                f"(exit {proc.returncode}):\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )
    return out_so


def load_extension() -> ModuleType:
    """Locate or build the extension module (strategies a -> b -> c)."""
    errors: list[str] = []

    # (a) already-built shared object.
    for path in prebuilt_candidates():
        try:
            return _import_so(path)
        except BaseException as exc:  # noqa: BLE001 - ABI mismatch etc.
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    # (b) JIT build via torch.
    try:
        return _jit_build()
    except BaseException as exc:  # noqa: BLE001
        errors.append(f"torch cpp_extension.load: {type(exc).__name__}: {exc}")

    # (c) direct c++ fallback, built in-tree so (a) finds it next time.
    try:
        so = _manual_build(PKG_DIR)
        return _import_so(so)
    except BaseException as exc:  # noqa: BLE001
        errors.append(f"manual c++ build: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "could not build or load the freetoken_vulkan_ext Vulkan extension; "
        "tried in-tree .so, torch cpp_extension.load, and a direct c++ build. "
        "Details:\n  - " + "\n  - ".join(errors)
    )


def build_extension(
    out_dir: pathlib.Path | None = None, *, force: bool = False
) -> pathlib.Path:
    """Build the extension into ``out_dir`` (default: this package) and
    return the .so path. Reuses an existing matching build unless ``force``.

    Used by ``python -m freetoken.kernel.vulkan.build``. Prefers the fast
    direct-c++ recipe; falls back to torch's JIT builder and copies its
    output into ``out_dir`` so the next in-tree load hits strategy (a).
    """
    out_dir = pathlib.Path(out_dir) if out_dir is not None else PKG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        for name in _so_names():
            cand = out_dir / name
            if cand.is_file():
                return cand

    try:
        return _manual_build(out_dir)
    except Exception as manual_exc:  # noqa: BLE001
        try:
            module = _jit_build()
        except Exception as jit_exc:  # noqa: BLE001
            raise RuntimeError(
                "extension build failed via both paths:\n"
                f"  manual c++: {manual_exc}\n"
                f"  torch JIT: {jit_exc}"
            ) from jit_exc
        built = pathlib.Path(getattr(module, "__file__", "") or "")
        if built.is_file() and built.parent.resolve() != out_dir.resolve():
            target = out_dir / built.name
            shutil.copy2(built, target)
            return target
        if built.is_file():
            return built
        raise RuntimeError(
            f"torch JIT build produced no loadable artifact (got {built!r})"
        ) from manual_exc
