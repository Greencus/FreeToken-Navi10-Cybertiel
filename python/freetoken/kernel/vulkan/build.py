"""CLI: build the Vulkan extension and compile all shaders.

Usage:
    python -m freetoken.kernel.vulkan.build [--out DIR] [--force]

Builds ``freetoken_vulkan_ext`` into ``--out`` (default: this package dir, so
the normal ``load_ext()`` path finds it) and compiles every ``shaders/*.comp``
into the shader cache (``$FREETOKEN_VULKAN_SHADER_CACHE`` or
``~/.cache/freetoken/vulkan/shaders``). Prints the resulting paths.
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m freetoken.kernel.vulkan.build",
        description="Build the freetoken_vulkan_ext extension and compile "
        "the Vulkan compute shaders.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="directory for the built .so (default: the package dir)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild the extension and recompile all shaders even if "
        "up to date",
    )
    args = parser.parse_args(argv)

    # Import via the package path when possible; fall back to direct file
    # loading so the CLI also works from a bare checkout without install.
    try:
        from freetoken.kernel.vulkan import _build, compile_shaders
    except Exception:  # noqa: BLE001 - standalone fallback
        pkg_dir = pathlib.Path(__file__).resolve().parent
        sys.path.insert(0, str(pkg_dir.parents[2]))  # .../python
        from freetoken.kernel.vulkan import _build, compile_shaders

    so = _build.build_extension(out_dir=args.out, force=args.force)
    spv_dir = compile_shaders(force=args.force)
    n_spv = len(list(spv_dir.glob("*.spv")))

    print(f"extension: {so}")
    print(f"shaders:   {spv_dir}  ({n_spv} .spv)")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
