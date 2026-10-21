"""``freetoken.kernel.vulkan.ext`` — the raw pybind module (stable alias).

Importing this submodule loads the native ``freetoken_vulkan_ext`` extension
and substitutes itself in ``sys.modules`` with the extension module, so every
import style yields the same raw pybind module:

    import freetoken.kernel.vulkan.ext as ext
    from freetoken.kernel.vulkan import ext
    import importlib; importlib.import_module("freetoken.kernel.vulkan.ext")

``freetoken.kernel.vulkan.ext_module`` is an identical alias.
"""
from __future__ import annotations

import sys as _sys

from freetoken.kernel.vulkan import require_ext as _require_ext

_real = _require_ext()

# Re-export the public op names so attribute access keeps working even if the
# import machinery hands back this (pre-substitution) shim module object.
for _name in dir(_real):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_real, _name)

# Substitute: CPython re-reads sys.modules after exec_module, so the importer
# (and the parent-package attribute set) resolve to the raw pybind module.
_sys.modules[__name__] = _real
