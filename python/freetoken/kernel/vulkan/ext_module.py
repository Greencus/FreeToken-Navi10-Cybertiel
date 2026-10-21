"""``freetoken.kernel.vulkan.ext_module`` — the raw pybind module (alias).

Identical to ``freetoken.kernel.vulkan.ext`` (which see): both names resolve
to the same native ``freetoken_vulkan_ext`` pybind module.
"""
from __future__ import annotations

import sys as _sys

from freetoken.kernel.vulkan import require_ext as _require_ext

_real = _require_ext()

for _name in dir(_real):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_real, _name)

_sys.modules[__name__] = _real
