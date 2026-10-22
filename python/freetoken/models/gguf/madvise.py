"""Page-cache advice for a memory-mapped GGUF - stop the kernel LRU thrash.

The CyberTiel GGUF is one 22.8 GB file of which the expert bank is 21.3 GB,
more than this box's free RAM, while the engine's expert LRU only ever reads
~14% of it. The kernel's page-cache LRU does not know that: every one-shot
expert fetch parks its pages in cache and pushes the hot set out, so the same
decode swings between 0.11 s/tok and 15+ s/tok depending on what other
tenants left in RAM. This module lets the ENGINE tell the kernel which byte
ranges of the one big file matter, per file range (the tensor index already
tracks offset/nbytes per tensor).

Two syscalls do the work, and the order matters:

- ``advise_willneed`` -> ``posix_fadvise(POSIX_FADV_WILLNEED)``: demand-
  prefetch a range the engine is about to fetch. Measured on this box
  (kernel 7.2.4-zen2, xfs, NVMe) it starts NO measurable readahead - the
  range is still read on demand at touch time - so today it is a cheap
  no-op; it stays because other kernels/filesystems do honor it (see
  ``docs/vulkan/page-cache.md``).
- ``advise_dontneed`` -> ``mmap.madvise(MADV_DONTNEED)`` on the READER'S
  mapping + ``posix_fadvise(POSIX_FADV_DONTNEED)``: actually FREE a cold
  range from the page cache. The two-step is mandatory: the GGUF stays
  mapped for the process lifetime, and the kernel skips fadvise-DONTNEED
  on pages that still have PTEs (``invalidate_mapping_pages`` ignores
  ``page_mapped`` pages - measured: fadvise alone drops nothing, 0 of
  128 MiB). ``MADV_DONTNEED`` on a read-only shared mapping only zaps our
  PTEs (the bytes stay in cache, so a live view of the range re-faults
  from RAM, never from disk, and cannot be corrupted); the following
  fadvise then finds ``mapcount == 0`` and frees the pages for real.

Everything is opt-in through ``FTV_GGUF_MADV`` (default 0 = off):

    0  off - the helpers below are no-ops (the default; nothing calls them
       yet, the engine lane wires its fetch/evict sites first)
    1  advise_willneed() fires on fetch
    2  advise_willneed() on fetch AND advise_dontneed() on evict

Fault accounting for the measurement lives here too: ``proc_counters()``
snapshots ``/proc/self/io`` (read_bytes - the ONLY honest disk-read signal
for an mmap'd file on Linux; page-cache refaults of mapped pages are counted
as MINOR faults, so maj_flt stays 0 even while the disk spins) and
``/proc/self/stat`` (min_flt/maj_flt as the brief asks, rchar for contrast).
"""
from __future__ import annotations

import mmap
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

from .reader import GgufTensorInfo

ENV_KNOB = "FTV_GGUF_MADV"
MODE_OFF = 0
MODE_WILLNEED = 1
MODE_DONTNEED = 2

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

# One open O_RDONLY fd + file size per path; fds stay open for the process
# lifetime (one per split file) so the per-fetch advice is one syscall, not
# open+stat+close.
_files_lock = threading.Lock()
_files: dict[str, tuple[int, int]] = {}          # path -> (fd, size)
_mappings: dict[str, mmap.mmap] = {}             # path -> the reader's mmap
_real: dict[str, str] = {}                       # path -> realpath (memoized)

_stats_lock = threading.Lock()
_stats: dict[str, float] = {
    "willneed_calls": 0, "dontneed_calls": 0, "bytes_willneed": 0,
    "bytes_dontneed": 0, "total_us": 0.0, "max_us": 0.0, "errors": 0,
}


def madvise_mode() -> int:
    """The active ``FTV_GGUF_MADV`` level (default 0; junk -> 0)."""
    raw = os.environ.get(ENV_KNOB)
    if not raw:
        return MODE_OFF
    try:
        level = int(raw.strip())
    except ValueError:
        return MODE_OFF
    return max(MODE_OFF, min(level, MODE_DONTNEED))


def _realpath(path: str) -> str:
    key = str(path)
    hit = _real.get(key)
    if hit is None:
        hit = os.path.realpath(key)
        _real[key] = hit
    return hit


def register_mapping(path: str, mapping: mmap.mmap) -> None:
    """Remember the reader's mmap for ``path`` so advise_dontneed can zap
    its PTEs. Called by ``reader._reader``; also fine for tests to call."""
    real = _realpath(path)
    with _files_lock:
        _mappings[real] = mapping
        filename = getattr(mapping, "filename", None)
        if isinstance(filename, str) and filename:
            _mappings[_realpath(filename)] = mapping


def _file(path: str) -> tuple[int, int] | None:
    """(fd, size) for ``path``, opened once; None when absent/unopenable."""
    real = _realpath(path)
    hit = _files.get(real)
    if hit is not None:
        return hit
    try:
        fd = os.open(real, os.O_RDONLY)
        size = os.fstat(fd).st_size
    except OSError:
        return None
    with _files_lock:
        hit = _files.setdefault(real, (fd, size))
    if hit[0] != fd:
        os.close(fd)  # lost a first-open race; keep the winner's fd
    return hit


def _aligned(path: str, offset: int, length: int) -> tuple[int, int, int] | None:
    """Page-aligned (fd, offset, length), clamped to the file; None if empty."""
    f = _file(path)
    if f is None:
        return None
    fd, size = f
    start = max(0, int(offset))
    end = start + max(0, int(length))
    start -= start % _PAGE
    end = min(size, -(-end // _PAGE) * _PAGE)
    if end <= start:
        return None
    return fd, start, end - start


def _record(kind: str, nbytes: int, seconds: float) -> None:
    us = seconds * 1e6
    with _stats_lock:
        _stats[f"{kind}_calls"] += 1
        _stats[f"bytes_{kind}"] += nbytes
        _stats["total_us"] += us
        if us > _stats["max_us"]:
            _stats["max_us"] = us


def advise_range(path: str, offset: int, length: int, advice: int) -> float:
    """Raw ungated ``posix_fadvise`` on one file range; seconds, -1 on failure.

    The substrate every gated helper uses (and what a bench uses to force an
    arm regardless of ``FTV_GGUF_MADV``). ``POSIX_FADV_DONTNEED`` alone does
    NOT free ranges this process has mapped - use :func:`advise_dontneed`
    (or :func:`drop_file_pages` when no mapping holds the range).
    """
    a = _aligned(path, offset, length)
    if a is None:
        return -1.0
    fd, off, length = a
    t0 = time.perf_counter()
    try:
        os.posix_fadvise(fd, off, length, advice)
    except OSError:
        with _stats_lock:
            _stats["errors"] += 1
        return -1.0
    dt = time.perf_counter() - t0
    kind = "dontneed" if advice == os.POSIX_FADV_DONTNEED else "willneed"
    _record(kind, length, dt)
    return dt


def advise_willneed(path: str, offset: int, length: int) -> bool:
    """Prefetch ``[offset, offset+length)``; no-op unless ``FTV_GGUF_MADV >= 1``.

    The fetch-side hook: call it BEFORE the engine reads the slab
    (``packed_gate_up_expert`` / ``expert_packed``) so the read, if it has to
    hit the disk, is at least queued. On this box it measures as a no-op
    (kernel does not start readahead for it); harmless and ~2-5 us.
    """
    if madvise_mode() < MODE_WILLNEED:
        return False
    return advise_range(path, offset, length, os.POSIX_FADV_WILLNEED) >= 0.0


def advise_dontneed(path: str, offset: int, length: int) -> bool:
    """FREE ``[offset, offset+length)`` from the page cache; no-op unless
    ``FTV_GGUF_MADV == 2``.

    The evict-side hook (the half that matters here): zap our PTEs with
    ``MADV_DONTNEED`` on the reader's mapping first, then ``fadvise
    DONTNEED`` so the pages actually leave the cache. Only ranges the
    engine has decided are cold should get this - blanket-advising the
    21.3 GB bank would evict the hot set (see docs for the measured
    failure mode). A live view into the range is safe: the bytes are clean
    file pages, so any re-access faults them back from RAM or - worst
    case - from disk, identical data, never SIGBUS (file is not truncated).
    """
    if madvise_mode() != MODE_DONTNEED:
        return False
    a = _aligned(path, offset, length)
    if a is None:
        return False
    _fd, off, length = a
    t0 = time.perf_counter()
    try:
        mapping = _mappings.get(_realpath(path))
        if mapping is not None:
            try:
                mapping.madvise(mmap.MADV_DONTNEED, off, length)
            except (OSError, ValueError):  # range outside the VMA: fadvise still helps
                pass
        os.posix_fadvise(_fd, off, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        with _stats_lock:
            _stats["errors"] += 1
        return False
    _record("dontneed", length, time.perf_counter() - t0)
    return True


def drop_file_pages(path: str) -> bool:
    """Measurement tool: evict the WHOLE file from the page cache (ungated).

    Reproduces a cold-cache run without root (``echo 3 > drop_caches`` is
    not available). Zaps the reader's PTEs across the whole mapping first -
    this is the one case a whole-file advice is correct, because the bench
    wants the file gone; the engine must never do this (it would evict the
    hot set it is about to read).
    """
    a = _aligned(path, 0, 1 << 62)
    if a is None:
        return False
    fd, off, length = a
    t0 = time.perf_counter()
    try:
        mapping = _mappings.get(_realpath(path))
        if mapping is not None:
            try:
                mapping.madvise(mmap.MADV_DONTNEED, off, length)
            except (OSError, ValueError):
                pass
        os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        return False
    _record("dontneed", length, time.perf_counter() - t0)
    return True


def expert_slab_range(
    index: dict[str, GgufTensorInfo], tensor: str | GgufTensorInfo, expert: int
) -> tuple[str, int, int] | None:
    """``(path, offset, nbytes)`` of ONE expert slab of a 3-D bank, or None.

    The file range ``packed_expert`` / ``dequant_expert`` will read for that
    expert - the unit the page-cache advice works on.
    """
    info = index.get(tensor) if isinstance(tensor, str) else tensor
    if info is None or len(info.ne) != 3:
        return None
    _k, rows, experts = info.ne
    if not 0 <= expert < experts:
        return None
    per_expert = int(info.nbytes) // int(experts)
    return info.path, info.offset + expert * per_expert, per_expert


def expert_ranges(
    index: dict[str, GgufTensorInfo], layer: int, expert: int
) -> list[tuple[str, int, int]]:
    """Every file range one routed-expert fetch of layer ``layer`` reads.

    ``blk.{layer}.ffn_{gate,up,down}_exps`` (or the stacked
    ``ffn_gate_up_exps`` bundle when the model has one) - the exact bytes
    ``packed_gate_up_expert`` + ``expert_packed`` touch for expert
    ``expert`` of that layer. Empty for unknown layers.
    """
    names = [f"blk.{layer}.ffn_gate_up_exps.weight"]
    if not any(name in index for name in names):
        names = [
            f"blk.{layer}.ffn_gate_exps.weight",
            f"blk.{layer}.ffn_up_exps.weight",
        ]
    names.append(f"blk.{layer}.ffn_down_exps.weight")
    out: list[tuple[str, int, int]] = []
    for name in names:
        ran = expert_slab_range(index, name, expert)
        if ran is not None:
            out.append(ran)
    return out


def advise_tensor_ranges(
    index: dict[str, GgufTensorInfo],
    ranges: Iterable[tuple[Any, int, int]],
    *,
    willneed: bool = True,
) -> int:
    """Advise a list of ``(tensor, offset, length)`` ranges (offset within the
    tensor, as the reader's offset/nbytes express it). Tensor entries are
    names or ``GgufTensorInfo``. Returns the number of ranges advised.

    This is the generic form the engine can drive from any policy (e.g. a
    batch of experts the router just picked); ``advise_expert_fetch`` /
    ``advise_expert_evict`` are the qwen35moe-shaped conveniences over it.
    """
    fired = 0
    for tensor, offset, length in ranges:
        info = index.get(tensor) if isinstance(tensor, str) else tensor
        if info is None:
            continue
        if willneed:
            fired += advise_willneed(info.path, info.offset + int(offset), int(length))
        else:
            fired += advise_dontneed(info.path, info.offset + int(offset), int(length))
    return int(fired)


def advise_expert_fetch(
    index: dict[str, GgufTensorInfo], layer: int, expert: int
) -> bool:
    """THE fetch-side call the engine should make, before reading the slab.

    Fires ``advise_willneed`` over every range ``expert_ranges`` returns.
    Returns True when any range was advised (i.e. the knob is on).
    """
    if madvise_mode() < MODE_WILLNEED:
        return False
    fired = False
    for path, off, length in expert_ranges(index, layer, expert):
        fired |= advise_willneed(path, off, length)
    return fired


def advise_expert_evict(
    index: dict[str, GgufTensorInfo], layer: int, expert: int
) -> bool:
    """THE evict-side call the engine should make, in its LRU victim loop.

    Fires ``advise_dontneed`` over every range ``expert_ranges`` returns -
    the pages of an expert the VRAM LRU just dropped leave the page cache
    too, so the hot set is the only part of the 21.3 GB bank competing for
    RAM. Returns True when any range was advised.
    """
    if madvise_mode() != MODE_DONTNEED:
        return False
    fired = False
    for path, off, length in expert_ranges(index, layer, expert):
        fired |= advise_dontneed(path, off, length)
    return fired


# -- fault / IO accounting ---------------------------------------------------
@dataclass(frozen=True)
class ProcCounters:
    """``/proc/self`` deltas: the honest cost of disk the decode paid."""

    minflt: int
    majflt: int
    read_bytes: int  # bytes actually fetched from storage (page-cache misses)
    rchar: int       # read() bytes - stays ~0 for an mmap'd model

    def delta(self, earlier: "ProcCounters") -> "ProcCounters":
        return ProcCounters(
            minflt=self.minflt - earlier.minflt,
            majflt=self.majflt - earlier.majflt,
            read_bytes=self.read_bytes - earlier.read_bytes,
            rchar=self.rchar - earlier.rchar,
        )

    @property
    def mib_read(self) -> float:
        return self.read_bytes / 2**20

    def as_dict(self) -> dict[str, float | int]:
        return {
            "minflt": self.minflt, "majflt": self.majflt,
            "read_bytes": self.read_bytes, "rchar": self.rchar,
            "mib_read": self.mib_read,
        }


def proc_counters() -> ProcCounters:
    """Snapshot min_flt/maj_flt + read_bytes/rchar. Never raises.

    ``maj_flt`` reads ~0 for mmap-backed model IO on Linux even when the
    disk is spinning (refaults of mapped file pages are minor faults), so
    ``read_bytes`` is the number to quote; both are recorded because the
    brief asks for both.
    """
    minflt = majflt = read_bytes = rchar = 0
    try:
        with open("/proc/self/stat") as f:
            p = f.read().split()
        minflt, majflt = int(p[9]), int(p[10])
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/self/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k == "read_bytes":
                    read_bytes = int(v)
                elif k == "rchar":
                    rchar = int(v)
    except OSError:
        pass
    return ProcCounters(minflt, majflt, read_bytes, rchar)


def advice_stats() -> dict[str, float]:
    """Cumulative advice cost: calls, bytes, total/max microseconds, errors."""
    with _stats_lock:
        return dict(_stats)


def reset_advice_stats() -> None:
    with _stats_lock:
        for key in _stats:
            _stats[key] = 0


def close_advice_fds() -> None:
    """Drop the cached fds/mappings (tests only; the engine keeps them)."""
    with _files_lock:
        for fd, _size in _files.values():
            try:
                os.close(fd)
            except OSError:
                pass
        _files.clear()
        _mappings.clear()
        _real.clear()


__all__ = [
    "ENV_KNOB", "MODE_OFF", "MODE_WILLNEED", "MODE_DONTNEED",
    "ProcCounters", "advice_stats", "advise_dontneed", "advise_expert_evict",
    "advise_expert_fetch", "advise_range", "advise_tensor_ranges",
    "advise_willneed", "close_advice_fds", "drop_file_pages", "expert_ranges",
    "expert_slab_range", "madvise_mode", "proc_counters", "register_mapping",
    "reset_advice_stats",
]
