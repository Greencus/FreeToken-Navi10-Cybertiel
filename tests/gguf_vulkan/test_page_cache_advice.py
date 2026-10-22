"""Page-cache advice helpers (freetoken/models/gguf/madvise.py).

CPU-only, no model, no Vulkan device. The invariants that matter before any
engine lane is allowed to call these on the 22.8 GB CyberTiel file:

- the knobs gate every side effect (``FTV_GGUF_MADV`` default 0 = no-op);
- advising never corrupts the bytes, including through a live mmap view
  (the DONTNEED path zaps PTEs of the reader's mapping - the re-read must
  be byte-identical);
- absent paths are silent no-ops (returns False, no exception);
- the expert-slab range math matches what ``packed_expert`` would read;
- ``proc_counters`` deltas are non-negative and observe a real read.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from freetoken.models.gguf.madvise import (
    ENV_KNOB,
    MODE_DONTNEED,
    MODE_OFF,
    MODE_WILLNEED,
    ProcCounters,
    advice_stats,
    advise_dontneed,
    advise_expert_evict,
    advise_expert_fetch,
    advise_range,
    advise_tensor_ranges,
    advise_willneed,
    close_advice_fds,
    drop_file_pages,
    expert_ranges,
    expert_slab_range,
    madvise_mode,
    proc_counters,
    register_mapping,
    reset_advice_stats,
)
from freetoken.models.gguf.reader import GgufTensorInfo


@pytest.fixture(autouse=True)
def _clean_advice_state(monkeypatch):
    monkeypatch.delenv(ENV_KNOB, raising=False)
    reset_advice_stats()
    yield
    monkeypatch.delenv(ENV_KNOB, raising=False)
    close_advice_fds()
    reset_advice_stats()


@pytest.fixture()
def gguf_bytes(tmp_path):
    """A small file with content at a non-page-aligned-looking layout."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, size=3 << 20, dtype=np.uint8)
    path = tmp_path / "advice.gguf"
    path.write_bytes(data.tobytes())
    return str(path), data


def test_mode_defaults_off_and_parses(monkeypatch):
    assert madvise_mode() == MODE_OFF
    monkeypatch.setenv(ENV_KNOB, "1")
    assert madvise_mode() == MODE_WILLNEED
    monkeypatch.setenv(ENV_KNOB, "2")
    assert madvise_mode() == MODE_DONTNEED
    monkeypatch.setenv(ENV_KNOB, "junk")
    assert madvise_mode() == MODE_OFF
    monkeypatch.setenv(ENV_KNOB, "99")
    assert madvise_mode() == MODE_DONTNEED  # clamped, never wilder than evict


def test_gated_helpers_are_noops_by_default(gguf_bytes):
    path, data = gguf_bytes
    stats0 = advice_stats()
    assert advise_willneed(path, 0, 4096) is False
    assert advise_dontneed(path, 0, 4096) is False
    assert advise_expert_fetch({}, 0, 0) is False
    assert advice_stats() == stats0
    # and the bytes are untouched
    assert np.frombuffer(pathlib.Path(path).read_bytes(), dtype=np.uint8).tobytes() == data.tobytes()


def test_absent_file_is_silent_noop(monkeypatch):
    monkeypatch.setenv(ENV_KNOB, "2")
    ghost = "/nonexistent/path/definitely_missing.gguf"
    assert advise_willneed(ghost, 0, 1 << 20) is False
    assert advise_dontneed(ghost, 0, 1 << 20) is False
    assert drop_file_pages(ghost) is False
    assert advise_range(ghost, 0, 4096, os.POSIX_FADV_WILLNEED) == -1.0
    assert advise_expert_fetch({}, 3, 7) is False  # empty index, no crash


def test_dontneed_through_a_live_view_keeps_bytes(gguf_bytes, monkeypatch):
    """The zap+fadvise path must never corrupt data a reader view can see."""
    path, data = gguf_bytes
    monkeypatch.setenv(ENV_KNOB, str(MODE_DONTNEED))
    mm = np.memmap(path, mode="r")
    register_mapping(path, mm._mmap)
    view = mm[1 << 20 : 2 << 20]
    before = bytes(view[: 1 << 16])  # fault some pages in through the view

    assert advise_dontneed(path, 1 << 20, 1 << 20) is True
    assert advice_stats()["dontneed_calls"] == 1

    # through the SAME view (PTEs were zapped -> re-fault) ...
    assert bytes(view[: 1 << 16]) == before
    # ... and through a fresh mapping + plain read
    assert np.memmap(path, mode="r")[1 << 20 : 2 << 20][: 1 << 16].tobytes() == before
    assert pathlib.Path(path).read_bytes() == data.tobytes()
    del view, mm


def test_drop_file_pages_reads_back_identical(gguf_bytes, monkeypatch):
    path, data = gguf_bytes
    monkeypatch.setenv(ENV_KNOB, str(MODE_DONTNEED))
    mm = np.memmap(path, mode="r")
    register_mapping(path, mm._mmap)
    probe = bytes(mm[4096 : 4096 + 8192])
    assert drop_file_pages(path) is True
    assert np.memmap(path, mode="r")[4096 : 4096 + 8192].tobytes() == probe
    assert pathlib.Path(path).read_bytes() == data.tobytes()
    del mm


def test_willneed_fires_when_enabled(gguf_bytes, monkeypatch):
    path, _ = gguf_bytes
    monkeypatch.setenv(ENV_KNOB, str(MODE_WILLNEED))
    assert advise_willneed(path, 0, 1 << 20) is True
    assert advice_stats()["willneed_calls"] == 1
    assert advice_stats()["bytes_willneed"] >= 1 << 20
    # level 1 must NOT evict
    assert advise_dontneed(path, 0, 1 << 20) is False
    assert advice_stats()["dontneed_calls"] == 0


def test_expert_range_math_matches_packed_layout():
    """expert_slab_range/expert_ranges must slice the bank exactly like packed_expert."""
    from freetoken.models.gguf.dequant import expert_bytes_packed

    # Q5_K bank, k=256 (a K-quant superblock) x 128 rows x 8 experts
    per = 128 * 210  # rows * row_bytes(Q5_K, 256)
    bank = GgufTensorInfo(
        name="blk.5.ffn_down_exps.weight", path="/x.gguf", offset=1000,
        nbytes=8 * per, ggml_type=14, ne=(256, 128, 8), root="",
    )
    index = {bank.name: bank}
    ran = expert_slab_range(index, bank.name, 3)
    assert ran == ("/x.gguf", 1000 + 3 * per, per)
    assert ran[2] == expert_bytes_packed(bank)  # same per-slab bytes
    assert expert_slab_range(index, bank.name, 8) is None   # out of range
    assert expert_slab_range(index, "nope", 0) is None      # unknown tensor

    # Q4_K gate/up banks, k=256 x 64 rows x 8 experts
    gper = 64 * 144
    gate = GgufTensorInfo("blk.5.ffn_gate_exps.weight", "/x.gguf", 0, 8 * gper, 12, (256, 64, 8))
    up = GgufTensorInfo("blk.5.ffn_up_exps.weight", "/x.gguf", 8 * gper, 8 * gper, 12, (256, 64, 8))
    idx = {i.name: i for i in (gate, up, bank)}
    assert expert_ranges(idx, 5, 2) == [
        ("/x.gguf", 2 * gper, gper), ("/x.gguf", 8 * gper + 2 * gper, gper),
        ("/x.gguf", 1000 + 2 * per, per),
    ]
    # a stacked gate_up bundle model names one tensor instead
    bundle = GgufTensorInfo("blk.5.ffn_gate_up_exps.weight", "/x.gguf", 0, 8 * 2 * gper, 12, (256, 128, 8))
    assert expert_ranges({bundle.name: bundle, bank.name: bank}, 5, 0) == [
        ("/x.gguf", 0, 2 * gper), ("/x.gguf", 1000, per),
    ]
    assert expert_ranges(idx, 6, 0) == []  # layer without tensors


def test_advise_tensor_ranges_offsets_are_within_tensor(gguf_bytes, monkeypatch):
    path, data = gguf_bytes
    monkeypatch.setenv(ENV_KNOB, str(MODE_DONTNEED))
    info = GgufTensorInfo("t", path, 1 << 20, 1 << 20, 12, (128, 1024))
    mm = np.memmap(path, mode="r")
    register_mapping(path, mm._mmap)
    n = advise_tensor_ranges({"t": info}, [("t", 0, 1 << 20), ("missing", 0, 4)], willneed=False)
    assert n == 1  # unknown tensor skipped, offset shifted by info.offset
    assert pathlib.Path(path).read_bytes() == data.tobytes()
    del mm


def test_expert_fetch_evict_gating(gguf_bytes, monkeypatch):
    path, data = gguf_bytes
    bank = GgufTensorInfo("blk.0.ffn_down_exps.weight", path, 0, 1 << 20, 14, (64, 128, 8))
    index = {bank.name: bank}
    assert advise_expert_fetch(index, 0, 1) is False   # knob off
    assert advise_expert_evict(index, 0, 1) is False
    monkeypatch.setenv(ENV_KNOB, str(MODE_WILLNEED))
    assert advise_expert_fetch(index, 0, 1) is True
    assert advise_expert_evict(index, 0, 1) is False   # evict needs level 2
    monkeypatch.setenv(ENV_KNOB, str(MODE_DONTNEED))
    assert advise_expert_evict(index, 0, 1) is True
    assert pathlib.Path(path).read_bytes() == data.tobytes()


def _fstype(path: str) -> str:
    """Filesystem type of the mount holding ``path`` (tmpfs has no disk reads)."""
    best, kind = "", ""
    try:
        with open("/proc/self/mounts") as f:
            for line in f:
                _dev, mp, fst, *_ = line.split()
                if (path == mp or path.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best):
                    best, kind = mp, fst
    except OSError:
        pass
    return kind


def test_proc_counters_delta_is_honest(tmp_path):
    before = proc_counters()
    # /tmp may be tmpfs (no storage behind it); FTV_TEST_PROBE_DIR lets the
    # caller point at a real filesystem so the read_bytes half runs there.
    probe_dir = os.environ.get("FTV_TEST_PROBE_DIR") or str(tmp_path)
    on_disk = _fstype(probe_dir) not in ("tmpfs", "ramfs", "")
    path = os.path.join(probe_dir, f"io_probe_{os.getpid()}.bin")
    blob = os.urandom(4 << 20)
    try:
        with open(path, "wb") as f:
            f.write(blob)
            f.flush()
            os.fdatasync(f.fileno())
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # clean, no mapping -> evicts
        os.close(fd)
        mm = np.memmap(path, mode="r")
        acc = 0
        for i in range(0, len(blob), 4096):
            acc ^= int(mm[i])
        after = proc_counters()
        del mm
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    delta = after.delta(before)
    assert delta.minflt > 0 and delta.majflt >= 0
    # NOT 1-fault-per-page: the kernel's fault-around maps ~16 pages per
    # file fault, so minflt counts faults, not pages - read_bytes is the
    # robust disk signal (which is the point of the helper).
    assert delta.minflt < len(blob) // 4096
    if on_disk:
        assert delta.read_bytes > 0, "cold mmap read must show up in /proc/self/io read_bytes"
        assert delta.mib_read > 1.0
    assert isinstance(delta, ProcCounters) and delta.as_dict()["mib_read"] == delta.mib_read
    assert acc >= 0  # the read itself is what we measured, keep it used


def test_register_mapping_publishes_realpath(tmp_path):
    """register_mapping (what reader._reader calls) keys the mapping by realpath."""
    path = tmp_path / "tiny.gguf"
    path.write_bytes(os.urandom(1 << 20))
    mm = np.memmap(str(path), mode="r")
    register_mapping(str(path), mm._mmap)
    from freetoken.models.gguf.madvise import _mappings

    assert os.path.realpath(str(path)) in _mappings
    del mm


@pytest.mark.skipif(
    not os.environ.get("FREETOKEN_TEST_MODEL_GGUF")
    or not os.path.isfile(os.environ["FREETOKEN_TEST_MODEL_GGUF"]),
    reason="FREETOKEN_TEST_MODEL_GGUF not set",
)
def test_real_gguf_advice_roundtrip():
    """On the real file: registration happened, advice fires, bytes intact."""
    from freetoken.models.gguf import reader as reader_mod
    from freetoken.models.gguf.madvise import _mappings

    path = os.environ["FREETOKEN_TEST_MODEL_GGUF"]
    # Re-open through reader._reader so the mapping registration this test
    # asserts on definitely runs in THIS process (the readers are cached).
    reader_mod.gguf_tensor_index.cache_clear()
    reader_mod._reader.cache_clear()
    index = reader_mod.gguf_tensor_index(path)
    assert os.path.realpath(reader_mod.gguf_split_paths(path)[0]) in _mappings

    gate = next(n for n in index if "ffn_gate_exps" in n or "ffn_gate_up_exps" in n)
    ran = expert_slab_range(index, gate, 0)
    assert ran is not None and ran[2] > 0
    view = np.memmap(path, mode="r")
    before = bytes(view[ran[1] : ran[1] + min(ran[2], 1 << 16)])
    os.environ[ENV_KNOB] = str(MODE_DONTNEED)
    try:
        assert advise_dontneed(ran[0], ran[1], ran[2]) is True
        assert bytes(view[ran[1] : ran[1] + min(ran[2], 1 << 16)]) == before
    finally:
        del os.environ[ENV_KNOB]
        del view
