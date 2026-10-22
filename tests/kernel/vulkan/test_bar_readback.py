"""Download staging readback: exact bytes + bandwidth floor.

The MTP rollback snapshot pulls the GDN state back from the GPU each round
through batch_download -> vku::download: GPU copies device-local -> pooled
HOST staging, then the CPU reads the mapping into the snapshot tensor.
Two things this file gates:

  (a) bit-exactness - the readback must return the exact uploaded bytes
      (rollback semantics depend on it; NaN-capable patterns compare as
      int32 so torch.equal's NaN rule cannot mask a mismatch);
  (b) a bandwidth floor - when the staging pool resolved to the UNCACHED
      host memory type, this read ran at ~340 MB/s and the 63 MB
      pull-down cost ~185 ms (the MTP snapshot's dominant cost). The
      HOST_CACHED type reads at ~10 GB/s, so the old path lands ~3x
      below this floor and any healthy path sits ~10x above it.
"""
from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

import freetoken.kernel.vulkan as vk  # noqa: E402

# Exact snapshot shapes (mtp.py pulls gdn_state + conv_buf for all 30
# linear-attention layers: nvfp4_resident.py allocates state as
# heads*key*value fp32 = 2 MB, conv as conv_dim*3 fp32 = 96 KB).
N_GDN = 30
STATE_ELEMS = 16 * 128 * 256
CONV_ELEMS = 8192 * 3
TOTAL_BYTES = N_GDN * (STATE_ELEMS + CONV_ELEMS) * 4
# Old uncached read: ~340 MB/s. New: ~10000 MB/s. Generous both ways.
FLOOR_MB_S = 1000.0


@pytest.fixture(scope="module")
def ext():
    if not vk.available():
        pytest.skip("no usable Vulkan device/extension on this machine")
    return vk.require_ext()


def test_snapshot_readback_bit_exact_and_fast(ext):
    torch.manual_seed(1234)
    src_state = [
        torch.randint(-2**31, 2**31 - 1, (STATE_ELEMS,), dtype=torch.int32)
        for _ in range(N_GDN)
    ]
    src_conv = [
        torch.randint(-2**31, 2**31 - 1, (CONV_ELEMS,), dtype=torch.int32)
        for _ in range(N_GDN)
    ]
    h_state = [ext.upload_resident(t) for t in src_state]
    h_conv = [ext.upload_resident(t) for t in src_conv]
    out_state = [torch.empty(STATE_ELEMS, dtype=torch.float32)
                 for _ in range(N_GDN)]
    out_conv = [torch.empty(CONV_ELEMS, dtype=torch.float32)
                for _ in range(N_GDN)]

    def pull():
        t0 = time.perf_counter()
        ext.batch_begin()
        for h, o in zip(h_state, out_state):
            ext.batch_download(h, o)
        for h, o in zip(h_conv, out_conv):
            ext.batch_download(h, o)
        ext.batch_submit()
        return (time.perf_counter() - t0) * 1e3

    try:
        pull()  # warmup: pool allocation + pipeline-free but caches settle
        walls = sorted(pull() for _ in range(3))
        wall_ms = walls[len(walls) // 2]

        for src, out in zip(src_state, out_state):
            assert torch.equal(src, out.view(torch.int32)), \
                "state readback bytes differ"
        for src, out in zip(src_conv, out_conv):
            assert torch.equal(src, out.view(torch.int32)), \
                "conv readback bytes differ"

        mb_s = TOTAL_BYTES / (1 << 20) / (wall_ms / 1e3)
        assert mb_s >= FLOOR_MB_S, (
            f"snapshot readback {mb_s:.0f} MB/s below {FLOOR_MB_S:.0f} MB/s "
            f"(wall {wall_ms:.1f} ms for {TOTAL_BYTES >> 20} MB) - staging "
            f"pool likely fell back to the uncached host memory type"
        )
    finally:
        for h in h_state + h_conv:
            ext.free_resident(h)
