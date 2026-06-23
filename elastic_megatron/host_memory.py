"""Host-memory release helpers used by CPU-offload reshard paths."""

import ctypes
import gc
import os
import time

import torch


_LIBC = None


def empty_host_cache() -> None:
    """Return cached pinned host blocks to the OS when PyTorch exposes the hook."""
    empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if empty_cache is not None:
        empty_cache()


def malloc_trim() -> None:
    """Return freed regular CPU heap pages to the OS on glibc systems."""
    global _LIBC
    if os.name != "posix":
        return
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL("libc.so.6")
        except OSError:
            _LIBC = False
    if _LIBC:
        try:
            _LIBC.malloc_trim(0)
        except AttributeError:
            pass


def trim_host_memory() -> None:
    # Gated per-call timing (ELASTIC_RESHARD_PHASE_TIMING=1) to isolate the host-mem
    # reclaim cost during a reshard; rank-0 only, near-zero overhead when off.
    _timing = os.environ.get("ELASTIC_RESHARD_PHASE_TIMING", "0") == "1"
    _t0 = time.perf_counter() if _timing else 0.0
    gc.collect()
    empty_host_cache()
    malloc_trim()
    if _timing:
        _dt_ms = (time.perf_counter() - _t0) * 1000.0
        try:
            _rank = torch.distributed.get_rank()
        except Exception:
            _rank = 0
        if _rank == 0:
            print(
                f"[ElasticMegatron-Perf] : reshard-phase trim_host_memory: {_dt_ms:.2f} ms",
                flush=True,
            )
