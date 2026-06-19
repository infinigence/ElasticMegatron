"""Host-memory release helpers used by CPU-offload reshard paths."""

import ctypes
import gc
import os

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
    gc.collect()
    empty_host_cache()
    malloc_trim()
