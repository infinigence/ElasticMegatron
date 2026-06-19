"""Static guard for CPU-adam host-memory release.

The A100 reproduction showed that ``resize_(0)`` on pinned CPU tensors does not
return memory to the OS until PyTorch's host allocator cache is emptied. This
test keeps the release path wired to that flush.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_optimizer_flushes_pinned_host_cache():
    src = (ROOT / "elastic_megatron" / "megatron_manager" / "training_state.py").read_text()
    helper = (ROOT / "elastic_megatron" / "host_memory.py").read_text()
    release_body = src.split("    def release_optimizer(self):", 1)[1].split(
        "    def rebuild_optimizer(self):", 1
    )[0]

    assert "release_offload_host_buffers()" in release_body
    assert "trim_host_memory()" in release_body
    assert "malloc_trim" in helper


def test_cpu_adam_transfer_flushes_host_cache_in_chunked_path():
    """The chunked flat-peak main process subsumes the old cpu-adam-only
    ``_main_process_streaming``. It must still bound cpu-adam host memory: barrier
    the async H2D pack copy before freeing host src, and return the freed pinned
    blocks to the OS (so host RSS matches a non-elastic run). The standalone
    streaming method and its env knob are gone -- guard against reintroduction."""
    src = (ROOT / "elastic_megatron" / "transfer" / "transfer.py").read_text()
    # host-cache reclaim wired into the UNIFIED chunked path
    assert "def _main_process_chunk" in src
    assert "def _host_storage_nbytes" in src
    assert "empty_host_cache()" in src
    assert "current_stream().synchronize()" in src
    assert 'os.getenv("ELASTIC_TRANSFER_LOG_LEVEL", "0")' in src
    # the cpu-adam-only streaming path is subsumed, not kept alongside
    assert "_main_process_streaming" not in src
    assert "ELASTIC_TRANSFER_STREAM_OPT_TENSORS" not in src


def test_hdo_dst_init_uses_storage_zero_placeholders():
    src = (ROOT / "elastic_megatron" / "resharding" / "optimizer_adapter.py").read_text()
    hdo_body = src.split("class HybridDeviceOptimizerAdapter", 1)[1]
    assert "dummy_step()" in hdo_body
    assert "def _init_empty_hdo_state" in hdo_body
    assert "_empty_like_storage_zero" in hdo_body
    assert "_release_all_param_shaped_states()" in hdo_body
    assert "_release_param_shaped_states" in hdo_body
    assert "def release_offload_host_buffers" in hdo_body
    assert "trim_host_memory()" in hdo_body


def test_training_shutdowns_dataloader_workers_before_reshard():
    src = (ROOT / "examples" / "intra_process" / "training_016.py").read_text()
    assert "def shutdown_data_iterators_before_reshard" in src
    assert '"_shutdown_workers"' in src
    assert "shutdown_workers()" in src
    assert "shutdown_data_iterators_before_reshard(" in src
    assert "elastic_megatron_manager.reshard(" in src
    loop_body = src.split("while iteration < args.train_iters:", 1)[1]
    shutdown_call = "                shutdown_data_iterators_before_reshard("
    reshard_call = "                training_state = elastic_megatron_manager.reshard("
    assert loop_body.index(shutdown_call) < loop_body.index(reshard_call)


if __name__ == "__main__":
    test_release_optimizer_flushes_pinned_host_cache()
    test_cpu_adam_transfer_flushes_host_cache_in_chunked_path()
    test_hdo_dst_init_uses_storage_zero_placeholders()
    test_training_shutdowns_dataloader_workers_before_reshard()
    print("PASS tests/test_host_memory_release_static.py")
