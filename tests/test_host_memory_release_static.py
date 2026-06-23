"""Static guard for CPU-adam host-memory (POOL B) release on the core branch.

Core-adapted port of the union test. The A100 reproduction showed that
``resize_(0)`` on pinned CPU tensors does not return memory to the OS until
PyTorch's host allocator cache is emptied. This test keeps the release path
wired to that flush.

Core differs from the union here:
  * POOL A (param-shaped master + moments) is freed by core's existing
    ``OptimizerTensorInfo.release()`` loop — there is NO chunked dst-setup, so the
    union's ``_init_empty_hdo_state`` / ``_empty_like_storage_zero`` /
    ``_release_all_param_shaped_states`` storage-0 helpers do NOT exist here.
  * The HDO override is CORE-MINIMAL: it frees ONLY POOL B (the pinned
    ``cpu_copy_map_grad`` grad buffers) and relies on ``dummy_step`` + the existing
    release loop for POOL A.
  * The union's chunked transfer path and the training_016 dataloader-shutdown
    guard are union-only and intentionally not asserted here.

These assertions are static (read the source text) and run without a GPU.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_optimizer_flushes_pinned_host_cache():
    src = (
        ROOT / "elastic_megatron" / "megatron_manager" / "training_state.py"
    ).read_text()
    helper = (ROOT / "elastic_megatron" / "host_memory.py").read_text()
    release_body = src.split("    def release_optimizer(self):", 1)[1].split(
        "    def rebuild_optimizer(self):", 1
    )[0]

    # POOL B release is wired into release_optimizer, gated default-ON, and the
    # host trim runs unconditionally once per release_optimizer.
    assert "release_offload_host_buffers()" in release_body
    assert 'os.environ.get("ELASTIC_RELEASE_HDO_HOST_BUFFERS", "1") == "1"' in release_body
    assert "trim_host_memory()" in release_body
    # POOL A is still freed first by the existing per-param release loop.
    assert "optimizer_tensor_info.release()" in release_body
    # The host helper returns pinned blocks (host cache) AND heap pages to the OS.
    assert "malloc_trim" in helper
    assert "_host_emptyCache" in helper


def test_hdo_release_frees_pool_b_only():
    """The core HDO override frees POOL B (cpu_copy_map_grad) and leaves POOL A to
    the existing release loop. It must drop the param.grad reference, resize the
    pinned storage to 0, clear the map (so the HDO lazily re-creates buffers on the
    next step), and trim host memory. It must NOT reintroduce the union's chunked
    storage-0 dst-setup helpers."""
    src = (
        ROOT / "elastic_megatron" / "resharding" / "optimizer_adapter.py"
    ).read_text()
    hdo_body = src.split("class HybridDeviceOptimizerAdapter", 1)[1]

    assert "def release_offload_host_buffers" in hdo_body
    assert "cpu_copy_map_grad" in hdo_body
    assert "param.grad = None" in hdo_body
    assert "untyped_storage().resize_(0)" in hdo_body
    assert "cpu_copy_map_grad.clear()" in hdo_body
    assert "trim_host_memory()" in hdo_body
    # core uses dummy_step for dst init; union-only storage-0 helpers are absent.
    assert "dummy_step()" in hdo_body
    assert "_init_empty_hdo_state" not in hdo_body
    assert "_empty_like_storage_zero" not in hdo_body
    assert "_release_all_param_shaped_states" not in hdo_body
    assert "_release_param_shaped_states" not in hdo_body


def test_base_release_offload_host_buffers_is_noop():
    """The base OptimizerAdapter must expose release_offload_host_buffers as a no-op
    so non-offload optimizers are safe to call uniformly in release_optimizer."""
    src = (
        ROOT / "elastic_megatron" / "resharding" / "optimizer_adapter.py"
    ).read_text()
    base_body = src.split("class OptimizerAdapter", 1)[1].split(
        "class Float16OptimizerAdapter", 1
    )[0]
    assert "def release_offload_host_buffers(self) -> None:" in base_body
    assert "from ..host_memory import trim_host_memory" in src


if __name__ == "__main__":
    test_release_optimizer_flushes_pinned_host_cache()
    test_hdo_release_frees_pool_b_only()
    test_base_release_offload_host_buffers_is_noop()
    print("PASS tests/test_host_memory_release_static.py")
