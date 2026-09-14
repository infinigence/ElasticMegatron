from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Tuple

import torch
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0

_OVERLAY_PG_NCCL = None
_OVERLAY_PG_GLOO = None
_OVERLAY_PG_INITED = False
_OVERLAY_ASYNC_PROC_STARTED = False
_OVERLAY_ASYNC_READY = False
_OVERLAY_ASYNC_IPC_ATTACHED = False


def _overlay_warmup_after_init() -> None:
    """Warm up overlay collectives so first real metadata broadcast is cheaper.
    """
    if _OVERLAY_PG_NCCL is None or _OVERLAY_PG_GLOO is None:
        return

    warmup_t = time.perf_counter()
    try:
        if torch.cuda.is_available():
            nccl_t = torch.zeros(1, device="cuda", dtype=torch.float32)
            torch.distributed.all_reduce(nccl_t, group=_OVERLAY_PG_NCCL)

        gloo_t = torch.zeros(1, device="cpu", dtype=torch.int64)
        torch.distributed.broadcast(gloo_t, src=0, group=_OVERLAY_PG_GLOO)
        _overlay_timing("init_groups/warmup_collectives", time.perf_counter() - warmup_t)
    except Exception as e:
        _overlay_debug(f"overlay warmup skipped due to error: {e}")


def _get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _overlay_enabled() -> bool:
    return _get_bool_env("ELASTIC_OVERLAY_ENABLE", False)


def _get_int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _overlay_debug(msg: str) -> None:
    if not _get_bool_env("ELASTIC_OVERLAY_DEBUG", False):
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    except Exception:
        rank = -1
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [overlay-debug] rank={rank} {msg}", flush=True)


def _overlay_timing_enabled() -> bool:
    return _get_bool_env(
        "ELASTIC_OVERLAY_TIMING",
        _get_bool_env("ELASTIC_OVERLAY_DEBUG", False),
    )


def _overlay_timing_rank0_only() -> bool:
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        return False
    return True


def _overlay_timing(stage: str, elapsed_sec: float) -> None:
    if not _overlay_timing_enabled():
        return
    if not _overlay_timing_rank0_only():
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [overlay-timing] rank=0 {stage}: {elapsed_sec * 1000.0:.2f} ms", flush=True)


def get_overlay_role() -> str:
    return os.getenv("ELASTIC_OVERLAY_ROLE", "receiver").strip().lower()


def is_overlay_new_process() -> bool:
    return os.getenv("NEW_PROCESS") is not None


def is_overlay_sender_proxy_process() -> bool:
    return (
        os.getenv("NEW_PROCESS_REDEPLOY") is not None
    )


def _overlay_broadcast_ready_from_rank0() -> bool:
    if not torch.distributed.is_initialized():
        return False

    ready_val = 0
    if torch.distributed.get_rank() == 0:
        try:
            from tools.elastic_control.env_utils import _get_ready_flag

            ready_val = 1 if int(_get_ready_flag()) == 1 else 0
        except Exception:
            ready_val = 0

    ready_tensor = torch.tensor(
        [ready_val], dtype=torch.int, device=("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.distributed.broadcast(ready_tensor, src=0)
    return bool(ready_tensor.item())


def launch_async_overlay_processes(iteration: int) -> None:
    global _OVERLAY_ASYNC_PROC_STARTED

    if get_overlay_role() != "sender" or is_overlay_new_process():
        return
    if _OVERLAY_ASYNC_PROC_STARTED:
        return

    launch_iter = _get_int_env("ELASTIC_OVERLAY_ASYNC_LAUNCH_ITER", -1)
    if iteration != launch_iter:
        return

    launch_ok = torch.tensor(
        [0], dtype=torch.int, device=("cuda" if torch.cuda.is_available() else "cpu")
    )
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if local_rank == 0:
            try:
                from tools.elastic_control.env_utils import ElasticMode, _start_new_megatron_proc

                redeploy_script_path = os.getenv("ELASTIC_OVERLAY_ASYNC_REDEPLOY_SCRIPT")
                node1_ip = os.getenv("ELASTIC_OVERLAY_ASYNC_NODE1_IP")
                _start_new_megatron_proc(
                    ElasticMode.SCALE_DOWN,
                    redeploy_script_path=redeploy_script_path,
                    node1_ip=node1_ip,
                )
                launch_ok[0] = 1
                _overlay_debug(f"async launch done at iteration={iteration}")
            except Exception as e:
                print_rank_0(f"[overlay-async] launch new process failed: {e}")

    if torch.distributed.is_initialized():
        torch.distributed.broadcast(launch_ok, src=0)

    if launch_ok.item() == 1:
        _OVERLAY_ASYNC_PROC_STARTED = True


def _overlay_async_set_ready_flag() -> None:
    # READY means the new sender process has completed 16-rank overlay PG init.
    if not is_overlay_new_process():
        return
    if not is_overlay_sender_proxy_process():
        return
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() != 0:
        return
    if int(os.getenv("LOCAL_RANK", "0")) != 0:
        return

    shm_name = os.getenv("NEW_NODE_READY_SHM")
    if not shm_name:
        return

    from multiprocessing import shared_memory

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.buf[0] = 1
        shm.close()
        _overlay_debug("async ready flag set after overlay group init")
    except FileNotFoundError:
        return
    except Exception as e:
        print_rank_0(f"[overlay-async] set ready flag failed: {e}")


def update_async_overlay_ready_state() -> bool:
    global _OVERLAY_ASYNC_READY

    if not _OVERLAY_ASYNC_PROC_STARTED:
        _OVERLAY_ASYNC_READY = False
        return False

    _OVERLAY_ASYNC_READY = _overlay_broadcast_ready_from_rank0()
    return _OVERLAY_ASYNC_READY


def attach_ipc_state_if_async_sender_proxy(model, optimizer, opt_param_scheduler) -> bool:
    global _OVERLAY_ASYNC_IPC_ATTACHED

    if _OVERLAY_ASYNC_IPC_ATTACHED:
        return False
    if get_overlay_role() != "sender" or not is_overlay_new_process():
        return False
    if not is_overlay_sender_proxy_process():
        return False

    from elastic_megatron.transfer.ipc_manager import _attach_state_to_model, receive_training_state

    # Pre-init overlay groups in the new sender process, then mark ready for old process.
    if _OVERLAY_PG_NCCL is None or _OVERLAY_PG_GLOO is None:
        async_pg_init_t = time.perf_counter()
        init_overlay_process_groups()
        _overlay_timing("async_proxy/init_overlay_groups", time.perf_counter() - async_pg_init_t)
    _overlay_async_set_ready_flag()

    state = receive_training_state()
    _attach_state_to_model(state, model, optimizer, opt_param_scheduler)
    _OVERLAY_ASYNC_IPC_ATTACHED = True
    _overlay_debug("async sender process attached IPC state")
    return True


def sender_proxy_redeploy() -> bool:
    return (
        get_overlay_role() == "sender"
        and is_overlay_new_process()
        and is_overlay_sender_proxy_process()
        and _OVERLAY_ASYNC_IPC_ATTACHED
    )


def ipc_sender(iteration, model, optimizer, opt_param_scheduler) -> bool:
    if get_overlay_role() != "sender" or is_overlay_new_process():
        return False
    if not _OVERLAY_ASYNC_PROC_STARTED or not _OVERLAY_ASYNC_READY:
        return False

    _overlay_debug(f"async handoff begin at iteration={iteration}")
    from elastic_megatron.transfer.ipc_manager import send_training_state

    args = get_args()
    cur_parallel_strategy = {
        "world_size": int(getattr(args, "world_size", 1)),
        "tensor_model_parallel_size": int(getattr(args, "tensor_model_parallel_size", 1)),
        "pipeline_model_parallel_size": int(getattr(args, "pipeline_model_parallel_size", 1)),
        "data_parallel_size": int(getattr(args, "data_parallel_size", 1)),
    }
    # Keep sender/receiver IPC ports consistent in async scale-down workflow.
    ipc_base_port = os.getenv("ELASTIC_IPC_PORT_SCALE_DOWN", os.getenv("ELASTIC_IPC_PORT", "7000"))
    _overlay_debug(f"async handoff ipc base_port={ipc_base_port}")

    send_training_state(
        model,
        optimizer,
        iteration,
        opt_param_scheduler,
        base_port=ipc_base_port,
        cur_parallel_strategy=cur_parallel_strategy,
    )
    return True


def _init_custom_process_group(
    backend: str,
    init_method: str,
    world_size: int,
    rank: int,
    group_name: str,
    timeout=None,
    store=None,
):
    """Create an additional main ProcessGroup without touching the default WORLD group.
    """
    import importlib

    if _get_bool_env("ELASTIC_OVERLAY_USE_SGLANG", True):
        try:
            sglang_utils = importlib.import_module("sglang.srt.utils")
            _sglang_init_cpg = getattr(sglang_utils, "init_custom_process_group")
            return _sglang_init_cpg(
                backend=backend,
                init_method=init_method,
                timeout=timeout,
                world_size=world_size,
                rank=rank,
                store=store,
                group_name=group_name,
                pg_options=None,
            )
        except Exception:
            pass

    c10d = importlib.import_module("torch.distributed.distributed_c10d")
    Backend = c10d.Backend
    PrefixStore = c10d.PrefixStore
    _new_process_group_helper = c10d._new_process_group_helper
    _world = c10d._world
    default_pg_timeout = c10d.default_pg_timeout
    rendezvous = c10d.rendezvous

    def _torch_version_ge(major: int, minor: int) -> bool:
        version_str = str(torch.__version__).split("+", 1)[0]
        parts = version_str.split(".")
        try:
            v_major = int(parts[0]) if len(parts) > 0 else 0
            v_minor = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            return False
        return (v_major, v_minor) >= (major, minor)

    if timeout is None:
        timeout = default_pg_timeout

    if (store is None) and (init_method is None):
        raise ValueError("init_method or store must be provided for overlay process group")

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    backend = Backend(backend) if backend else Backend("undefined")

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    pg_options_param_name = "backend_options" if _torch_version_ge(2, 6) else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: None},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


def init_overlay_process_groups() -> None:
    global _OVERLAY_PG_INITED, _OVERLAY_PG_NCCL, _OVERLAY_PG_GLOO
    init_start_t = time.perf_counter()
    if _OVERLAY_PG_INITED:
        return

    if not _get_bool_env("ELASTIC_OVERLAY_ENABLE", False):
        _OVERLAY_PG_INITED = True
        return

    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before overlay redeploy")

    init_method = os.getenv("ELASTIC_OVERLAY_INIT_METHOD")
    if not init_method:
        raise RuntimeError(
            "ELASTIC_OVERLAY_INIT_METHOD is required when ELASTIC_OVERLAY_ENABLE=1 "
            "(e.g., tcp://<host>:<port>)"
        )

    group_name = os.getenv("ELASTIC_OVERLAY_GROUP_NAME", "elastic_overlay")
    overlay_world_size = int(os.getenv("ELASTIC_OVERLAY_WORLD_SIZE", "16"))
    overlay_rank_offset = int(os.getenv("ELASTIC_OVERLAY_RANK_OFFSET", "0"))
    local_rank = torch.distributed.get_rank()
    overlay_rank = local_rank + overlay_rank_offset

    timeout_sec = _get_int_env("ELASTIC_OVERLAY_TIMEOUT_SEC", 180)
    print(f"ELASTIC_OVERLAY_ROLE {get_overlay_role()} initializing overlay groups")

    _overlay_debug(
        f"init overlay groups: init_method={init_method} group_name={group_name} "
        f"overlay_world_size={overlay_world_size} local_rank={local_rank} "
        f"overlay_rank={overlay_rank} timeout_sec={timeout_sec}"
    )

    if overlay_rank < 0 or overlay_rank >= overlay_world_size:
        raise ValueError(
            f"Invalid overlay rank mapping: local_rank={local_rank}, "
            f"offset={overlay_rank_offset}, overlay_rank={overlay_rank}, "
            f"overlay_world_size={overlay_world_size}"
        )
    import importlib

    c10d = importlib.import_module("torch.distributed.distributed_c10d")
    PrefixStore = c10d.PrefixStore
    default_pg_timeout = c10d.default_pg_timeout
    rendezvous = c10d.rendezvous
    timeout = default_pg_timeout
    if timeout_sec is not None and timeout_sec > 0:
        timeout = timedelta(seconds=timeout_sec)

    # Build base store.
    store_start_t = time.perf_counter()
    if init_method.startswith("tcp://"):
        from urllib.parse import urlparse

        parsed = urlparse(init_method)
        host = parsed.hostname
        port = parsed.port
        if not host or not port:
            raise ValueError(
                f"Invalid ELASTIC_OVERLAY_INIT_METHOD (expect tcp://host:port): {init_method}"
            )

        
        listen_addr = os.getenv("ELASTIC_OVERLAY_LISTEN_ADDR", "0.0.0.0")
        store_host = listen_addr if overlay_rank == 0 else host

        _overlay_debug(
            f"TCPStore begin: is_master={overlay_rank==0} connect_host={host} "
            f"store_host={store_host} port={port}"
        )
        base_store = torch.distributed.TCPStore(
            store_host,
            port,
            overlay_world_size,
            overlay_rank == 0,
            timeout=timeout,
            wait_for_workers=True,
            multi_tenant=True,
        )
        base_store = PrefixStore(group_name, base_store)
        _overlay_debug("TCPStore done")
    else:
        _overlay_debug("rendezvous for overlay store begin")
        rendezvous_iterator = rendezvous(
            init_method, overlay_rank, overlay_world_size, timeout=timeout
        )
        base_store, _, _ = next(rendezvous_iterator)
        base_store.set_timeout(timeout)
        base_store = PrefixStore(group_name, base_store)
        _overlay_debug("rendezvous for overlay store done")
    _overlay_timing("init_groups/store", time.perf_counter() - store_start_t)

    nccl_store = PrefixStore(f"{group_name}_nccl", base_store)
    gloo_store = PrefixStore(f"{group_name}_gloo", base_store)

    # Build two overlay groups: NCCL for tensor send/recv, Gloo for object broadcast.
    _overlay_debug("create overlay nccl pg begin")
    nccl_pg_start_t = time.perf_counter()
    _OVERLAY_PG_NCCL = _init_custom_process_group(
        backend="nccl",
        init_method=None,
        store=nccl_store,
        world_size=overlay_world_size,
        rank=overlay_rank,
        group_name=f"{group_name}_nccl",
        timeout=timeout,
    )
    _overlay_debug("create overlay nccl pg done")
    _overlay_timing("init_groups/create_nccl_pg", time.perf_counter() - nccl_pg_start_t)

    _overlay_debug("create overlay gloo pg begin")
    gloo_pg_start_t = time.perf_counter()
    _OVERLAY_PG_GLOO = _init_custom_process_group(
        backend="gloo",
        init_method=None,
        store=gloo_store,
        world_size=overlay_world_size,
        rank=overlay_rank,
        group_name=f"{group_name}_gloo",
        timeout=timeout,
    )
    _overlay_debug("create overlay gloo pg done")
    _overlay_timing("init_groups/create_gloo_pg", time.perf_counter() - gloo_pg_start_t)

    # warm up overlay collectives to reduce latency of first real metadata broadcast
    _overlay_warmup_after_init()

    _OVERLAY_PG_INITED = True
    _overlay_timing("init_groups/total", time.perf_counter() - init_start_t)


def redeploy_transfer(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    *,
    local_trigger_iter: int | None = None,
):
    """Overlay redeployment hook.

    Env vars:
      - ELASTIC_OVERLAY_ENABLE=1
      - ELASTIC_OVERLAY_INIT_METHOD=tcp://<host>:<port>
      - ELASTIC_OVERLAY_REDEPLOY_ITER=<int>
      - ELASTIC_OVERLAY_ROLE=sender|receiver
      - ELASTIC_OVERLAY_WORLD_SIZE=16 (optional)
      - ELASTIC_OVERLAY_RANK_OFFSET=0 or 8 (optional)
      - ELASTIC_OVERLAY_PEER_STRIDE=8 (optional)
      - ELASTIC_OVERLAY_SENDER_EXIT=1 (optional, default true)
    """
    if not _get_bool_env("ELASTIC_OVERLAY_ENABLE", False):
        return False

    role = os.getenv("ELASTIC_OVERLAY_ROLE", "receiver").strip().lower()
    if role not in ("sender", "receiver"):
        raise ValueError(f"Invalid ELASTIC_OVERLAY_ROLE: {role}")

    redeploy_iter = os.getenv("ELASTIC_OVERLAY_REDEPLOY_ITER")
    if redeploy_iter is None:
        return False
    redeploy_iter = int(redeploy_iter)

    if local_trigger_iter is None:
        local_trigger_iter = redeploy_iter

    if iteration != int(local_trigger_iter):
        return False

    redeploy_total_t = time.perf_counter()

    print_rank_0(
        f"[overlay-redeploy] trigger at iteration={iteration} role={role} "
        f"(sender_iter={redeploy_iter})"
    )

    pg_init_t = time.perf_counter()
    group_init_to_transfer_t = pg_init_t
    init_overlay_process_groups()
    _overlay_timing("redeploy/init_overlay_groups", time.perf_counter() - pg_init_t)
    if _OVERLAY_PG_NCCL is None or _OVERLAY_PG_GLOO is None:
        return False

    overlay_world_size = int(os.getenv("ELASTIC_OVERLAY_WORLD_SIZE", "16"))
    overlay_rank_offset = int(os.getenv("ELASTIC_OVERLAY_RANK_OFFSET", "0"))
    peer_stride = int(os.getenv("ELASTIC_OVERLAY_PEER_STRIDE", str(overlay_world_size // 2)))

    local_rank = torch.distributed.get_rank()
    overlay_rank = local_rank + overlay_rank_offset
    peer_overlay_rank = overlay_rank + peer_stride if role == "sender" else overlay_rank - peer_stride
    if peer_overlay_rank < 0 or peer_overlay_rank >= overlay_world_size:
        raise ValueError(
            f"Invalid peer mapping: overlay_rank={overlay_rank}, peer_stride={peer_stride}, "
            f"role={role}, peer_overlay_rank={peer_overlay_rank}, overlay_world_size={overlay_world_size}"
        )

    _overlay_debug(
        f"redeploy begin: role={role} local_rank={local_rank} overlay_rank={overlay_rank} "
        f"peer_overlay_rank={peer_overlay_rank} sender_iter={redeploy_iter}"
    )

    # Build TrainingState metadata to enumerate optimizer tensors.
    from elastic_megatron.megatron_manager.parallel_strategy import ParallelStrategy
    from elastic_megatron.megatron_manager.training_state import (
        TrainingState,
        load_state_dict_with_no_step,
        update_optimizer_by_state_dict,
    )

    args = get_args()
    parallel_strategy = ParallelStrategy(
        world_size=args.world_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=getattr(args, "context_parallel_size", 1),
        num_distributed_optimizer_instances=getattr(args, "num_distributed_optimizer_instances", 1),
        expert_model_parallel_size=getattr(args, "expert_model_parallel_size", 1),
        expert_tensor_parallel_size=getattr(args, "expert_tensor_parallel_size", None),
        sequence_parallel=getattr(args, "sequence_parallel", None),
    )
    parallel_strategy.register_mpu_state()

    training_state_t = time.perf_counter()
    training_state = TrainingState(model, optimizer, opt_param_scheduler)
    training_state.init_metadata(parallel_strategy, offload_opt_tensors=False)
    _overlay_timing("redeploy/build_training_state", time.perf_counter() - training_state_t)

    # Small metadata via overlay NCCL group
    def _broadcast_object_nccl(obj, src_overlay_rank: int):
        import pickle

        if not torch.cuda.is_available():
            raise RuntimeError("Overlay NCCL metadata broadcast requires CUDA")

        payload = b""
        if overlay_rank == src_overlay_rank:
            payload = pickle.dumps(obj)

        length = torch.tensor([len(payload)], device="cuda", dtype=torch.int64)
        torch.distributed.broadcast(length, src=src_overlay_rank, group=_OVERLAY_PG_NCCL)
        nbytes = int(length.item())

        if overlay_rank == src_overlay_rank:
            try:
                cpu_buf = torch.frombuffer(memoryview(payload), dtype=torch.uint8).clone()
            except Exception:
                cpu_buf = torch.tensor(list(payload), dtype=torch.uint8)
            buf = cpu_buf.to(device="cuda", non_blocking=True)
        else:
            buf = torch.empty((nbytes,), device="cuda", dtype=torch.uint8)

        torch.distributed.broadcast(buf, src=src_overlay_rank, group=_OVERLAY_PG_NCCL)
        if overlay_rank != src_overlay_rank:
            payload = buf.cpu().numpy().tobytes()
        return pickle.loads(payload)

    meta_obj = None
    if role == "sender" and overlay_rank == 0:
        # Send hyperparams and scheduler state.
        opt_param_groups = []
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            for param_group in optimizer.param_groups:
                opt_param_groups.append(
                    {k: v for k, v in param_group.items() if k != "params"}
                )

        meta_obj = {
            "iteration": int(iteration),
            "global_step": int(iteration) + 1,
            "consumed_train_samples": getattr(args, "consumed_train_samples", 0),
            "skipped_train_samples": getattr(args, "skipped_train_samples", 0),
            "num_floating_point_operations_so_far": getattr(
                args, "num_floating_point_operations_so_far", 0.0
            ),
            "optimizer_param_groups": opt_param_groups,
            "scheduler_state": opt_param_scheduler.state_dict()
            if opt_param_scheduler is not None
            else None,
        }
    _overlay_debug("broadcast_meta(nccl) begin")
    meta_bcast_t = time.perf_counter()
    meta_obj = _broadcast_object_nccl(meta_obj, src_overlay_rank=0)
    _overlay_debug("broadcast_meta(nccl) done")
    _overlay_timing("redeploy/broadcast_meta", time.perf_counter() - meta_bcast_t)

    # Transfer optimizer tensors via NCCL overlay group.
    if role == "sender":
        total_tensors = sum(
            len(opt_info.optimizer_tensors)
            for opt_info in training_state.optimizer_tensor_info_list
        )
        _overlay_debug(f"sender send begin: total_tensors={total_tensors}")
        send_t = time.perf_counter()
        for opt_info in training_state.optimizer_tensor_info_list:
            for t in opt_info.optimizer_tensors:
                torch.distributed.send(t, dst=peer_overlay_rank, group=_OVERLAY_PG_NCCL)
        _overlay_debug("sender send done")
        _overlay_timing("redeploy/sender_send_tensors", time.perf_counter() - send_t)
        _overlay_timing(
            "summary/group_init_to_transfer_done",
            time.perf_counter() - group_init_to_transfer_t,
        )
        _overlay_debug("sender barrier(nccl) begin")
        sender_barrier_t = time.perf_counter()
        torch.distributed.barrier(group=_OVERLAY_PG_NCCL)
        _overlay_debug("sender barrier(nccl) done")
        _overlay_timing("redeploy/sender_barrier", time.perf_counter() - sender_barrier_t)
        _overlay_timing("redeploy/sender_total", time.perf_counter() - redeploy_total_t)
        if _get_bool_env("ELASTIC_OVERLAY_SENDER_EXIT", True):
            return "sender_exit"
        return True

    # receiver
    total_tensors = sum(
        len(opt_info.optimizer_tensors)
        for opt_info in training_state.optimizer_tensor_info_list
    )
    _overlay_debug(f"receiver recv begin: total_tensors={total_tensors}")
    recv_t = time.perf_counter()
    for opt_info in training_state.optimizer_tensor_info_list:
        opt_info.rebuild()
        for t in opt_info.optimizer_tensors:
            torch.distributed.recv(t, src=peer_overlay_rank, group=_OVERLAY_PG_NCCL)
    _overlay_debug("receiver recv done")
    _overlay_timing("redeploy/receiver_recv_tensors", time.perf_counter() - recv_t)
    _overlay_timing(
        "summary/group_init_to_transfer_done",
        time.perf_counter() - group_init_to_transfer_t,
    )

    # Apply optimizer/scheduler state (hyperparams) and sync model weights from main params.
    apply_state_t = time.perf_counter()
    if meta_obj is not None:
        if optimizer is not None and meta_obj.get("optimizer_param_groups") is not None:
            update_optimizer_by_state_dict(optimizer, meta_obj["optimizer_param_groups"])
        if opt_param_scheduler is not None and meta_obj.get("scheduler_state") is not None:
            load_state_dict_with_no_step(opt_param_scheduler, meta_obj["scheduler_state"])
        if hasattr(args, "consumed_train_samples") and meta_obj.get("consumed_train_samples") is not None:
            args.consumed_train_samples = meta_obj["consumed_train_samples"]
        if hasattr(args, "skipped_train_samples") and meta_obj.get("skipped_train_samples") is not None:
            args.skipped_train_samples = meta_obj["skipped_train_samples"]
        if hasattr(args, "num_floating_point_operations_so_far") and meta_obj.get(
            "num_floating_point_operations_so_far"
        ) is not None:
            args.num_floating_point_operations_so_far = meta_obj[
                "num_floating_point_operations_so_far"
            ]
    _overlay_timing("redeploy/apply_meta_state", time.perf_counter() - apply_state_t)

    update_weight_t = time.perf_counter()
    training_state.update_model_weight()
    _overlay_timing("redeploy/update_model_weight", time.perf_counter() - update_weight_t)
    _overlay_debug("receiver barrier(nccl) begin")
    receiver_barrier_t = time.perf_counter()
    torch.distributed.barrier(group=_OVERLAY_PG_NCCL)
    _overlay_debug("receiver barrier(nccl) done")
    _overlay_timing("redeploy/receiver_barrier", time.perf_counter() - receiver_barrier_t)
    _overlay_timing("redeploy/receiver_total", time.perf_counter() - redeploy_total_t)
    sender_iter = int(meta_obj.get("iteration", redeploy_iter)) if meta_obj is not None else redeploy_iter
    return {"receiver_next_iteration": sender_iter + 1}


def sync_training_state(
    redeploy_result: Any,
    *,
    args,
    get_num_microbatches_fn: Callable[[], int],
) -> Optional[Tuple[int, int, float]]:
    """Apply receiver-side iteration jump after overlay state transfer.
    """
    if not isinstance(redeploy_result, dict) or redeploy_result.get("receiver_next_iteration") is None:
        return None
    next_iteration = int(redeploy_result["receiver_next_iteration"])
    args.curr_iteration = next_iteration
    num_microbatches = get_num_microbatches_fn()
    flops = (
        float(args.num_floating_point_operations_so_far)
        if hasattr(args, "num_floating_point_operations_so_far")
        else 0.0
    )
    return next_iteration, num_microbatches, flops
