import pickle
import sys

import torch
import zmq
from megatron.training.global_vars import get_args
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from ..resharding.resharding_metadata import generate_optimizer_tensor_info


def set_zmq_ctx():
    global _ZMQ_CTX
    _ZMQ_CTX = zmq.Context.instance()


set_zmq_ctx()


def get_zmq_ctx():
    global _ZMQ_CTX
    return _ZMQ_CTX


def _make_cuda_tensor_info(tensor):
    """Return dict of kwargs acceptable by rebuild_cuda_tensor."""
    storage = tensor.storage()
    (
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()
    return {
        "dtype": tensor.dtype,
        "tensor_size": tuple(tensor.size()),
        "tensor_stride": tuple(tensor.stride()),
        "tensor_offset": tensor.storage_offset(),
        "storage_cls": type(storage),
        "storage_device": storage_device,
        "storage_handle": storage_handle,
        "storage_size_bytes": storage_size_bytes,
        "storage_offset_bytes": storage_offset_bytes,
        "requires_grad": tensor.requires_grad,
        "ref_counter_handle": ref_counter_handle,
        "ref_counter_offset": ref_counter_offset,
        "event_handle": event_handle,
        "event_sync_required": event_sync_required,
    }


def send_training_state(
    model,
    optimizer,
    iter_idx,
    scheduler=None,
    base_port=None,
    cur_parallel_strategy=None,
):
    """
    export the training state (to IPC)
        1. gather all model parameters, optimizer states, rng states, iteration index, scheduler
        states, and pack them into a payload
        2. send the payload to the IPC server via ZMQ
        3. exit the current process after exporting the state

    """
    import torch
    import zmq
    from megatron.training.training import get_parallel_strategy_list

    if cur_parallel_strategy is None:
        cur_parallel_strategy = get_parallel_strategy_list()[0]
    args = get_args()

    metadata_map = generate_optimizer_tensor_info(model, optimizer)

    param_to_name = {}
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            param_to_name[param] = name

    params = list(metadata_map.keys())

    share_dict = {}
    # send optimizer tensors (fp32/exp_avg/exp_avg_sq) for each param.
    # NOTE: this inter-process path still assumes Adam-shaped states (master +
    # the two moments). After the F1 state-model generalization the intra-process
    # path is state-agnostic; this path is not yet (no inter-process test covers
    # it). To support non-Adam optimizers here, iterate opt_info.states / .state_names
    # instead of the hard-coded triple below. See docs/project/optimizer_state_model.md.
    for p in params:
        opt_info = metadata_map[p]
        if opt_info is None:
            continue

        name = param_to_name.get(p)
        if not name:
            continue

        for tag, tensor in zip(
            ["fp32", "exp_avg", "exp_avg_sq"],
            [opt_info.main_weight, opt_info.exp_avg, opt_info.exp_avg_sq],
        ):
            share_dict[f"opt::{name}::{tag}"] = _make_cuda_tensor_info(tensor)

    #  misc（rng, iter, scheduler）
    misc = {
        "iter_idx": iter_idx,
        "consumed_samples": args.consumed_train_samples,
        "rng_cpu": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "optimizer_groups": [
            {"lr": g["lr"], "step": g.get("step", 0)} for g in optimizer.param_groups
        ],
        "cur_parallel_strategy": cur_parallel_strategy,
    }
    if misc["iter_idx"] == 0:
        misc["iter_idx"] = args.iteration

    payload = {"tensor_handles": share_dict, "misc": misc}
    from tools.agent.env_utils import _get_zmq_ports

    port = _get_zmq_ports(base_port=base_port)
    sock = get_zmq_ctx().socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")

    sock.send_pyobj(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.recv_string()

    sock.close()

    torch.distributed.destroy_process_group()
    sys.exit(0)


def receive_training_state():
    """
    receive the training state (from IPC)
    """
    from tools.agent.env_utils import _get_zmq_ports

    port = _get_zmq_ports()
    sock = get_zmq_ctx().socket(zmq.REP)

    sock.bind(f"tcp://*:{port}")

    payload = sock.recv_pyobj()
    sock.send_string("ack")

    handles = payload["tensor_handles"]
    rebuilt = {
        k: rebuild_cuda_tensor(torch.Tensor, **info) for k, info in handles.items()
    }
    payload["rebuilt_tensors"] = rebuilt
    sock.close()
    return payload


def _apply_misc_state(misc, args, optimizer, scheduler=None):
    """
    Apply misc state (rng, scheduler, optimizer groups, iteration) to the current process context.
    """
    torch.random.set_rng_state(misc["rng_cpu"])
    torch.cuda.set_rng_state(misc["rng_cuda"])

    if scheduler and misc["scheduler_state"]:
        scheduler.load_state_dict(misc["scheduler_state"])

    for g, info in zip(optimizer.param_groups, misc.get("optimizer_groups", [])):
        g["lr"] = info["lr"]
        if "step" in info:
            g["step"] = info["step"]

    args.iteration = misc["iter_idx"]
    args.curr_iteration = misc["iter_idx"]
    args.consumed_train_samples = misc["consumed_samples"]


def _attach_state_to_model(
    state, model, optimizer, scheduler=None, old_parallel_strategy=None
):
    """
    attach the received training state (from IPC) to model, optimizer, scheduler, and args
    """
    rebuilt = state["rebuilt_tensors"]
    args = get_args()

    metadata_map = generate_optimizer_tensor_info(model, optimizer)

    param_to_name = {}
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            param_to_name[param] = name

    params = list(metadata_map.keys())

    for p in params:
        opt_info = metadata_map[p]
        if opt_info is None:
            continue

        name = param_to_name.get(p)
        if not name:
            continue

        fp32_param = opt_info.main_weight
        fp32_param.data.copy_(rebuilt[f"opt::{name}::fp32"])

        # exp_avg & exp_avg_sq
        opt_info.exp_avg.copy_(rebuilt[f"opt::{name}::exp_avg"])
        opt_info.exp_avg_sq.copy_(rebuilt[f"opt::{name}::exp_avg_sq"])

    # rng / scheduler / iteration
    _apply_misc_state(state["misc"], args, optimizer, scheduler)

    optimizer._copy_main_params_to_model_params()

    # sync optimizer state to model
    optimizer.update_successful = True
    for model_chunk in optimizer.model_chunks:
        model_chunk.start_param_sync(force_sync=True)
    optimizer.update_successful = False


def _map_ipc_data(
    training_state, ipc_state
) -> dict[torch.nn.Parameter, list[torch.Tensor]]:
    rebuilt_tensors = ipc_state["rebuilt_tensors"]
    ipc_optimizer_tensors_list = {}
    for param, resharding_metadata in training_state.get_metadata().items():
        optimizer_tensor_info = resharding_metadata.optimizer_tensor_info
        if optimizer_tensor_info is not None:
            meta = resharding_metadata
            name = getattr(
                getattr(meta, "param_position_attr", None), "module_name", None
            )
            tensors = []
            for tag in ["fp32", "exp_avg", "exp_avg_sq"]:
                key = f"opt::{name}::{tag}"
                if key in rebuilt_tensors:
                    tensors.append(rebuilt_tensors[key].clone().detach())
                else:
                    raise KeyError(f"Missing key {key} in IPC state")
            ipc_optimizer_tensors_list[param] = tensors

    return ipc_optimizer_tensors_list
