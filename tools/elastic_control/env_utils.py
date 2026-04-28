import os
import torch
import socket
import subprocess
import shlex
from multiprocessing import shared_memory
from pathlib import Path
from multiprocessing import Process, sys
from megatron.training.global_vars import get_args
from elastic_megatron.transfer.ipc_manager import (
    receive_training_state,
    _map_ipc_data,
    _apply_misc_state,
    send_training_state,
)

# node info format: [[ip, port, scale_down/scale_up], ...] to add or delete
NEW_NODE_IP_LIST = None

# ranks of deleted nodes in scale down
_DELETED_NODE_RANK = []

# shared memory for node/process ready signal
_SHM_HANDLE = None

# shared memory block used for elastic state transfer
_ELASTIC_SHM = None


def _parse_elastic_trigger_iters():
    """Parse elastic trigger iters from env"""
    scale_up_env = os.environ.get("ELASTIC_SCALE_UP_ITER", "2")
    scale_up_iter = int(scale_up_env) if scale_up_env != "" else None

    scale_down_env = os.environ.get("ELASTIC_SCALE_DOWN_ITER", "3000")
    scale_down_iter = int(scale_down_env) if scale_down_env != "" else None

    trigger_iters = set()
    if scale_up_iter is not None:
        trigger_iters.add(scale_up_iter)
    if scale_down_iter is not None:
        trigger_iters.add(scale_down_iter)

    return scale_up_iter, scale_down_iter, trigger_iters


def get_deleted_node_rank():
    global _DELETED_NODE_RANK
    return _DELETED_NODE_RANK


def set_deleted_node_rank():
    global _DELETED_NODE_RANK
    #sym:set_deleted_node_rank
    _DELETED_NODE_RANK = [8,9,10,11,12,13,14,15]
    # _DELETED_NODE_RANK = [2,3]


def set_new_node_ip_list():
    global NEW_NODE_IP_LIST
    #sym:set_new_node_ip_list
    #sym:NEW_NODE_IP_LIST
    NEW_NODE_IP_LIST = [["10.204.30.217", 23455, 1]]

set_new_node_ip_list()
set_deleted_node_rank()


def get_shm_handle():
    global _SHM_HANDLE
    return _SHM_HANDLE


def set_shm_handle(value):
    global _SHM_HANDLE
    _SHM_HANDLE = value


def get_elastic_shm():
    global _ELASTIC_SHM
    return _ELASTIC_SHM


def set_elastic_shm(value):
    global _ELASTIC_SHM
    _ELASTIC_SHM = value


def _get_ready_flag():
    """
    Check if the new node is ready via shared memory.
    """
    shm_name = os.getenv("NEW_NODE_READY_SHM")
    if not shm_name:
        return 0

    current_shm = get_shm_handle()
    if current_shm is None or current_shm.name != shm_name:
        if current_shm is not None:
            try:
                current_shm.close()
            except Exception:
                pass
        try:
            set_shm_handle(shared_memory.SharedMemory(name=shm_name))
        except FileNotFoundError:
            return 0

    return get_shm_handle().buf[0]


def _reset_ready_flag():
    """Reset ready flag in shared memory (rank0 only)."""
    if not torch.distributed.is_initialized():
        return
    if torch.distributed.get_rank() != 0:
        return
    shm_name = ElasticEnv.get_ready_shm_name()
    if not shm_name:
        return
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.buf[0] = 0
    except FileNotFoundError:
        return


def get_new_node_ip_list():
    return NEW_NODE_IP_LIST


def trigger_new_node():
    """Launch scale-up new nodes through ssh from rank0."""
    nodes_info = get_new_node_ip_list()
    if not nodes_info or torch.distributed.get_rank() != 0:
        return

    ssh_user = os.getenv("ELASTIC_OVERLAY_REMOTE_USER")
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "8"))
    log_dir = Path(os.environ.get("LOG_DIR", "log"))
    log_dir.mkdir(parents=True, exist_ok=True)
    script_path = "run_e2e_demo.sh"
    workdir = str(Path(__file__).resolve().parents[2])
    venv_activate = os.environ.get(
        "ELASTIC_SCALEUP_VENV_ACTIVATE",
        os.environ.get("VENV_ACTIVATE", "/opt/venv/reason/bin/activate"),
    )
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = str(int(os.environ.get("MASTER_PORT", "6368")) + 1)
    nnodes = os.environ.get("ELASTIC_TARGET_NNODES", os.environ.get("NNODES", "2"))
    tp = os.environ.get("ELASTIC_TARGET_TP", os.environ.get("TP", "2"))
    pp = os.environ.get("ELASTIC_TARGET_PP", os.environ.get("PP", "8"))

    if not master_addr:
        raise RuntimeError("MASTER_ADDR is required for scale-up ssh launch")

    for idx, (node_addr, _node_port, node_rank) in enumerate(nodes_info):
        remote_target = f"{ssh_user}@{node_addr}" if ssh_user else str(node_addr)
        log_file = log_dir / f"new_megatron_scaleup_remote_{str(node_addr).replace('.', '_')}.log"
        remote_exports = [
            "export NEW_PROCESS=1",
            "export NEW_PROCESS_SCALE_UP=1",
            "export NEW_NODE=1",
            f"export MASTER_ADDR={shlex.quote(str(master_addr))}",
            f"export MASTER_PORT={shlex.quote(master_port)}",
            f"export GPUS_PER_NODE={gpus_per_node}",
            f"export NNODES={shlex.quote(str(nnodes))}",
            f"export TP={shlex.quote(str(tp))}",
            f"export PP={shlex.quote(str(pp))}",
            f"export NODE_RANK={int(node_rank)}",
            f"export RANK={int(node_rank)}",
        ]
        for key, value in os.environ.items():
            if key.startswith("ELASTIC_"):
                remote_exports.append(f"export {key}={shlex.quote(str(value))}")

        remote_cmd = (
            "bash -lc "
            + shlex.quote(
                f"cd {workdir} && "
                + f"[ -f {shlex.quote(venv_activate)} ] && source {shlex.quote(venv_activate)}; "
                + " && ".join(remote_exports)
                + f" && bash {script_path}"
            )
        )
        print(
            f"[elastic-ssh-scaleup] launch idx={idx} target={remote_target} "
            f"node_rank={node_rank} master={master_addr}:{master_port} "
            f"nnodes={nnodes} tp={tp} pp={pp}"
        )
        remote_log = open(log_file, "a")
        subprocess.Popen(
            ["ssh", remote_target, remote_cmd],
            stdout=remote_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return


class ElasticMode:
    SCALE_UP = 1
    SCALE_DOWN = 0

    @classmethod
    def get_mode_from_strategy(cls, current_strategy, target_strategy):
        def _get_ws(strategy):
            if isinstance(strategy, dict):
                return strategy.get("world_size")
            elif hasattr(strategy, "world_size"):
                return strategy.world_size

        cur_ws = _get_ws(current_strategy)
        tgt_ws = _get_ws(target_strategy)

        if tgt_ws > cur_ws:
            return cls.SCALE_UP
        elif tgt_ws < cur_ws:
            return cls.SCALE_DOWN
        else:
            return None


class ElasticEnv:
    """Utility class to handle environment variables related to elastic training."""

    VAR_SCALE_UP = "NEW_PROCESS_SCALE_UP"
    VAR_SCALE_DOWN_SENDER = "NEW_PROCESS_SCALE_DOWN_SENDER"
    VAR_SCALE_DOWN_RECEIVER = "NEW_PROCESS_SCALE_DOWN_RECEIVER"
    VAR_NEW_PROCESS = "NEW_PROCESS"
    VAR_NEW_NODE = "NEW_NODE"
    VAR_READY_SHM = "NEW_NODE_READY_SHM"
    VAR_COUNTER_SHM = "NEW_NODE_COUNTER_SHM"

    @classmethod
    def is_scale_up(cls) -> bool:
        return os.getenv(cls.VAR_SCALE_UP) is not None

    @classmethod
    def is_scale_down(cls) -> bool:
        return os.getenv(cls.VAR_SCALE_DOWN_SENDER) is not None

    @classmethod
    def is_scale_down_receiver(cls) -> bool:
        return os.getenv(cls.VAR_SCALE_DOWN_RECEIVER) is not None

    @classmethod
    def is_new_process(cls) -> bool:
        return os.getenv(cls.VAR_NEW_PROCESS) is not None

    @classmethod
    def is_new_node(cls) -> bool:
        return os.getenv(cls.VAR_NEW_NODE) is not None

    @classmethod
    def get_ready_shm_name(cls) -> str:
        return os.getenv(cls.VAR_READY_SHM)

    @classmethod
    def get_counter_shm_name(cls) -> str:
        return os.getenv(cls.VAR_COUNTER_SHM)

    @classmethod
    def clear_new_node(cls):
        if cls.VAR_NEW_NODE in os.environ:
            os.environ.pop(cls.VAR_NEW_NODE)

    @classmethod
    def clear_new_process_env(cls):
        for var in [
            cls.VAR_NEW_PROCESS,
            cls.VAR_SCALE_UP,
            cls.VAR_SCALE_DOWN_SENDER,
            cls.VAR_SCALE_DOWN_RECEIVER,
        ]:
            if var in os.environ:
                os.environ.pop(var)

    @classmethod
    def clear_scale_up(cls):
        if cls.VAR_SCALE_UP in os.environ:
            os.environ.pop(cls.VAR_SCALE_UP)

    @classmethod
    def is_normal_train(cls) -> bool:
        return not cls.is_scale_up() and not cls.is_new_node()


def _get_zmq_ports(base_port=None):
    """
    Get the ZMQ port number for IPC communication.
        1. If base_port is provided, use it as the base port.
        2. If base_port is None, read the base port from the environment variable
        "ELASTIC_IPC_PORT", defaulting to 6000 if the variable is not set.
        3. The final port number is calculated by adding the local rank (from
        environment variable "LOCAL_RANK", defaulting to 0) to the base port.
    """
    if base_port is None:
        base = int(os.getenv("ELASTIC_IPC_PORT", 6000))
    else:
        base = int(base_port)
    # Calculate the port number based on the base port and local rank
    dev = torch.cuda.current_device()
    return base + dev


def _exec_wrapper(cmd, args, env, log_file):
    """
    Execute a command with redirected stdout and stderr to a log file.
    """
    with open(log_file, "w") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())
    os.execvpe(cmd, args, env)


def _start_new_megatron_proc(scale_action, redeploy_script_path=None, node1_ip=None):
    """
    start new megatron process for scale up or scale down:
        1. create shared memory as ready signal
        2. set necessary envs for new megatron process
        3. start new megatron process
        4. for scale down, start two new megatron processes
        5. log new megatron process output to log files

    If redeploy_script_path is provided, use a dedicated redeployment path
    and start one 8-GPU training process from the provided script.
    If node1_ip is provided together with redeploy_script_path, also start
    the same script on node1 with REDEPLOY=1 via ssh.
    """
    pass_env_list = [
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
    ]
    new_env = {key: os.environ[key] for key in pass_env_list if key in os.environ}
    for key, value in os.environ.items():
        if key.startswith("ELASTIC_"):
            new_env[key] = value

    shm = shared_memory.SharedMemory(create=True, size=1)
    shm.buf[0] = 0
    new_env["NEW_PROCESS"] = "1"

    # old &new training process record the shared memory name as ready signal
    os.environ["NEW_NODE_READY_SHM"] = shm.name
    new_env["NEW_NODE_READY_SHM"] = shm.name

    if scale_action == ElasticMode.SCALE_DOWN:
        scale_down_ipc_base = int(os.environ.get("ELASTIC_IPC_PORT_SCALE_DOWN", "7000"))
        os.environ["ELASTIC_IPC_PORT"] = str(scale_down_ipc_base)
        new_env["ELASTIC_IPC_PORT"] = str(scale_down_ipc_base)

        new_env_proc_2 = {
            key: os.environ[key] for key in pass_env_list if key in os.environ
        }
        for key, value in os.environ.items():
            if key.startswith("ELASTIC_"):
                new_env_proc_2[key] = value
        new_env_proc_2["ELASTIC_IPC_PORT"] = str(scale_down_ipc_base)
        new_env_proc_2["NEW_NODE_READY_SHM"] = shm.name
        # for scale down, two process are needed to cover the communication build time cost,
        # therefore a shared memory with two bytes is created to record the ready status of two processes
        shm_1 = shared_memory.SharedMemory(create=True, size=2)
        shm_1.buf[0] = 0
        shm_1.buf[1] = 0
        new_env["NEW_NODE_COUNTER_SHM"] = shm_1.name
        new_env_proc_2["NEW_NODE_COUNTER_SHM"] = shm_1.name

        new_env_proc_2["NEW_PROCESS_SCALE_DOWN_RECEIVER"] = "1"

    set_elastic_shm(shm)

    log_dir = Path(os.environ.get("LOG_DIR"))
    log_dir.mkdir(parents=True, exist_ok=True)

    rank = os.environ.get("RANK", "unknown")
    hostname = socket.gethostname()
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    log_path = log_dir / f"new_megatron_{hostname}_rank{rank}_local{local_rank}.log"
    log_path_2 = (
        log_dir / f"new_megatron_proc2_{hostname}_rank{rank}_local{local_rank}.log"
    )

    # Redeployment
    if redeploy_script_path:
        new_env["NEW_PROCESS_REDEPLOY"] = "1"
        new_env["REDEPLOY"] = "0"
        new_env["NNODES"] = "1"
        new_env["MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT")) + 3)
        # hardcode for 8-GPU redeployment
        new_env["GPUS_PER_NODE"] = "8"

        script_path = str(Path(redeploy_script_path).expanduser())
        log_path_redeploy = (
            log_dir / f"new_megatron_redeploy_{hostname}_rank{rank}_local{local_rank}.log"
        )
        p_redeploy = Process(
            target=_exec_wrapper,
            args=("bash", ["bash", script_path], new_env, log_path_redeploy),
        )
        p_redeploy.start()

        if node1_ip:
            ssh_user = os.getenv("ELASTIC_OVERLAY_REMOTE_USER")
            remote_target = f"{ssh_user}@{node1_ip}" if ssh_user else node1_ip
            remote_workdir = str(Path(script_path).resolve().parent)
            remote_script = script_path
            remote_cmd = (
                "bash -lc "
                + shlex.quote(
                    f"cd {remote_workdir} && "
                    f"export REDEPLOY=1 && "
                    f"bash {shlex.quote(remote_script)}"
                )
            )
            remote_log_path = log_dir / (
                f"new_megatron_redeploy_remote_{str(node1_ip).replace('.', '_')}.log"
            )
            remote_log = open(remote_log_path, "a")
            subprocess.Popen(
                ["ssh", remote_target, remote_cmd],
                stdout=remote_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return

    # all existing nodes start process 2 for scale down (using communication group before scale down)
    if scale_action == ElasticMode.SCALE_DOWN:
        from megatron.training.training import get_parallel_strategy_list

        gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "8"))
        node_rank = int(os.environ.get("RANK", "0")) // gpus_per_node

        new_env["NEW_PROCESS_SCALE_DOWN_SENDER"] = "1"
        new_env["GPUS_PER_NODE"] = str(gpus_per_node)
        new_env["RANK"] = str(node_rank)
        new_env["NODE_RANK"] = str(node_rank)

        new_env["MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT")) + 1)
        p2 = Process(
            target=_exec_wrapper,
            args=("bash", ["bash", "run_e2e_demo.sh"], new_env, log_path_2),
        )
        p2.start()

        # only surviving nodes start process 1 for scale down (using communication group after scale down)
        if torch.distributed.get_rank() not in get_deleted_node_rank():
            new_env_proc_2["NEW_PROCESS_SCALE_DOWN_RECEIVER"] = "1"
            new_env_proc_2["GPUS_PER_NODE"] = str(gpus_per_node)
            new_env_proc_2["RANK"] = str(node_rank)
            new_env_proc_2["NODE_RANK"] = str(node_rank)
            new_env_proc_2["MASTER_PORT"] = str(
                int(os.environ.get("MASTER_PORT")) + 2
            )
            new_env_proc_2["ELASTIC_IPC_PORT"] = str(
                int(os.environ.get("ELASTIC_IPC_PORT", str(scale_down_ipc_base)))
                + torch.cuda.device_count()
            )

            new_env_proc_2["NNODES"] = str(os.environ.get("ELASTIC_TARGET_NNODES"))
            new_env_proc_2["PP"] = str(
                get_parallel_strategy_list()[1]["pipeline_model_parallel_size"]
            )
            new_env_proc_2["TP"] = str(
                get_parallel_strategy_list()[1]["tensor_model_parallel_size"]
            )
            p = Process(
                target=_exec_wrapper,
                args=("bash", ["bash", "run_e2e_demo.sh"], new_env_proc_2, log_path),
            )
            p.start()

    else:
        new_env["NEW_PROCESS_SCALE_UP"] = "1"
        new_env["GPUS_PER_NODE"] = os.environ.get("GPUS_PER_NODE")
        new_env["RANK"] = str(int(os.environ.get("RANK"))//int(new_env["GPUS_PER_NODE"]))
        

        new_env["MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT")) + 1)
        
        #sym:elastic_target_env
        new_env["NNODES"] = "2"
        new_env["PP"] = "8"
        new_env["TP"] = "2"
        p = Process(
            target=_exec_wrapper,
            args=("bash", ["bash", "run_e2e_demo.sh"], new_env, log_path),
        )
        p.start()


def _warmup_communication_groups():
    """Warm up NCCL communicators with tiny tensors before elastic resharding."""
    if not torch.distributed.is_initialized():
        return

    rank = torch.distributed.get_rank()

    dummy = torch.ones(1, device="cuda")

    # Warm up world group first.
    torch.distributed.all_reduce(dummy)

    try:
        from megatron.core import mpu

        # Warm up common Megatron communication groups.
        for getter_name in [
            "get_tensor_model_parallel_group",
            "get_pipeline_model_parallel_group",
            "get_data_parallel_group",
        ]:
            getter = getattr(mpu, getter_name, None)
            if getter is None:
                continue
            group = getter()
            if group is None:
                continue
            if torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(dummy, group=group)

        torch.distributed.barrier()

    except Exception as e:
        if rank == 0:
            print(f"[Elastic-Perf] warmup skipped due to error: {e}")


def _warmup_p2p_collective_groups():
    """Warm up 2-rank NCCL groups used by P2P-to-collective transfer."""
    print(f"process {os.getpid()} warming up p2p collective groups...")
    if not torch.distributed.is_initialized():
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return

    try:
        from elastic_megatron.distributed.elastic_process_group import (
            create_p2p_collective_groups,
            get_p2p_collective_group,
        )
    except Exception:
        return

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return

    create_p2p_collective_groups()

    dummy = torch.empty(1, device="cuda", dtype=torch.int32)
    for i in range(world_size):
        for j in range(i + 1, world_size):
            if rank != i and rank != j:
                continue
            group = get_p2p_collective_group(ranks=[i, j], backend="nccl")
            if rank == i:
                dummy.fill_(1)
            torch.distributed.broadcast(dummy, src=i, group=group.group)

    torch.distributed.barrier()


def _warmup_elastic_communication_before_ipc():
    import time

    if not torch.distributed.is_initialized():
        return

    rank = torch.distributed.get_rank()
    t0 = time.perf_counter()

    _warmup_communication_groups()

    _warmup_p2p_collective_groups()

    t1 = time.perf_counter()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def meta_resharding_for_inter_process(
    model_provider, model_type, checkpointing_context
):
    from multiprocessing import shared_memory
    from megatron.training.training import (
        setup_model_and_optimizer,
        get_parallel_strategy_list,
    )

    _warmup_elastic_communication_before_ipc()

    # Init signaling for scale_down_receiver (Process 2 Ready)
    if ElasticEnv.is_scale_down_receiver():
        counter_shm = shared_memory.SharedMemory(name=ElasticEnv.get_counter_shm_name())
        if torch.distributed.get_rank() == 0:
            counter_shm.buf[1] = 1

    # Step-1 : Build model(after sacle-up) in meta device
    from elastic_megatron import meta_device_context
    with meta_device_context():
        meta_model, meta_optimizer, meta_opt_param_scheduler = (
            setup_model_and_optimizer(
                model_provider, model_type, checkpointing_context=checkpointing_context
            )
        )

    # Step-2 : Init elastic megatron manager
    parallel_strategy_list = get_parallel_strategy_list()

    training_state = None

    if ElasticEnv.is_scale_down_receiver():
        parallel_strategy_list_1 = [parallel_strategy_list[0]]
        from elastic_megatron import ElasticMegatronManager
        meta_elastic_megatron_manager = ElasticMegatronManager(
            parallel_strategy_list_1,
            meta_model,
            meta_optimizer,
            meta_opt_param_scheduler,
        )
        training_state = meta_elastic_megatron_manager.state_manager.training_state
    else:
        from elastic_megatron import ElasticMegatronManager
        meta_elastic_megatron_manager = ElasticMegatronManager(
            parallel_strategy_list, meta_model, meta_optimizer, meta_opt_param_scheduler
        )

        # Step-3 : Scale-down/up in meta-device
        if ElasticEnv.is_scale_down():
            # prebuild nccl communicaiton for scale-down
            training_state = meta_elastic_megatron_manager.reshard(
                parallel_strategy_list[1], is_meta_device=True
            )
            training_state = meta_elastic_megatron_manager.reshard(
                parallel_strategy_list[0], is_meta_device=True
            )
        else:
            training_state = meta_elastic_megatron_manager.reshard(
                parallel_strategy_list[1], is_meta_device=True
            )
    args = get_args()
    consumed_samples_tensor = torch.tensor(0, dtype=torch.long, device="cuda")
    iter_idx_tensor = torch.tensor(0, dtype=torch.long, device="cuda")

    # Step-4 : If training_state is not None, recive the IPC data
    if training_state is not None and not ElasticEnv.is_scale_down_receiver():
        meta_model = training_state.model
        meta_optimizer = training_state.optimizer
        meta_opt_param_scheduler = training_state.opt_param_scheduler
        # IPC

        # Step-4.1 : Get the IPC data
        # 1. send signal to old-process to send ipc
        if torch.distributed.get_rank() == 0:
            # change ready flag
            shm = shared_memory.SharedMemory(name=ElasticEnv.get_ready_shm_name())

            if ElasticEnv.is_scale_down():
                counter_shm = shared_memory.SharedMemory(
                    name=ElasticEnv.get_counter_shm_name()
                )
                counter_shm.buf[0] = 1
                while True:
                    if counter_shm.buf[0] == 1 and counter_shm.buf[1] == 1:
                        shm.buf[0] = 1
                        break
            else:
                shm.buf[0] = 1

        # 2. receive ipc data from old-process
        state = receive_training_state()

        # Clear stale ready signal
        _reset_ready_flag()
        # 3. copy
        ipc_optimizer_tensors_list = _map_ipc_data(training_state, state)
        misc = state["misc"]
        _apply_misc_state(misc, args, meta_optimizer, meta_opt_param_scheduler)

        consumed_samples_tensor = torch.tensor(
            args.consumed_train_samples, dtype=torch.long, device="cuda"
        )
        iter_idx_tensor = torch.tensor(args.iteration, dtype=torch.long, device="cuda")

        # Step-4.2 : Replace the optimizer tensors with the IPC data
        params_to_resharding_metadata = training_state.params_to_resharding_metadata
        params = list(params_to_resharding_metadata.keys())
        for param in params:
            if params_to_resharding_metadata[param].optimizer_tensor_info is not None:
                params_to_resharding_metadata[
                    param
                ].optimizer_tensor_info.update_optimizer_tensors(
                    ipc_optimizer_tensors_list[param]
                )  # copy
        training_state.is_meta_device = False

    # Step-5 : Scale-up/Scale-down
    if ElasticEnv.is_scale_down():
        # Step-5.1  Clear new-model state in reshard-manager
        if not ElasticEnv.is_scale_down_receiver():
            meta_elastic_megatron_manager.clear_meta_state(parallel_strategy_list[1])

            # Step-5.2 : Scale-down (rebuild the cuda-device model/optimizer)
            training_state = meta_elastic_megatron_manager.reshard(
                parallel_strategy_list[1]
            )

            # deleted node exit
            if training_state is None:
                os._exit(0)
    elif not (ElasticEnv.is_scale_down() or ElasticEnv.is_scale_down_receiver()):
        # Step-5.1  Clear new-model state in reshard-manager
        meta_elastic_megatron_manager.clear_meta_state(parallel_strategy_list[0])

        # Step-5.2 : Scale-up (rebuild the cuda-device model/optimizer)
        training_state = meta_elastic_megatron_manager.reshard(
            parallel_strategy_list[0]
        )
        assert training_state is not None

    torch.distributed.broadcast(iter_idx_tensor, src=0)
    torch.distributed.broadcast(consumed_samples_tensor, src=0)

    args.iteration = iter_idx_tensor.item()
    args.curr_iteration = iter_idx_tensor.item()
    args.consumed_train_samples = consumed_samples_tensor.item()

    # Step-6 : Export training state to new process
    if ElasticEnv.is_scale_down() and not ElasticEnv.is_scale_down_receiver():
        send_training_state(
            training_state.model,
            training_state.optimizer,
            args.iteration,
            training_state.opt_param_scheduler,
            base_port=str(
                int(os.getenv("ELASTIC_IPC_PORT", "6000"))
                + int(torch.cuda.device_count())
            ),
            cur_parallel_strategy=parallel_strategy_list[1],
        )

    return (
        training_state.model,
        training_state.optimizer,
        training_state.opt_param_scheduler,
    )
