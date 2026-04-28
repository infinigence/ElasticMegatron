* README (training.py in Megatron training)

1. At the beginning of `pretrain_gpt.py`, import our elastic scaling project
- **Import `elastic_megatron` at the very beginning of your program (e.g., at the first import location in `pretrain_gpt.py`)**:
```python
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""
import elastic_megatron
...
```

2. Add the following code segments in  `megatron/training/training.py`and modify them as needed
* If training is conducted using Megaron 0.11, you can directly replace `megatron/training/training.py` with `exmaples/inter_process/training_011.py`.

2.1 Initialize the parallel strategy. The first one is the current one (the one before resharding), and the second one is the one after resharding.
```python
from tools.agent.env_utils import trigger_new_node,ElasticEnv,ElasticMode
_PARALLEL_STRATEGY_LIST = None

def init_parallel_strategy_list():
    global _PARALLEL_STRATEGY_LIST
    if _PARALLEL_STRATEGY_LIST is not None:
        return
  
    args = get_args()
    _PARALLEL_STRATEGY_LIST = [
        {
            "world_size": args.world_size,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "context_parallel_size": args.context_parallel_size,
            "num_distributed_optimizer_instances": args.num_distributed_optimizer_instances,
            "expert_model_parallel_size": args.expert_model_parallel_size,
            "expert_tensor_parallel_size": args.expert_tensor_parallel_size,
            "sequence_parallel": args.sequence_parallel,
        },
        {
            #sym:init_parallel_strategy_list
            "world_size": 4,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
        },
    ]
```

2.2 Specify the time points at which resharding occur.
```python
def check_reshard(iteration):
    elastic_signal = False
    ready_signal = torch.tensor([0], dtype=torch.int, device="cuda")
    from tools.elastic_control.env_utils import _parse_elastic_trigger_iters

    _, _, trigger_iters = _parse_elastic_trigger_iters()

    if iteration in trigger_iters:
        elastic_signal = True
    
    from tools.elastic_control.env_utils import _get_ready_flag
    if torch.distributed.get_rank() == 0:
        if _get_ready_flag() == 1:
            ready_signal[0] = 1
        else:
            ready_signal[0] = 0

    # broadcast ready signal to all ranks from rank 0
    torch.distributed.broadcast(ready_signal, src=0)

    return elastic_signal, ready_signal.item() == 1


```

2.3 Register the logic for resharding and replacing the `setup model` function at the beginning of `pretrain()`.
```python
def pretrain(
    ...
    ):
    from functools import partial
    from elastic_megatron import ElasticMegatronManager
    ElasticMegatronManager.register(
        setup_model_and_optimizer_func=partial(setup_model_and_optimizer, model_provider_func=model_provider, model_type=model_type),
        train_valid_test_datasets_provider=train_valid_test_dataset_provider,
    )

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
    ...

    # Modify at the place where the model application is made:
        app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()

    init_parallel_strategy_list()
    from tools.elastic_control.env_utils import meta_resharding_for_inter_process
    if ElasticEnv.is_scale_up() or ElasticEnv.is_new_node() or ElasticEnv.is_scale_down():
        model, optimizer, opt_param_scheduler = meta_resharding_for_inter_process(model_provider, model_type, checkpointing_context)
    elif ElasticEnv.is_scale_down_receiver():
        from multiprocessing import shared_memory
        counter_shm = shared_memory.SharedMemory(name=ElasticEnv.get_counter_shm_name())
        
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, model_type, checkpointing_context=checkpointing_context)
        if torch.distributed.get_rank() == 0 and ElasticEnv.is_scale_down_receiver():
            counter_shm.buf[1] = 1
            print("[PROCESS 2] READY")
        from elastic_megatron.transfer.ipc_manager import receive_training_state,_attach_state_to_model
        state = receive_training_state()
        _attach_state_to_model(state, model, optimizer, opt_param_scheduler, get_parallel_strategy_list()[0])
    else:
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, model_type, checkpointing_context=checkpointing_context)



```
2.4 Add the code for inter-process communication during the training process of `train()`.
```python
    ......
    # Run training iterations till done.
    overlay_enabled = _overlay_enabled()
    has_triggered_scale = False
    triggered_scale_iters = set()
    scale_up_iter = None
    scale_down_iter = None
    # scale_action = ElasticMode.SCALE_DOWN
    #sym:scale_action
    scale_action = ElasticMode.SCALE_DOWN

    if not overlay_enabled:
        if (
            ElasticEnv.is_new_process()
            or ElasticEnv.is_new_node()
            or ElasticEnv.is_scale_down_receiver()
        ):
            triggered_scale_iters.add(iteration)

        from tools.elastic_control.env_utils import _parse_elastic_trigger_iters

        scale_up_iter, scale_down_iter, _ = _parse_elastic_trigger_iters()
    while iteration < args.train_iters:
        if overlay_enabled:
            if not is_overlay_new_process():
                launch_async_overlay_processes(iteration)
                update_async_overlay_ready_state()

            if sender_proxy_redeploy():
                overlay_redeploy_result = redeploy_transfer(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    local_trigger_iter=iteration,
                )
                if overlay_redeploy_result == "sender_exit":
                    should_exit = True
                    exit_code = 0
                    break

            if ipc_sender(iteration, model, optimizer, opt_param_scheduler):
                should_exit = True
                exit_code = 0
                break
        else:
            elastic_signal, ready_signal = check_reshard(iteration)
            active_scale_action = scale_action
            if scale_up_iter is not None and iteration == scale_up_iter:
                active_scale_action = ElasticMode.SCALE_UP
            elif scale_down_iter is not None and iteration == scale_down_iter:
                active_scale_action = ElasticMode.SCALE_DOWN

            allow_new_node_trigger = active_scale_action == ElasticMode.SCALE_DOWN
            if (
                elastic_signal
                and iteration not in triggered_scale_iters
                and (allow_new_node_trigger or not ElasticEnv.is_new_node())
            ):
                if active_scale_action == ElasticMode.SCALE_DOWN:
                    os.environ["ELASTIC_IPC_PORT"] = os.environ.get(
                        "ELASTIC_IPC_PORT_SCALE_DOWN", "7000"
                    )
                else:
                    os.environ["ELASTIC_IPC_PORT"] = os.environ.get(
                        "ELASTIC_IPC_PORT", "6000"
                    )

                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    from tools.elastic_control.env_utils import _start_new_megatron_proc
                    _start_new_megatron_proc(active_scale_action)
                    if (
                        torch.distributed.get_rank() == 0
                        and active_scale_action == ElasticMode.SCALE_UP
                    ):
                        # Scale-up now launches new nodes through ssh from rank0.
                        trigger_new_node()
                has_triggered_scale = True
                triggered_scale_iters.add(iteration)
            if has_triggered_scale and ready_signal:
                from elastic_megatron.transfer.ipc_manager import send_training_state

                send_training_state(model, optimizer, iteration, opt_param_scheduler)
                has_triggered_scale = False


        if args.profile and torch.distributed.get_rank() in args.profile_ranks:
        ......
```


### 3. start training（Scale Up/Down）

#### Scale Down
```bash
MODE=scale_down \
    DOWN_SRC_TP=8 DOWN_SRC_PP=2 \
    DOWN_TGT_TP=2 DOWN_TGT_PP=4 \
    SCALE_DOWN_ITER=2 \
    /workpath/run_inter_process.sh
```

#### Scale Up
```bash
MODE=scale_up \
    UP_SRC_TP=4 UP_SRC_PP=2 \
    UP_TGT_TP=2 UP_TGT_PP=8 \
    SCALE_UP_ITER=2 \
    /workpath/run_inter_process.sh
```

#### Parameters

- `*_SRC_TP` / `*_SRC_PP`：Source Tensor/Pipeline Parallelism before scaling
- `*_TGT_TP` / `*_TGT_PP`：Target Tensor/Pipeline Parallelism after scaling
- `SCALE_*_ITER`：The training iteration at which scaling is triggered