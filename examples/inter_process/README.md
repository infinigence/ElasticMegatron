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
    if iteration == 5:
        elastic_signal = True
    
    from tools.agent.env_utils import _get_ready_flag
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
    iapp_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()

    init_parallel_strategy_list()
    from elastic_megatron.transfer.ipc_manager import meta_resharding_for_inter_process
    if ElasticEnv.is_scale_up() or ElasticEnv.is_new_node() or ElasticEnv.is_scale_down():
        model, optimizer, opt_param_scheduler = meta_resharding_for_inter_process(model_provider, model_type, checkpointing_context)
    elif ElasticEnv.is_scale_down_2():
        from multiprocessing import shared_memory
        counter_shm = shared_memory.SharedMemory(name=ElasticEnv.get_counter_shm_name())
        
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, model_type, checkpointing_context=checkpointing_context)
        if torch.distributed.get_rank() == 0 and ElasticEnv.is_scale_down_2():
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
    has_triggered_scale = False
    scale_action = ElasticMode.SCALE_DOWN
    # scale_action = ElasticMode.SCALE_UP
    while iteration < args.train_iters:
        elastic_signal, ready_signal = check_reshard(iteration)
        if elastic_signal and not has_triggered_scale and not ElasticEnv.is_scale_up() and not ElasticEnv.is_new_node() and not ElasticEnv.is_new_process():
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                from elastic_megatron.transfer.ipc_manager import _start_new_megatron_proc
                _start_new_megatron_proc(scale_action)
                if torch.distributed.get_rank() == 0 and scale_action == ElasticMode.SCALE_UP:
                    trigger_new_node()
            has_triggered_scale = True
        if has_triggered_scale and ready_signal:
            from elastic_megatron.transfer.ipc_manager import send_training_state
            send_training_state(model, optimizer, iteration, opt_param_scheduler)

        if args.profile and torch.distributed.get_rank() in args.profile_ranks:
        ......
```

* If training is conducted using Megaron 0.11, you can directly replace `megatron/training/training.py` with `exmaples/inter_process/training_011.py`.