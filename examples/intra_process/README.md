* Usage

1. At the beginning of `pretrain_gpt.py`, import the elastic scaling project
- **Import `elastic_megatron` at the very beginning of your program (e.g., at the first import location in `pretrain_gpt.py`)**:
```python
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""
import elastic_megatron
...
```

2. Add the following three code sections to `megatron/training/training.py`, modify as needed.

2.1 Specify parallel strategy and resharding timing
```python
_PARALLEL_STRATEGY_LIST = None

def init_parallel_strategy_list():
    global _PARALLEL_STRATEGY_LIST
    assert _PARALLEL_STRATEGY_LIST is None
  
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
            "world_size": 2,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
        },
    ]

def get_parallel_strategy_list():
    global _PARALLEL_STRATEGY_LIST
    assert _PARALLEL_STRATEGY_LIST is not None
    return _PARALLEL_STRATEGY_LIST

def init_elastic_megatron_manager(model, optimizer, opt_param_scheduler):
    from elastic_megatron import ElasticMegatronManager
    init_parallel_strategy_list()
    return ElasticMegatronManager(get_parallel_strategy_list(), model, optimizer, opt_param_scheduler)

def check_reshard(iteration) -> Optional[dict[str, int]]:
    from elastic_megatron import ElasticMegatronManager
    ElasticMegatronManager.global_barrier_by_gloo()

    reshard_interval = 2
    if iteration > 0 and iteration % reshard_interval == 0:
        parallel_strategy_list = get_parallel_strategy_list()
        return parallel_strategy_list[(iteration // reshard_interval) % len(parallel_strategy_list)]
    return None
```

2.2 Register elastic scaling at the beginning of `pretrain()`
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
    ...
```

2.3 Call elastic scaling in `train()` as needed
Original code:
```python
def train(...):
    ...
    while iteration < args.train_iters:
        # training code
```
Modified code:
```python
def train(...):
    ...
    elastic_megatron_manager = init_elastic_megatron_manager(model, optimizer, opt_param_scheduler)
    is_running = True
    while iteration < args.train_iters:
        # Check and execute resharding
        new_parallel_strategy = check_reshard(iteration)
        if new_parallel_strategy is not None:
            training_state = elastic_megatron_manager.reshard(new_parallel_strategy)
            if training_state is not None:
                # For re-start process, start the timer
                if not is_running:
                    timers('interval-time', log_level=0).start(barrier=False)
                
                # Update model, optimizer, opt_param_scheduler
                model = training_state.model
                optimizer = training_state.optimizer
                opt_param_scheduler = training_state.opt_param_scheduler
                
                # Update dataloader
                train_data_iterator, valid_data_iterator, _ = elastic_megatron_manager.build_iterators()

                # Update config
                config = training_state.refresh_config(model,optimizer)

                num_microbatches = get_num_microbatches()
                is_running = True
            else:
                is_running = False
                if timers('interval-time')._started:
                    timers('interval-time').stop()
        
        if not is_running:
            args.curr_iteration = iteration
            iteration += 1
            continue

        # training code
```


* If using Megatron 0.11 for training, you can directly replace `megatron/training/training.py` with `examples/intra_process/training_011.py`.
* If using Megatron 0.16 for training, you can directly replace `megatron/training/training.py` with `examples/intra_process/training_016.py`. It is a snapshot of a working `training.py` after applying the ElasticMegatron patches required for 0.16, including:
  - `ELASTIC_ENABLED` / `ELASTIC_STRATEGY_MODE` / `ELASTIC_RESHARD_INTERVAL` env-var driven `init_parallel_strategy_list` / `check_reshard` / `init_elastic_megatron_manager`.
  - `ELASTIC_SAVE_CKPT=1` hook inside `train()` that saves before/after-reshard ckpts for offline verification (see `tools/ckpt/verify_all.sh`).
  - `model[:] = training_state.model` slice-assignment inside the elastic loop so `pretrain()`'s `model` reference stays live for the post-train `evaluate_and_print_results` (avoids `setStorage size 0` on the embedding weight after reshard).