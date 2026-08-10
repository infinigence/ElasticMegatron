# ElasticMegatron

ElasticMegatron provides millisecond-level online parallel strategy switching and scaling capabilities for Megatron-LM, enabling dynamic reconfiguration of parallel configurations during training without interrupting the training process.

## Features

- **Dynamic Parallel Strategy Switching**: Seamlessly switch between different parallel configurations (TP, PP, DP, CP, EP, Group ZeRO) during training
- **Scaling Support**: Scale up or down the number of active GPUs/nodes dynamically
- **Group Zero Redundancy**: Support for redundant backup and removal of optimizer states (useful for fault recovery scenarios)
- **Two Deployment Modes**: 
  - **Intra-process switching**: For scenarios with fixed cluster topology (e.g., RL training)
  - **Inter-process switching**: For scenarios with dynamic cluster topology (e.g., fault recovery, opportunistic scheduling)

## Compatibility

ElasticMegatron is **version-agnostic** and does not depend on a specific Megatron-LM version. It has been tested and verified on multiple Megatron versions and can also work with custom Megatron implementations. 

**Recommended versions**: We recommend using Megatron-LM **0.11** or **0.13** for the best compatibility and performance.

## Usage
### Quick Start
Follow the steps below to quickly try ElasticMegatron:

1. Prepare the environment: ElasticMegatron uses the same runtime environment as Megatron-LM. For example, you can use `nvcr.io/nvidia/pytorch:24.03-py3`.

2. Clone the repositories:

```bash
mkdir dynatrain && cd dynatrain
git clone https://github.com/infinigence/ElasticMegatron.git
git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.11.0
```

3. Download `tokenizer.model` from `https://huggingface.co/meta-llama/Llama-2-7b/blob/main/tokenizer.model` and place it in the `dynatrain` directory.

4. Import `elastic_megatron` at the beginning of `Megatron-LM/pretrain_gpt.py`:

```python
"""Pretrain GPT."""
import elastic_megatron
```

5. Replace `Megatron-LM/megatron/training/training.py` with `ElasticMegatron/examples/intra_process/training_011.py`.

6. Set `BASE_PATH` in `ElasticMegatron/run_e2e_demo.sh` to `/path/to/dynatrain`.

7. Run `bash run_e2e_demo.sh`.

This example trains a tiny LLaMA 2 model on a mock dataset and switches parallel strategies sequentially according to `_PARALLEL_STRATEGY_LIST` in `training.py`.

### Examples

The `examples/` directory provides ready-to-use examples for both deployment modes:

- **Intra-process switching** (`examples/intra_process/`): Examples for switching parallel strategies within a single process
- **Inter-process switching** (`examples/inter_process/`): Examples for scaling across multiple processes

**Quick Start**: For Megatron-LM 0.11, you can directly replace the `training.py` file with `examples/intra_process/training_011.py` to enable elastic parallel strategy switching. The example file includes all necessary modifications and can serve as a drop-in replacement.

#### Example On RL Training

[RLinf](https://github.com/RLinf/RLinf) leverages ElasticMegatron's capabilities to build a dynamic scheduling system for RL training. By dynamically adjusting resources across components, the system maximizes GPU utilization and addresses the Rollout long-tail problem. Experiments demonstrate **30%~50% end-to-end performance improvement**.

- **Documentation**: [RLinf's dynamic scheduling module](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/scheduler/index.html)
- **Implementation**: [feat: add dynamic-scheduler in RLinf](https://github.com/RLinf/RLinf/pull/105)

### Basic Setup
**Important Import Note:**

- **Import `elastic_megatron` at the very beginning of your program (e.g., at the first import location in `pretrain_gpt.py`)**:

  ```python
  import elastic_megatron  # Must be imported before torch and megatron
  ```

- This ensures that ElasticMegatron can correctly take over and inject scaling and parallel switching capabilities, avoiding import order issues.


Then, register the model provider and dataset provider.

```python
from elastic_megatron import ElasticMegatronManager
from functools import partial

# Register setup functions
ElasticMegatronManager.register(
    setup_model_and_optimizer_func=partial(
        setup_model_and_optimizer,
        model_provider_func=model_provider,
        model_type=model_type
    ),
    train_valid_test_datasets_provider=train_valid_test_datasets_provider, # Optional
)
```

### Initialize ElasticMegatronManager

Create an instance of `ElasticMegatronManager` with a list of parallel strategies:

```python
# Define parallel strategy list
parallel_strategy_list = [
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
        "world_size": 8,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
    },
    # ... more strategies
]

# Initialize manager
elastic_megatron_manager = ElasticMegatronManager(
    parallel_strategy_list=parallel_strategy_list,
    model=model,
    optimizer=optimizer,
    opt_param_scheduler=opt_param_scheduler,
)
```

### Resharding During Training

Call the `reshard()` method to switch to a new parallel strategy:

```python
# During training loop
for iteration in range(train_iters):
    # ... training step ...

    # Check if resharding is needed
    if should_reshard(iteration):
        # Perform resharding
        training_state = elastic_megatron_manager.reshard(
            new_parallel_strategy=get_new_parallel_strategy(iteration),
            logger=logger,  # Optional logger
            is_meta_device=False,  # Set to True for meta device scenarios
            save_ckpt=False,  # Set to True to save checkpoints for verification
        )

        if training_state is not None:
            # Update model, optimizer, and scheduler
            model = training_state.model
            optimizer = training_state.optimizer
            opt_param_scheduler = training_state.opt_param_scheduler

            # Rebuild data iterators
            train_data_iterator, valid_data_iterator, _ = (
                elastic_megatron_manager.build_iterators()
            )

            # Refresh training config
            config = training_state.refresh_config(model, optimizer)
```


### Checkpoint Verification

To verify the correctness of resharding, you can save checkpoints before and after resharding:

```python
training_state = elastic_megatron_manager.reshard(
    new_parallel_strategy=new_parallel_strategy,
    save_ckpt=True,  # Save checkpoints for verification
)
```

This will save checkpoints before and after resharding, which can be used for offline verification.



## Implementation Details

The `reshard()` method performs the following steps:

1. **Generate source and destination Megatron states**: Determines the current and target parallel configurations
2. **Resharding and parameter transfer**: Transfers model parameters and optimizer states between configurations
   - Uses redundant backup for Group Zero scale-up/down scenarios
   - Uses direct parameter transfer for other scenarios
3. **Transfer learning rate**: Preserves optimizer learning rate schedule state
4. **Release source training state**: Frees memory from the previous configuration
5. **Update model weights**: Applies the new parallel configuration to the model
6. **Log communication info**: Reports transfer time and communication bandwidth

## Limitations and Future Work

- **MoE Models**: Currently only supports `TEP==1` (Tensor Expert Parallelism == 1).
- **CPU Optimizer**: CPU optimizer scenarios are not yet supported
- **VPP and hetero-pp**: VPP and hetero-pp support is planned
- **FSDP**: FSDP2 and megatron custom-FSDP support is planned

## Citation

If you find DynaTrain helpful, please cite the paper:

```bibtex
@misc{wang2026dynatrainfastonlineparallelism,
      title={DynaTrain: Fast Online Parallelism Switching for Elastic LLM Training}, 
      author={Yuanqing Wang and Yuchen Zhang and Hao Lin and Junhao Hu and Chunyang Zhu and Quanlu Zhang and Boxun Li and Guohao Dai and Zhi Yang and Daning Cheng and Yunquan Zhang and Yu Wang},
      year={2026},
      eprint={2605.18815},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.18815}, 
}
```
