# Checkpoint Verification Tools

This tool helps validate the correctness of ElasticMegatron resharding by converting the original checkpoint (`before_reshard`) to the target topology using Megatron-LM's converter, and comparing it element-wise with the actual resharded checkpoint (`after_reshard`).

## Usage Steps

### 1. Generate Checkpoints
Enable `save_ckpt=True` when calling `reshard` in your training code:
```python 
training_state = elastic_megatron_manager.reshard(new_parallel_strategy, save_ckpt=True)
```
The checkpoints will be automatically saved in:
- Before reshard: `.../ElasticMegatron/tools/ckpt/before_reshard`
- After reshard: `.../ElasticMegatron/tools/ckpt/after_reshard`

### 2. Run Conversion and Comparison
Use the `convert_and_compare.sh` script to perform the validation. This script automates the conversion of the `before_reshard` checkpoint to the target topology and compares it against `after_reshard`.

Set the environment variables for the source/target strategy and iteration, then run the script:

```bash
# Example: Validating reshard from TP=2/PP=1 to TP=1/PP=2
export TP=2 PP=1 EP=1                   # Source strategy (before_reshard)
export TARGET_TP=1 TARGET_PP=2 TARGET_EP=1 # Target strategy (after_reshard)
export ITER=iter_0000005                # Checkpoint iteration folder name

bash tools/ckpt/convert_and_compare.sh
```

**Parameters:**
- `TP`/`PP`/`EP`: Parallel strategy of the source checkpoint (`before_reshard`).
- `TARGET_TP`/`TARGET_PP`/`TARGET_EP`: Parallel strategy of the target checkpoint (`after_reshard`).
- `ITER`: The iteration directory name to verify (default: `iter_0000005`).

The script will output the conversion log and the final comparison result (Match/Mismatch).
