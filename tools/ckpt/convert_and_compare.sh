#!/bin/bash
set -e

# Paths
BASE_PATH=${BASE_PATH:-/workspace}
ELASTIC_MEGATRON_PATH="${BASE_PATH}/ElasticMegatron"
MEGATRON_PATH=${MEGATRON_PATH:-${BASE_PATH}/Megatron-LM}
export PYTHONPATH=${ELASTIC_MEGATRON_PATH}:${MEGATRON_PATH}

# Source parallel strategy settings
TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-1}

# Target parallel strategy settings
TARGET_TP=${TARGET_TP:-1}
TARGET_PP=${TARGET_PP:-2}
TARGET_EP=${TARGET_EP:-1}

# Checkpoint directories
LOAD_DIR="${BASE_PATH}/ElasticMegatron/tools/ckpt/before_reshard"
SAVE_DIR="${BASE_PATH}/ElasticMegatron/tools/ckpt/convert_output"

echo "----------------------------------------------------------------"
echo "Starting Checkpoint Conversion"
echo "From: TP=${TP}, PP=${PP}, EP=${EP}"
echo "To:   TP=${TARGET_TP}, PP=${TARGET_PP}, EP=${TARGET_EP}"
echo "Input:  ${LOAD_DIR}"
echo "Output: ${SAVE_DIR}"
echo "----------------------------------------------------------------"

python ${ELASTIC_MEGATRON_PATH}/tools/ckpt/run_convert_patch_loader.py \
  --load-tp ${TP} \
  --load-pp ${PP} \
  --load-ep ${EP} \
  -- \
  --saver mcore \
  --load-dir ${LOAD_DIR} \
  --save-dir ${SAVE_DIR} \
  --model-type GPT \
  --target-tensor-parallel-size ${TARGET_TP} \
  --target-pipeline-parallel-size ${TARGET_PP} \
  --target-expert-parallel-size ${TARGET_EP} \
  --loader-transformer-impl transformer_engine \
  --saver-transformer-impl transformer_engine \
  --megatron-path ${MEGATRON_PATH} \
  --true-vocab-size 32000

echo "----------------------------------------------------------------"
echo "Conversion Completed."
echo "----------------------------------------------------------------"

ITER=${ITER:-iter_0000005}
REFERENCE_DIR="${BASE_PATH}/ElasticMegatron/tools/ckpt/after_reshard/${ITER}"
OUTPUT_CKPT="${SAVE_DIR}/${ITER}"

CKPT_A="${REFERENCE_DIR}"
CKPT_B="${OUTPUT_CKPT}"

echo "Starting Verification"
echo "Comparing Reference: ${CKPT_A}"
echo "With Output:         ${CKPT_B}"

if [ ! -d "${CKPT_A}" ]; then
    echo "Warning: Reference directory ${CKPT_A} does not exist. Skipping comparison."
else
    python ${ELASTIC_MEGATRON_PATH}/tools/ckpt/compare_ckpt.py \
      "${CKPT_A}" \
      "${CKPT_B}" \
      --thresh 1e-3
    
    echo "Verification Finished."
fi
