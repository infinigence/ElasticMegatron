#!/bin/bash

# Runs a small MoE model on 8 GPUs (adapted for ElasticMegatron + Megatron 0.16).
# For the original 8-GPU Mixtral recipe see git history.
set -ex

export PYTHONHASHSEED=1234
export TORCH_MANUAL_SEED=1234
BASE_PATH=${BASE_PATH:-/workspace}
MEGATRON_PATH=${MEGATRON_PATH:?'MEGATRON_PATH is not set. Set it to your Megatron-LM directory.'}
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH:-${_SCRIPT_DIR}:${MEGATRON_PATH}}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Short NCCL timeout so hangs surface quickly.
# Bump this for long real runs.
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-30}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-30}
TIMEOUT_AFTER_INIT_SEC=${TIMEOUT_AFTER_INIT_SEC:-30}

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 8-GPU friendly defaults. EP=2 by default; TPE must stay 1
# (ElasticMegatron's MoE resharding currently supports TPE==1 only).
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-2}
CP=${CP:-1}
TPE=${TPE:-1}

TRAIN_ITERS=${TRAIN_ITERS:-4}
NUM_EXPERTS=${NUM_EXPERTS:-8}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQ_LEN:-1024}
    --max-position-embeddings ${SEQ_LEN:-1024}
    --num-layers ${NUM_LAYERS:-8}
    --hidden-size ${HIDDEN_SIZE:-4096}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE:-11008}
    --num-attention-heads ${NUM_HEAD:-32}
    --init-method-std 0.008
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups ${NUM_QUERY_GROUP:-8}
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
)

DATA_ARGS=(
    --dataloader-type single
    --data-cache-path ${BASE_PATH}/data/data_cache
)
# ---------------------------------------------------------------------------
# 默认 --mock-data + NullTokenizer;runner 传 REAL_DATA_ARGS 时切真实数据。
# ---------------------------------------------------------------------------
if [[ -n "${REAL_DATA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    DATA_ARGS+=(${REAL_DATA_ARGS})
else
    DATA_ARGS+=(
        --tokenizer-type NullTokenizer
        --vocab-size 32000
        --mock-data
    )
fi

TRAINING_ARGS=(
    --micro-batch-size ${MBS:-1}
    --global-batch-size ${GBS:-8}
    --lr 1e-4
    --train-iters ${TRAIN_ITERS}
    --lr-decay-iters 1000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 2
    --clip-grad 1.0
    --bf16
    --distributed-timeout-minutes 1
    --seed 1234
)

TRAINING_ARGS+=(
    --distributed-timeout-seconds-after-init ${TIMEOUT_AFTER_INIT_SEC}
)

# CPU_OFFLOAD=1 builds Megatron's HybridDeviceOptimizer (CPU+GPU mixed optimizer
# state); requires the precision-aware optimizer code path. OFFLOAD_FRACTION is the
# fraction of GPU optimizer-state numel pushed to CPU. Default unset → plain GPU Adam.
if [ "${CPU_OFFLOAD:-0}" = "1" ]; then
    TRAINING_ARGS+=(
        --optimizer-cpu-offload
        --optimizer-offload-fraction ${OFFLOAD_FRACTION:-0.5}
        --use-precision-aware-optimizer
    )
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size ${TPE}
    --num-distributed-optimizer-instances ${NUM_DIST_OPT:-1}
    --use-distributed-optimizer
)

RECOMPUTE_ARGS=()

LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL:-1}
    --eval-interval ${EVAL_INTERVAL:-1000}
    --eval-iters ${EVAL_ITERS:-0}
    --no-load-optim
    --no-load-rng
)

SRC=${MEGATRON_PATH}/pretrain_gpt.py
export LOG_DIR=${LOG_DIR:-${BASE_PATH}/log}
LOG_PATH=${LOG_DIR}/moe_node${NODE_RANK}.log
mkdir -p ${LOG_DIR}

CMD="torchrun \
       ${DISTRIBUTED_ARGS[@]} \
       ${SRC} \
       ${MODEL_ARGS[@]} \
       ${MOE_ARGS[@]} \
       ${DATA_ARGS[@]} \
       ${TRAINING_ARGS[@]} \
       ${MODEL_PARALLEL_ARGS[@]} \
       ${RECOMPUTE_ARGS[@]} \
       ${LOGGING_ARGS[@]}"

echo ${CMD} | tee ${LOG_PATH}
${CMD} 2>&1 | tee -a ${LOG_PATH}
