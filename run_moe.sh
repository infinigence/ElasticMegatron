#!/bin/bash

# Runs Mixtral 8x7B model

export PYTHONHASHSEED=1234
export TORCH_MANUAL_SEED=1234
BASE_PATH=/workspace
MEGATRON_PATH=${MEGATRON_PATH:-${BASE_PATH}/Megatron-LM}
export PYTHONPATH=${BASE_PATH}/ElasticMegatron:${MEGATRON_PATH}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-4}
CP=${CP:-1}
TPE=${TPE:-1}


TOKENIZER_MODEL=${BASE_PATH}/tokenizer.model

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
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 2
    --hidden-size 2048
    --ffn-hidden-size 768
    --num-attention-heads 32
    --init-method-std 0.008
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
)


DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --mock-data
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 32
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size ${TPE}
    --use-distributed-optimizer
    --sequence-parallel
)

RECOMPUTE_ARGS=(
       --recompute-granularity full
       --recompute-method block
       --recompute-num-layers 4
)

LOGGING_ARGS=(
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
    )
fi

SRC=${MEGATRON_PATH}/pretrain_gpt.py
export LOG_DIR=${LOG_DIR:-${BASE_PATH}/log}
LOG_PATH=${LOG_DIR}/node${NODE_RANK}.log
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