#!/bin/bash
set -ex

export PYTHONHASHSEED=1234
export TORCH_MANUAL_SEED=1234

BASE_PATH=/workspace
MEGATRON_PATH=${MEGATRON_PATH:-${BASE_PATH}/Megatron-LM}
export PYTHONPATH=${BASE_PATH}/ElasticMegatron:${MEGATRON_PATH}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=${TP:-1}
PP=${PP:-1}

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}

NODE_RANK=${RANK:-0}
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


# Network size variables
export MODEL_SIZE=${MODEL_SIZE:-"tiny"}

if   [ ${MODEL_SIZE} == 7 ];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 13 ];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 70 ];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 130 ];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_QUERY_GROUP=8;  NUM_LAYERS=88; FFN_HIDDEN_SIZE=31232; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == "tiny" ]; then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=4; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


DROP_OUT=0.0
MAX_SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=4096


DATA_CACHE_PATH=${BASE_PATH}/data/data_cache
TOKENIZER_PATH=${BASE_PATH}/tokenizer.model
SAVE_PATH=${BASE_PATH}/log

SRC_PATH=${MEGATRON_PATH}/pretrain_gpt.py
export LOG_DIR=${LOG_DIR:-${BASE_PATH}/log}
LOG_PATH=${LOG_DIR}/node${NODE_RANK}.log
mkdir -p ${LOG_DIR}

LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

MBS=1
GBS=32

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --use-distributed-optimizer \
       "
       #  --sequence-parallel \
       #  --no-overlap-p2p-communication \
       #  --use-distributed-optimizer \
       # --context-parallel-size ${CP} \
       # --num-distributed-optimizer-instances 1\


NETWORK_SIZE_ARGS=" \
       --transformer-impl transformer_engine \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --norm-epsilon ${NORM_EPS} \
       --swiglu \
       --use-flash-attn \
       --disable-bias-linear \
       --untie-embeddings-and-output-weights \
       --use-rotary-position-embeddings \
       --no-masked-softmax-fusion \
       --no-position-embedding \
       --use-mcore-models \
       " 

LOGGING_ARGS="
       --log-throughput \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
       --micro-batch-size ${MBS} \
       --global-batch-size ${GBS} \
       --train-iters 100 \
       --log-interval 1 \
       --optimizer adam \
       "

RECOMPUTE_ARGS="
"


INITIALIZATION_ARGS=" \
       --seed 1234 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr 3e-4 \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.01 \
       --min-lr 3e-5 \
       "

CHECKPOINTING_ARGS=""

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --attention-softmax-in-fp32 \
       "

VALIDATION_ARGS=" \
       --eval-interval 10000 \
       --eval-iters 1 \
       --save ${SAVE_PATH} \
       --save-interval 1000000 \
       "

       # --data-path ${DATA_PATH} \
       # --split 98,2,0 \
DATA_ARGS=" \
       --mock-data \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 4 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --dataloader-type single \
       --data-cache-path ${DATA_CACHE_PATH} \
       "

CMD="${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${RECOMPUTE_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       "


echo ${CMD} | tee ${LOG_PATH}
${CMD} 2>&1 | tee -a ${LOG_PATH}