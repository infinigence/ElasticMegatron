#!/bin/bash
set -ex

# Llama2-style DENSE training launcher (Megatron 0.16, pretrain_gpt.py).
# Model size via MODEL_SIZE (tiny/medium/7/13/70/130) + NUM_LAYERS; parallel layout via
# TP/PP/CP. Elastic reshard: set ELASTIC_ENABLED=1 + ELASTIC_STRATEGY_LIST (inline JSON
# override-dict list) or ELASTIC_STRATEGY_LIST_FILE, plus ELASTIC_RESHARD_INTERVAL.
# For Qwen3-30B-A3B and MoE (incl. small MoE smokes), use run_qwen3_30b.sh.

export PYTHONHASHSEED=1234
export TORCH_MANUAL_SEED=1234

BASE_PATH=${BASE_PATH:-/workspace}
MEGATRON_PATH=${MEGATRON_PATH:?'MEGATRON_PATH is not set. Set it to your Megatron-LM directory.'}
# Default PYTHONPATH to this script's directory (so ElasticMegatron-clean tests
# import the clean tree, not BASE_PATH/ElasticMegatron). Override by exporting
# PYTHONPATH before invoking.
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH:-${_SCRIPT_DIR}:${MEGATRON_PATH}}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Short NCCL timeout so hangs surface quickly.
# Bump this for long real runs.
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-30}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-30}
TIMEOUT_AFTER_INIT_SEC=${TIMEOUT_AFTER_INIT_SEC:-30}

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

TP=${TP:-1}
PP=${PP:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}

NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


# Network size variables
export MODEL_SIZE=${MODEL_SIZE:-"tiny"}

if   [ ${MODEL_SIZE} == 7 ];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 13 ];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 70 ];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 130 ];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_QUERY_GROUP=8;  NUM_LAYERS=88; FFN_HIDDEN_SIZE=31232; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == "medium" ]; then HIDDEN_SIZE=4096; NUM_HEAD=32; NUM_QUERY_GROUP=8; NUM_LAYERS=${NUM_LAYERS:-8}; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == "tiny" ]; then HIDDEN_SIZE=256;  NUM_HEAD=8; NUM_QUERY_GROUP=8; NUM_LAYERS=8; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


DROP_OUT=0.0
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}

# TIE_EMBED=1 enables tied input embedding / output layer (exercises the
# PP=1 + tied-embed bucket-layout simulation path in resharding_dp.py).
# Default unties so legacy tests keep their original behaviour.
if [ "${TIE_EMBED:-0}" = "1" ]; then
    TIE_EMBED_ARG=""
else
    TIE_EMBED_ARG="--untie-embeddings-and-output-weights"
fi
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-${MAX_SEQ_LEN}}


DATA_CACHE_PATH=${BASE_PATH}/data/data_cache
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

MBS=${MBS:-1}
GBS=${GBS:-32}

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --context-parallel-size ${CP:-1} \
       --num-distributed-optimizer-instances ${NUM_DIST_OPT:-1} \
       --use-distributed-optimizer \
       "
       #  --sequence-parallel \
       #  --no-overlap-p2p-communication \


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
       ${TIE_EMBED_ARG} \
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

# CPU_OFFLOAD=1 builds Megatron's HybridDeviceOptimizer (cpu-adam: CPU+GPU mixed
# optimizer state) to exercise the hybrid-adam reshard path. It requires the
# precision-aware optimizer path (Megatron asserts this).
# --overlap-cpu-optimizer-d2h-h2d enables the HDO D2H/H2D streams (Megatron defaults
# the offload fraction to 1.0 = full offload, matching the slime 30B config).
# Default OFF — plain GPU Adam (the verify sweep + the GPU-adam perf baseline); set
# CPU_OFFLOAD=1 for the cpu-adam path.
if [ "${CPU_OFFLOAD:-0}" = "1" ]; then
    CPU_OFFLOAD_ARGS=" \
       --optimizer-cpu-offload \
       --overlap-cpu-optimizer-d2h-h2d \
       --use-precision-aware-optimizer \
       "
else
    CPU_OFFLOAD_ARGS=""
fi

# RERUN_MODE: Megatron's rerun_state_machine mode. Default validate_results re-runs
# each step to check bit-determinism — fundamentally incompatible with mid-training
# resharding (a reshard changes reduction order / replays a stateful step), so set
# RERUN_MODE=disabled for elastic runs that sweep many reshards.
if [ -n "${RERUN_MODE:-}" ]; then
    RERUN_ARG="--rerun-mode ${RERUN_MODE}"
else
    RERUN_ARG=""
fi

TRAIN_ITERS=${TRAIN_ITERS:-100}
TRAINING_ARGS=" \
       --micro-batch-size ${MBS} \
       --global-batch-size ${GBS} \
       --train-iters ${TRAIN_ITERS} \
       --log-interval 1 \
       --optimizer adam \
       ${CPU_OFFLOAD_ARGS} \
       ${RERUN_ARG} \
       --distributed-timeout-minutes 1 \
       --distributed-timeout-seconds-after-init ${TIMEOUT_AFTER_INIT_SEC} \
       "

RECOMPUTE_ARGS="
"

# Full recompute is required for 7B models to keep activation memory in bounds.
# Disabled by default; enable via RECOMPUTE_FULL=1.
# RECOMPUTE_LAYERS must be ≤ NUM_LAYERS/PP (layers per PP stage) — if it exceeds
# that, PP>1 reshard will IndexError. Default 8 is safe for PP up to 4 with 32 layers.
if [[ "${RECOMPUTE_FULL:-0}" == "1" ]]; then
RECOMPUTE_ARGS="
       --recompute-granularity full
       --recompute-method uniform
       --recompute-num-layers ${RECOMPUTE_LAYERS:-8}
"
fi


INITIALIZATION_ARGS=" \
       --seed 1234 \
       --init-method-std ${INIT_STD:-0.008} \
       "

LEARNING_RATE_ARGS=" \
       --lr ${LR:-5e-5} \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.1 \
       --min-lr ${MIN_LR:-5e-6} \
       "

CHECKPOINTING_ARGS=""

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --attention-softmax-in-fp32 \
       "

VALIDATION_ARGS=" \
       --eval-interval ${EVAL_INTERVAL:-10000} \
       --eval-iters ${EVAL_ITERS:-0} \
       "

       # --save ${SAVE_PATH} \
       # --save-interval 1000000 \

# ---------------------------------------------------------------------------
# DATA_ARGS:数据 + tokenizer 部分由 resolve_data_args.sh 按文件存在性自动判定
# (真实 dataset / --mock-data),其余 --seq-length / --num-workers /
# --dataloader-type / --data-cache-path 在这里拼上。判定优先级 / 可覆盖变量
# (DATA_DIR / TOKENIZER_DIR / DATA_PREFIX / ELASTIC_REQUIRE_REAL_DATA)见该文件。
# ---------------------------------------------------------------------------
source "${_SCRIPT_DIR}/resolve_data_args.sh"
DATA_ARGS=" \
       ${RESOLVED_DATA_ARGS} \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 4 \
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
       ${MOE_ARGS:-} \
       "


echo ${CMD} | tee ${LOG_PATH}
${CMD} 2>&1 | tee -a ${LOG_PATH}
