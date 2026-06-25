#!/bin/bash

# Runs Qwen3-30B-A3B (MoE: 128 experts, top-8) on a single 8-GPU node through
# MEGATRON_PATH/pretrain_gpt.py. Defaults reproduce Qwen3-30B-A3B exactly, but the
# arch/MoE knobs are overridable (NUM_LAYERS/NUM_EXPERTS/HIDDEN_SIZE/MOE_ROUTER_TOPK/...)
# so this script also serves smaller MoE smokes. By default this is pure Megatron
# (ELASTIC_ENABLED=0); setting ELASTIC_ENABLED=1 with a reshard sequence
# (ELASTIC_STRATEGY_LIST='[{}, {"world_size": 4}]' or
# ELASTIC_STRATEGY_LIST_FILE=examples/strategies/moe_30b.json) plus
# ELASTIC_RESHARD_INTERVAL exercises the TP4/EP4 DP2<->DP1 reshard path.
#
# Megatron args are transcribed from slime's source of truth:
#   slime-trainer/configs/models/qwen3-30B-A3B.sh   (architecture + MoE block)
#   + the slime 30B launcher's parallel dims TP4/PP1/CP1/EP4/ETP1 and training
#     shape (mbs1/gbs64/seq4096, full recompute, cpu-adam).
#
# Memory note: 30B on 8x A100-80GB does NOT fit GPU-resident Adam. cpu-adam
# offload (CPU_OFFLOAD=1) + full activation recompute (RECOMPUTE=1) are ON by
# default — this mirrors the slime 30B run. OMP_NUM_THREADS=14 keeps cpu-adam
# from being ~80x slow (S2-fix). Host peak ~1.2 TB for the full elastic run;
# pure pretrain is lighter but still wants a big-RAM box.
set -ex

export PYTHONHASHSEED=1234
export TORCH_MANUAL_SEED=1234
BASE_PATH=${BASE_PATH:-/workspace}
MEGATRON_PATH=${MEGATRON_PATH:?'MEGATRON_PATH is not set. Set it to your Megatron-LM(-custom) directory.'}
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH:-${_SCRIPT_DIR}:${MEGATRON_PATH}}

# Default: no elastic strategy switching. pretrain_gpt imports elastic_megatron at
# module top to patch parallel-state; ELASTIC_ENABLED=0 keeps it a passthrough.
export ELASTIC_ENABLED=${ELASTIC_ENABLED:-0}

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Post-init hang detection. NOTE: this is the *30B* launcher — a cpu-adam 30B
# iteration is ~70-160 s and a reshard transfer is tens of seconds, so the
# repo-wide ~30 s dev default (which surfaces hangs fast on tiny smoke tests)
# produces FALSE timeouts here. Default to values that clear a real 30B step
# (override down for fast hang-surfacing on a reduced model).
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-600}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}
TIMEOUT_AFTER_INIT_SEC=${TIMEOUT_AFTER_INIT_SEC:-600}

# cpu-adam (HybridDeviceOptimizer) is CPU-thread bound; 14 threads matches the
# slime 30B config (too few -> ~80x slower / host OOM).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Parallel layout: slime 30B = TP4/PP1/CP1/EP4/ETP1 on 8 GPUs (=> dense DP=2,
# expert grid EP=4 x expert-DP=2). TPE (expert-tensor-parallel) must stay 1.
TP=${TP:-4}
PP=${PP:-1}
EP=${EP:-4}
CP=${CP:-1}
TPE=${TPE:-1}

TRAIN_ITERS=${TRAIN_ITERS:-4}
NUM_EXPERTS=${NUM_EXPERTS:-128}

# Qwen3-30B-A3B has FIRST_K_DENSE_REPLACE=0 -> every layer is MoE. The int form
# `--moe-layer-freq 1` is the raw-Megatron equivalent of slime's all-ones list
# and is safe inside the string-built CMD below (no spaces/brackets).
MOE_LAYER_FREQ=${MOE_LAYER_FREQ:-1}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Qwen3 architecture knobs are overridable so this script also covers smaller MoE
# smokes (e.g. TP=1 EP=2 NUM_LAYERS=2 NUM_EXPERTS=8 HIDDEN_SIZE=256 MOE_ROUTER_TOPK=2).
# Default values reproduce Qwen3-30B-A3B exactly. QK_LAYERNORM=0 drops --qk-layernorm.
# --sequence-parallel is only valid for TP>1, so gate it on TP (TP=1 smokes need it off).
QK_LAYERNORM_ARG=$([ "${QK_LAYERNORM:-1}" = "1" ] && echo "--qk-layernorm")
SEQUENCE_PARALLEL_ARG=$([ "${TP:-4}" -gt 1 ] && echo "--sequence-parallel")

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --disable-bias-linear
    --seq-length ${SEQ_LEN:-4096}
    --max-position-embeddings ${MAX_POS:-4096}
    --num-layers ${NUM_LAYERS:-48}
    --hidden-size ${HIDDEN_SIZE:-2048}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE:-6144}
    --num-attention-heads ${NUM_HEAD:-32}
    --group-query-attention
    --num-query-groups ${NUM_QUERY_GROUP:-4}
    --kv-channels ${KV_CHANNELS:-128}
    ${QK_LAYERNORM_ARG}
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --attention-backend flash
    --attention-softmax-in-fp32
)

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE:-768}
    --moe-router-topk ${MOE_ROUTER_TOPK:-8}
    --moe-router-score-function ${MOE_SCORE_FUNCTION:-softmax}
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0
    --moe-token-dispatcher-type alltoall
    --moe-token-drop-policy probs
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-layer-freq ${MOE_LAYER_FREQ}
)

DATA_ARGS=(
    --dataloader-type single
    --data-cache-path ${BASE_PATH}/data/data_cache
)
# ---------------------------------------------------------------------------
# 数据 + tokenizer 部分由 resolve_data_args.sh 按文件存在性自动判定:A100 上数据
# 齐全走真实 dataset + Llama2 tokenizer,本地无数据则退化到 --mock-data +
# NullTokenizer。mock vocab 用 151936(= Qwen3-30B-A3B);真实数据路径同样可经
# DATA_DIR / TOKENIZER_DIR / DATA_PREFIX 覆盖,ELASTIC_REQUIRE_REAL_DATA=1 硬断言。
# ---------------------------------------------------------------------------
export MOCK_VOCAB_SIZE=${MOCK_VOCAB_SIZE:-151936}
source "${_SCRIPT_DIR}/resolve_data_args.sh"
# shellcheck disable=SC2206
DATA_ARGS+=(${RESOLVED_DATA_ARGS})

TRAINING_ARGS=(
    --micro-batch-size ${MBS:-1}
    --global-batch-size ${GBS:-64}
    --lr ${LR:-1e-6}
    --train-iters ${TRAIN_ITERS}
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --bf16
    --accumulate-allreduce-grads-in-fp32
    --calculate-per-token-loss
    --distributed-timeout-minutes ${TIMEOUT_MIN:-20}
    --seed 1234
)

TRAINING_ARGS+=(
    --distributed-timeout-seconds-after-init ${TIMEOUT_AFTER_INIT_SEC}
)

# CPU_OFFLOAD=1 (default for 30B) builds Megatron's HybridDeviceOptimizer
# (cpu-adam); requires the precision-aware optimizer path. OFFLOAD_FRACTION is
# the fraction of optimizer-state numel pushed to CPU (1.0 = all, slime 30B).
# Set CPU_OFFLOAD=0 only if you have GPU headroom (30B almost certainly OOMs).
if [ "${CPU_OFFLOAD:-1}" = "1" ]; then
    TRAINING_ARGS+=(
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
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
    ${SEQUENCE_PARALLEL_ARG}
)

# Full activation recompute is ON by default — 30B does not fit otherwise.
RECOMPUTE_ARGS=()
if [ "${RECOMPUTE:-1}" = "1" ]; then
    RECOMPUTE_ARGS=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers ${RECOMPUTE_LAYERS:-1}
    )
fi

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --eval-interval ${EVAL_INTERVAL:-100000}
    --eval-iters ${EVAL_ITERS:-0}
    --no-load-optim
    --no-load-rng
)

SRC=${MEGATRON_PATH}/pretrain_gpt.py
export LOG_DIR=${LOG_DIR:-${BASE_PATH}/log}
LOG_PATH=${LOG_DIR}/qwen3_30b_node${NODE_RANK}.log
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
