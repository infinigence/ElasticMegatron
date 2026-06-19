#!/bin/bash
# -----------------------------------------------------------------------------
# run_experiment.sh —— ElasticMegatron 真实数据实验 runner
#
# 用法:
#   ./run_experiment.sh <exp_name>
#
# 支持的 exp_name(每个都用 4 GPU,可以在另外 4 GPU 上同时跑第二组):
#   dense_baseline_tp1   —— Dense 静态 TP=1/DP=4, 用作 noise floor / 主 baseline
#   dense_baseline_tp2   —— Dense 静态 TP=2/DP=2, 用作不同并行方式对照
#   dense_mix            —— Dense reshard sweep(dense_mix 4-GPU 版,见下)
#   moe_baseline_ep2     —— MoE 静态 EP=2/DP=2
#   moe_baseline_ep1     —— MoE 静态 EP=1/DP=4 (expert-as-dense)
#   moe_mix              —— MoE reshard sweep(moe_mix 4-GPU 版)
#
# Phase B(8-GPU 模式,需 GPUS_PER_NODE=8 + CUDA_VISIBLE_DEVICES=0,...,7):
#   dense_baseline_8gpu  —— TP=1/DP=8 baseline,匹配 dense_mix_full[0]
#   moe_baseline_8gpu    —— TP=1/DP=8/EP=2/NUM_EXPERTS=8 baseline,匹配 moe_mix_full[0]
#   dense_mix_full       —— Dense 8-策略 sweep,含 PP/CP/Group-Zero
#   dense_cp_only        —— 仅 CP/Group-Zero 维度 sweep,隔离用
#   moe_mix_full         —— MoE 8-策略 sweep,含 EP/PP/CP
#   moe_cp_only          —— EP=2 固定,仅 CP/Group-Zero 维度 sweep,隔离用
#   dense_scale_world    —— Dense world 8↔4 scale-down/up,走 run_e2e_demo.sh
#   moe_scale_world      —— MoE world 8↔4 scale-down/up,走 run_moe.sh
#
# 关键环境变量:
#   GPUS_PER_NODE        —— 每节点 GPU 数,默认 4
#   CUDA_VISIBLE_DEVICES —— 默认 "0,1,2,3"
#   MASTER_PORT          —— 默认 6000
#   MODEL_SIZE           —— "medium"(hidden=4096) | "tiny"(hidden=256)
#   NUM_LAYERS           —— 默认 8,OOM 时降到 4
#   TRAIN_ITERS          —— 默认 100
#   ELASTIC_RESHARD_INTERVAL —— 默认 5
# -----------------------------------------------------------------------------

set -euo pipefail

EXP_NAME="${1:-}"
if [[ -z "${EXP_NAME}" ]]; then
    cat >&2 <<EOF
usage: $0 <exp_name>
  dense_baseline_tp1 | dense_baseline_tp2 | dense_mix
  moe_baseline_ep2   | moe_baseline_ep1   | moe_mix
EOF
    exit 1
fi

BASE_PATH=${BASE_PATH:-/workspace}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${BASE_PATH}/log/experiments/${TIMESTAMP}_${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Per-experiment 默认值(可被环境变量覆盖)
export LOG_DIR
export TRAIN_ITERS=${TRAIN_ITERS:-100}
export ELASTIC_RESHARD_INTERVAL=${ELASTIC_RESHARD_INTERVAL:-5}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export MASTER_PORT=${MASTER_PORT:-6000}
export MODEL_SIZE=${MODEL_SIZE:-medium}
export NUM_LAYERS=${NUM_LAYERS:-8}
# MoE 默认 num_experts=4(让 EP=1 时 4 experts 单卡能装下;EP=2 时每卡 2 experts)
export NUM_EXPERTS=${NUM_EXPERTS:-4}

# 真实数据 + Llama2 tokenizer(32000 vocab)。覆盖 run_*.sh 里默认的 --mock-data。
REAL_DATA_PATH=${BASE_PATH}/datasets/wiki_llama2_text_document
REAL_TOKENIZER_MODEL=${BASE_PATH}/tokenizer/tokenizer.model
export REAL_DATA_ARGS="${REAL_DATA_ARGS:---data-path ${REAL_DATA_PATH} --split 98,2,0 --tokenizer-type Llama2Tokenizer --tokenizer-model ${REAL_TOKENIZER_MODEL}}"

# 防御:清掉前一 run 留下的 iter_* ckpt,避免 resume(并行 run 共享同一 ckpt 路径会冲突,
# 但因为我们没开 --save,也不会真的写出 iter_*,这里只是清残留)
rm -rf ${BASE_PATH}/log/iter_* ${BASE_PATH}/log/latest_checkpointed_iteration.txt 2>/dev/null || true

echo "======================================================================="
echo "Experiment:   ${EXP_NAME}"
echo "Log dir:      ${LOG_DIR}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES}  (master_port=${MASTER_PORT})"
echo "Model size:   ${MODEL_SIZE}  (num_layers=${NUM_LAYERS})"
echo "Train iters:  ${TRAIN_ITERS}"
echo "Reshard int.: ${ELASTIC_RESHARD_INTERVAL}"
echo "======================================================================="

case "${EXP_NAME}" in
    dense_baseline_tp1)
        export ELASTIC_ENABLED=0
        export TP=1 PP=1
        bash "$(dirname "$0")/run_e2e_demo.sh"
        ;;
    dense_baseline_tp2)
        export ELASTIC_ENABLED=0
        export TP=2 PP=1
        bash "$(dirname "$0")/run_e2e_demo.sh"
        ;;
    dense_mix)
        # 4-GPU dense_mix 在 training.py 里的 ELASTIC_STRATEGY_MODE=dense_mix 当前 assert
        # world_size=8。4-GPU 用一个简化的 strategy,直接通过 ELASTIC_STRATEGY_MODE=tp_flip
        # 跑 TP=1/DP=4 ↔ TP=2/DP=2 反复切。外部传入 TP=2 时则反向(TP=2→TP=1)。
        export ELASTIC_ENABLED=1
        export ELASTIC_STRATEGY_MODE=tp_flip
        export TP=${TP:-1} PP=${PP:-1}
        bash "$(dirname "$0")/run_e2e_demo.sh"
        ;;
    moe_baseline_ep2)
        export ELASTIC_ENABLED=0
        export TP=1 PP=1 EP=2
        export NUM_LAYERS=${NUM_LAYERS_MOE:-4}
        export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-4096}
        bash "$(dirname "$0")/run_moe.sh"
        ;;
    moe_baseline_ep1)
        export ELASTIC_ENABLED=0
        export TP=1 PP=1 EP=1
        export NUM_LAYERS=${NUM_LAYERS_MOE:-4}
        export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-4096}
        bash "$(dirname "$0")/run_moe.sh"
        ;;
    moe_mix)
        # 4-GPU moe_mix 同样要回退到 ep_flip(EP=2 ↔ EP=1),因为 moe_mix mode 在
        # training.py 里 assert world_size=8。
        export ELASTIC_ENABLED=1
        export ELASTIC_STRATEGY_MODE=ep_flip
        export TP=${TP:-1} PP=${PP:-1} EP=${EP:-2}
        export NUM_LAYERS=${NUM_LAYERS_MOE:-4}
        export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-4096}
        bash "$(dirname "$0")/run_moe.sh"
        ;;

    # ---------------------------------------------------------------
    # Phase B: 8-GPU sweeps (TP/PP/EP/CP/Group-Zero)
    # 调用前必须设置 GPUS_PER_NODE=8 + CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # Launcher args must match strategy[0] = base (TP=1/PP=1/CP=1/EP=base),
    # otherwise ElasticMegatron's check_consistency will assert.
    # ---------------------------------------------------------------
    dense_baseline_8gpu)
        # 8-GPU dense baseline,TP=1/DP=8。和 dense_mix_full 的 strategy[0] 一致。
        export ELASTIC_ENABLED=0
        export TP=1 PP=1 CP=1 NUM_DIST_OPT=1
        bash "$(dirname "$0")/run_e2e_demo.sh"
        ;;
    moe_baseline_8gpu)
        # 8-GPU MoE baseline,TP=1/DP=8/EP=2。和 moe_mix_full 的 strategy[0] 一致。
        # num_experts 必须能被 EP=8 整除 → 强制 NUM_EXPERTS=8(覆盖顶层默认 4)。
        export ELASTIC_ENABLED=0
        export TP=1 PP=1 EP=2 CP=1 NUM_DIST_OPT=1
        export NUM_EXPERTS=8
        export NUM_LAYERS=${NUM_LAYERS_MOE:-4}
        export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-4096}
        bash "$(dirname "$0")/run_moe.sh"
        ;;
    dense_mix_full)
        export ELASTIC_ENABLED=1
        export ELASTIC_STRATEGY_MODE=dense_mix_full
        export TP=1 PP=1 CP=1 NUM_DIST_OPT=1   # base = launcher
        bash "$(dirname "$0")/run_e2e_demo.sh"
        ;;
    dense_cp_only)
        # base = TP=1/PP=1/CP=1/DP=8,sweep 只动 CP / num_distributed_optimizer_instances。
        export ELASTIC_ENABLED=1
        export ELASTIC_STRATEGY_MODE=dense_cp_only
        export TP=1 PP=1 CP=1 NUM_DIST_OPT=1
        bash "$(dirname "$0")/run_e2e_demo.sh"
        ;;
    moe_mix_full)
        # 要求 num_experts 能被 EP=8 整除 → 强制 NUM_EXPERTS=8(覆盖顶层默认 4)。
        export ELASTIC_ENABLED=1
        export ELASTIC_STRATEGY_MODE=moe_mix_full
        export TP=1 PP=1 EP=2 CP=1 NUM_DIST_OPT=1
        export NUM_EXPERTS=8
        export NUM_LAYERS=${NUM_LAYERS_MOE:-4}
        export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-4096}
        bash "$(dirname "$0")/run_moe.sh"
        ;;
    moe_cp_only)
        # base = TP=1/PP=1/EP=2/CP=1/DP=4,sweep 只动 CP / Group-Zero(EP 保持 2)。
        # num_experts 必须 ≥ EP=2,默认 8 与 _full 保持一致(便于复用 ckpt baseline)。
        export ELASTIC_ENABLED=1
        export ELASTIC_STRATEGY_MODE=moe_cp_only
        export TP=1 PP=1 EP=2 CP=1 NUM_DIST_OPT=1
        export NUM_EXPERTS=8
        export NUM_LAYERS=${NUM_LAYERS_MOE:-4}
        export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-4096}
        bash "$(dirname "$0")/run_moe.sh"
        ;;
    dense_scale_world)
        # Explicit asymmetric-world dense test: base world=8, target world=4,
        # then the interval cycle grows back to world=8. Keeps TP/PP/CP fixed.
        export ELASTIC_ENABLED=1
        export ELASTIC_STRATEGY_MODE=dense_scale_world
        export TP=${TP:-1} PP=${PP:-1} CP=${CP:-1} NUM_DIST_OPT=1
        bash "$(dirname "$0")/run_e2e_demo.sh"
        ;;
    moe_scale_world)
        # Explicit asymmetric-world MoE test: base world=8, target world=4,
        # preserving TP/PP/CP/EP/TPE. Small-model analogue of the 30B DP2<->DP1
        # shrink/grow path.
        export ELASTIC_ENABLED=1
        export ELASTIC_STRATEGY_MODE=moe_scale_world
        export TP=${TP:-1} PP=${PP:-1} EP=${EP:-2} CP=${CP:-1} NUM_DIST_OPT=1
        export NUM_EXPERTS=${NUM_EXPERTS:-8}
        export NUM_LAYERS=${NUM_LAYERS_MOE:-4}
        export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-4096}
        bash "$(dirname "$0")/run_moe.sh"
        ;;
    *)
        echo "Unknown exp: ${EXP_NAME}" >&2
        exit 1
        ;;
esac
