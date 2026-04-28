#!/bin/bash
# MODE=scale_up|scale_down controls which elastic action is configured.
set -euo pipefail

MODE="${MODE:-scale_up}"
# replace ip addr
NODES=(
	"10.204.3.27"
    "10.204.30.217"
)

BASE_DIR="workpath/ElasticMegatron"
SCRIPT_DEMO="run_e2e_demo.sh"
MASTER_PORT="${MASTER_PORT:-6368}"
GPU_PER_NODE="${GPU_PER_NODE:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$GPU_PER_NODE}"


SCALE_UP_ITER="${SCALE_UP_ITER:-999}"
SCALE_DOWN_ITER="${SCALE_DOWN_ITER:-999}"

# Strategy variables.
# - SRC_TP/SRC_PP: current strategy
# - TGT_TP/TGT_PP: target strategy

SRC_PP="${SRC_PP:-2}"
SRC_TP="${SRC_TP:-4}"

if [ "$MODE" = "scale_down" ]; then
    SRC_PP="${DOWN_SRC_PP:-$SRC_PP}"
    SRC_TP="${DOWN_SRC_TP:-$SRC_TP}"
    TGT_PP="${DOWN_TGT_PP:-${TGT_PP:-$SRC_PP}}"
    TGT_TP="${DOWN_TGT_TP:-${TGT_TP:-$SRC_TP}}"
else
    SRC_PP="${UP_SRC_PP:-$SRC_PP}"
    SRC_TP="${UP_SRC_TP:-$SRC_TP}"
    TGT_PP="${UP_TGT_PP:-${TGT_PP:-8}}"
    TGT_TP="${UP_TGT_TP:-${TGT_TP:-2}}"
fi

VENV_ACTIVATE="${VENV_ACTIVATE:-/opt/venv/reason/bin/activate}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o ConnectTimeout=10}"

NNODES=${#NODES[@]}
HALF_NODES=$((NNODES / 2))
MASTER_ADDR=${NODES[0]}

ENV_UTILS_FILE="$BASE_DIR/tools/elastic_control/env_utils.py"
TRAINING_FILE="$BASE_DIR/Megatron-LM/megatron/training/training.py"

inject_new_node_ip_list() {
    if [ ! -f "$ENV_UTILS_FILE" ]; then
        echo "Warning: env_utils not found: $ENV_UTILS_FILE" >&2
        return 0
    fi
    if ! grep -q "#sym:NEW_NODE_IP_LIST" "$ENV_UTILS_FILE"; then
        echo "Warning: missing #sym:NEW_NODE_IP_LIST in $ENV_UTILS_FILE" >&2
        return 0
    fi

    local entries=()
    local j
    for (( j=HALF_NODES; j<NNODES; j++ )); do
        local node_ip=${NODES[$j]}
        entries+=("[\"$node_ip\", 23455, 1]")
    done
    local literal="[$(IFS=,; echo "${entries[*]}")]"
    local escaped_literal
    escaped_literal="$(printf '%s' "$literal" | sed 's/[&|]/\\&/g')"
    sed -i "/#sym:NEW_NODE_IP_LIST/{n;s|^[[:space:]]*NEW_NODE_IP_LIST = .*|    NEW_NODE_IP_LIST = ${escaped_literal}|;}" "$ENV_UTILS_FILE"
    echo "Injected #sym:NEW_NODE_IP_LIST into $ENV_UTILS_FILE: $literal"
}

inject_deleted_node_rank() {
    if [ ! -f "$ENV_UTILS_FILE" ]; then
        echo "Warning: env_utils not found: $ENV_UTILS_FILE" >&2
        return 0
    fi
    if ! grep -q "#sym:set_deleted_node_rank" "$ENV_UTILS_FILE"; then
        echo "Warning: missing #sym:set_deleted_node_rank in $ENV_UTILS_FILE" >&2
        return 0
    fi

    local deleted=()
    local node_idx local_rank global_rank
    for ((node_idx=HALF_NODES; node_idx<NNODES; node_idx++)); do
        for ((local_rank=0; local_rank<GPU_PER_NODE; local_rank++)); do
            global_rank=$((node_idx * GPU_PER_NODE + local_rank))
            deleted+=("$global_rank")
        done
    done

    local deleted_literal="[$(IFS=,; echo "${deleted[*]}")]"
    local escaped_deleted
    escaped_deleted="$(printf '%s' "$deleted_literal" | sed 's/[&|]/\\&/g')"
    sed -i "/#sym:set_deleted_node_rank/{n;s|^[[:space:]]*_DELETED_NODE_RANK = .*|    _DELETED_NODE_RANK = ${escaped_deleted}|;}" "$ENV_UTILS_FILE"
    echo "Injected #sym:set_deleted_node_rank: $deleted_literal"
}

inject_scale_action() {
    if [ ! -f "$TRAINING_FILE" ]; then
        echo "Warning: training file not found: $TRAINING_FILE" >&2
        return 0
    fi
    if ! grep -q "#sym:scale_action" "$TRAINING_FILE"; then
        echo "Warning: missing #sym:scale_action in $TRAINING_FILE" >&2
        return 0
    fi
    if [ "$MODE" = "scale_down" ]; then
        sed -i "/#sym:scale_action/{n;s|^[[:space:]]*scale_action = .*|    scale_action = ElasticMode.SCALE_DOWN|;}" "$TRAINING_FILE"
        echo "Injected #sym:scale_action as SCALE_DOWN"
    else
        sed -i "/#sym:scale_action/{n;s|^[[:space:]]*scale_action = .*|    scale_action = ElasticMode.SCALE_UP|;}" "$TRAINING_FILE"
        echo "Injected #sym:scale_action as SCALE_UP"
    fi
}

inject_parallel_strategy() {
    if [ ! -f "$TRAINING_FILE" ]; then
        echo "Warning: training file not found: $TRAINING_FILE" >&2
        return 0
    fi
    if ! grep -q "#sym:init_parallel_strategy_list" "$TRAINING_FILE"; then
        echo "Warning: missing #sym:init_parallel_strategy_list in $TRAINING_FILE" >&2
        return 0
    fi

    if [ "$MODE" = "scale_down" ]; then
        local tgt_world_size=$((HALF_NODES * GPU_PER_NODE))
        local eff_tgt_tp="${TGT_TP}"
        local eff_tgt_pp="${TGT_PP}"
        if [ -n "${DOWN_TGT_TP}" ]; then
            eff_tgt_tp="${DOWN_TGT_TP}"
        fi
        if [ -n "${DOWN_TGT_PP}" ]; then
            eff_tgt_pp="${DOWN_TGT_PP}"
        fi
        sed -i "/#sym:init_parallel_strategy_list/{n;s|^[[:space:]]*\"world_size\": .*|            \"world_size\": ${tgt_world_size},|;n;s|^[[:space:]]*\"tensor_model_parallel_size\": .*|            \"tensor_model_parallel_size\": ${eff_tgt_tp},|;n;s|^[[:space:]]*\"pipeline_model_parallel_size\": .*|            \"pipeline_model_parallel_size\": ${eff_tgt_pp},|;}" "$TRAINING_FILE"
        echo "Injected #sym:init_parallel_strategy_list world_size=$tgt_world_size tp=$eff_tgt_tp pp=$eff_tgt_pp"
    else
        local tgt_world_size=$((NNODES * GPU_PER_NODE / 2))
        sed -i "/#sym:init_parallel_strategy_list/{n;s|^[[:space:]]*\"world_size\": .*|            \"world_size\": ${tgt_world_size},|;n;s|^[[:space:]]*\"tensor_model_parallel_size\": .*|            \"tensor_model_parallel_size\": ${SRC_TP},|;n;s|^[[:space:]]*\"pipeline_model_parallel_size\": .*|            \"pipeline_model_parallel_size\": ${SRC_PP},|;}" "$TRAINING_FILE"
        echo "Injected #sym:init_parallel_strategy_list world_size=$tgt_world_size tp=$SRC_TP pp=$SRC_PP"
    fi
}

inject_elastic_target_env() {
    if [ ! -f "$ENV_UTILS_FILE" ]; then
        echo "Warning: env_utils not found: $ENV_UTILS_FILE" >&2
        return 0
    fi
    if ! grep -q "#sym:elastic_target_env" "$ENV_UTILS_FILE"; then
        echo "Warning: missing #sym:elastic_target_env in $ENV_UTILS_FILE" >&2
        return 0
    fi
    if [ "$MODE" = "scale_up" ]; then
        local tgt_nnodes="${TGT_NNODES:-$NNODES}"
        sed -i "/#sym:elastic_target_env/{n;s|^[[:space:]]*new_env\\[\"NNODES\"\\] = .*|        new_env[\"NNODES\"] = \"${tgt_nnodes}\"|;n;s|^[[:space:]]*new_env\\[\"PP\"\\] = .*|        new_env[\"PP\"] = \"${TGT_PP}\"|;n;s|^[[:space:]]*new_env\\[\"TP\"\\] = .*|        new_env[\"TP\"] = \"${TGT_TP}\"|;}" "$ENV_UTILS_FILE"
        echo "Injected #sym:elastic_target_env NNODES=$tgt_nnodes PP=$TGT_PP TP=$TGT_TP"
    fi
}

echo "MODE: $MODE"
echo "Total nodes: $NNODES"
echo "Half nodes split index: $HALF_NODES"
echo "Master addr: $MASTER_ADDR"
echo "Master port (old group): $MASTER_PORT"
echo "GPUS per node: $GPUS_PER_NODE"
echo "Trigger iters: ELASTIC_SCALE_UP_ITER=$SCALE_UP_ITER ELASTIC_SCALE_DOWN_ITER=$SCALE_DOWN_ITER"

inject_new_node_ip_list
if [ "$MODE" = "scale_down" ]; then
    inject_deleted_node_rank
fi
inject_scale_action
inject_parallel_strategy
inject_elastic_target_env

for (( i=0; i<NNODES; i++ )); do
    NODE=${NODES[$i]}
    RANK=$i

    echo "Launching on $NODE with RANK=$RANK..."

    base_exports="export RANK=$RANK && export NODE_RANK=$RANK && export MASTER_ADDR=$MASTER_ADDR && export MASTER_PORT=$MASTER_PORT && export GPUS_PER_NODE=$GPUS_PER_NODE && export NNODES=$NNODES && export ELASTIC_SCALE_UP_ITER=$SCALE_UP_ITER && export ELASTIC_SCALE_DOWN_ITER=$SCALE_DOWN_ITER"

    if [ "$MODE" = "scale_up" ] && [ $i -lt $HALF_NODES ]; then
        # old group only; new nodes will be launched by rank0 via ssh later.
        base_exports="$base_exports && export TP=$SRC_TP && export PP=$SRC_PP && export NNODES=$HALF_NODES && export ELASTIC_TARGET_NNODES=${TGT_NNODES:-$NNODES} && export ELASTIC_TARGET_TP=$TGT_TP && export ELASTIC_TARGET_PP=$TGT_PP"
    elif [ "$MODE" = "scale_down" ]; then
        base_exports="$base_exports && export ELASTIC_TARGET_NNODES=$HALF_NODES"
    else
        if [ "$MODE" = "scale_up" ]; then
            echo "  -> Role: Reserved for SSH scale-up launch"
            continue
        fi
    fi

    remote_cmd="cd $BASE_DIR && [ -f $VENV_ACTIVATE ] && source $VENV_ACTIVATE; $base_exports && bash $SCRIPT_DEMO"

    if [ "$NODE" = "localhost" ] || [ "$NODE" = "127.0.0.1" ] || [ "$NODE" = "$(hostname)" ]; then
        bash -lc "$remote_cmd" &
    else
        if [ -n "$SSH_USER" ]; then
            ssh $SSH_OPTS "$SSH_USER@$NODE" "$remote_cmd" &
        else
            ssh $SSH_OPTS "$NODE" "$remote_cmd" &
        fi
    fi
done

wait
echo "All tasks finished."

