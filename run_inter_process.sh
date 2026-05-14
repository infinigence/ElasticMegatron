#!/bin/bash
# MODE=scale_up|scale_down controls which elastic action is configured.
set -euo pipefail

MODE="${MODE:-scale_up}"
# replace ip addr
NODES=(
    "<node-1-ip>"
    "<node-2-ip>"
    ...
)

BASE_DIR="workpath/ElasticMegatron"
SCRIPT_DEMO="run_e2e_demo.sh"
MASTER_PORT="${MASTER_PORT:-6000}"
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

VENV_ACTIVATE="${VENV_ACTIVATE:-/path/to/your/venv/bin/activate}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o ConnectTimeout=10}"

NNODES=${#NODES[@]}
HALF_NODES=$((NNODES / 2))
MASTER_ADDR=${NODES[0]}
ELASTIC_RENDEZVOUS_MASTER_ADDR="${ELASTIC_RENDEZVOUS_MASTER_ADDR:-${RENDEZVOUS_MASTER:-$MASTER_ADDR}}"

export ELASTIC_INTER_PROCESS_MODE="$MODE"

if [ "$MODE" = "scale_down" ]; then
    ELASTIC_STRATEGY1_WORLD_SIZE=$((HALF_NODES * GPU_PER_NODE))
    ELASTIC_STRATEGY1_TP="${TGT_TP}"
    ELASTIC_STRATEGY1_PP="${TGT_PP}"
    if [ -n "${DOWN_TGT_TP:-}" ]; then
        ELASTIC_STRATEGY1_TP="${DOWN_TGT_TP}"
    fi
    if [ -n "${DOWN_TGT_PP:-}" ]; then
        ELASTIC_STRATEGY1_PP="${DOWN_TGT_PP}"
    fi
else
    ELASTIC_STRATEGY1_WORLD_SIZE=$((NNODES * GPU_PER_NODE / 2))
    ELASTIC_STRATEGY1_TP="${SRC_TP}"
    ELASTIC_STRATEGY1_PP="${SRC_PP}"
fi
export ELASTIC_STRATEGY1_WORLD_SIZE ELASTIC_STRATEGY1_TP ELASTIC_STRATEGY1_PP

entries=()
for (( j=HALF_NODES; j<NNODES; j++ )); do
    node_ip="${NODES[$j]}"
    entries+=("[\"${node_ip}\",23455,1]")
done
ELASTIC_NEW_NODE_IP_LIST="[$(IFS=,; echo "${entries[*]}")]"
export ELASTIC_NEW_NODE_IP_LIST

ELASTIC_DELETED_RANKS=""
if [ "$MODE" = "scale_down" ]; then
    for ((node_idx=HALF_NODES; node_idx<NNODES; node_idx++)); do
        for ((local_rank=0; local_rank<GPU_PER_NODE; local_rank++)); do
            global_rank=$((node_idx * GPU_PER_NODE + local_rank))
            if [ -n "$ELASTIC_DELETED_RANKS" ]; then
                ELASTIC_DELETED_RANKS+=","
            fi
            ELASTIC_DELETED_RANKS+="$global_rank"
        done
    done
fi
export ELASTIC_DELETED_RANKS

elastic_exports="export ELASTIC_INTER_PROCESS_MODE=$(printf %q "$ELASTIC_INTER_PROCESS_MODE")"
elastic_exports="$elastic_exports && export ELASTIC_STRATEGY1_WORLD_SIZE=$ELASTIC_STRATEGY1_WORLD_SIZE"
elastic_exports="$elastic_exports && export ELASTIC_STRATEGY1_TP=$(printf %q "$ELASTIC_STRATEGY1_TP")"
elastic_exports="$elastic_exports && export ELASTIC_STRATEGY1_PP=$(printf %q "$ELASTIC_STRATEGY1_PP")"
elastic_exports="$elastic_exports && export ELASTIC_NEW_NODE_IP_LIST=$(printf %q "$ELASTIC_NEW_NODE_IP_LIST")"
elastic_exports="$elastic_exports && export ELASTIC_DELETED_RANKS=$(printf %q "$ELASTIC_DELETED_RANKS")"
elastic_exports="$elastic_exports && export ELASTIC_RENDEZVOUS_MASTER_ADDR=$(printf %q "$ELASTIC_RENDEZVOUS_MASTER_ADDR")"

echo "MODE: $MODE"
echo "Total nodes: $NNODES"
echo "Master addr: $MASTER_ADDR"
echo "ELASTIC_RENDEZVOUS_MASTER_ADDR: $ELASTIC_RENDEZVOUS_MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "GPUS per node: $GPUS_PER_NODE"
echo "Trigger iters: ELASTIC_SCALE_UP_ITER=$SCALE_UP_ITER ELASTIC_SCALE_DOWN_ITER=$SCALE_DOWN_ITER"
echo "Elastic env: ELASTIC_WORLD_SIZE=$ELASTIC_STRATEGY1_WORLD_SIZE TP=$ELASTIC_STRATEGY1_TP PP=$ELASTIC_STRATEGY1_PP"
if [ "$MODE" = "scale_down" ]; then
    echo "Scale-down initial torchrun TP=$SRC_TP PP=$SRC_PP"
fi
echo "ELASTIC_NEW_NODE_IP_LIST=$ELASTIC_NEW_NODE_IP_LIST"
echo "ELASTIC_DELETED_RANKS=$ELASTIC_DELETED_RANKS"

for (( i=0; i<NNODES; i++ )); do
    NODE=${NODES[$i]}
    RANK=$i

    echo "Launching on $NODE with RANK=$RANK..."

    base_exports="export RANK=$RANK && export NODE_RANK=$RANK && export MASTER_ADDR=$MASTER_ADDR && export MASTER_PORT=$MASTER_PORT && export GPUS_PER_NODE=$GPUS_PER_NODE && export NNODES=$NNODES && export ELASTIC_SCALE_UP_ITER=$SCALE_UP_ITER && export ELASTIC_SCALE_DOWN_ITER=$SCALE_DOWN_ITER"
    base_exports="$base_exports && $elastic_exports"

    if [ "$MODE" = "scale_up" ] && [ $i -lt $HALF_NODES ]; then
        base_exports="$base_exports && export TP=$SRC_TP && export PP=$SRC_PP && export NNODES=$HALF_NODES && export ELASTIC_TARGET_NNODES=${TGT_NNODES:-$NNODES} && export ELASTIC_TARGET_TP=$TGT_TP && export ELASTIC_TARGET_PP=$TGT_PP"
    elif [ "$MODE" = "scale_down" ]; then
        base_exports="$base_exports && export TP=$SRC_TP && export PP=$SRC_PP && export ELASTIC_TARGET_NNODES=$HALF_NODES"
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

