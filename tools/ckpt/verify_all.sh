#!/usr/bin/env bash
# Batch verify all (before_reshard, after_reshard) iter pairs in tools/ckpt/.
#
# Usage:
#   tools/ckpt/verify_all.sh [device_for_optim]
#
#   device_for_optim: "cuda" or "cuda:0" (default: "cuda")
#
# Parallelism (the iter pairs are independent):
#   GPUS=0,1,2,3,4,5,6,7   GPUs to spread work over (default: CUDA_VISIBLE_DEVICES,
#                          else all GPUs from nvidia-smi). Each pair is pinned to ONE
#                          GPU and the pairs run concurrently.
#   JOBS=N                 max concurrent pairs (default: number of GPUs).
#   THRESH=1e-3            rel_rms threshold.
#
# 输出每对 (model weight 是否 bit-exact, optim state 是否 multiset-equal),
# 最后统计通过率。失败 case 的 detail 写到 /tmp/verify_<iter>.log.{weight,optim}.

set -uo pipefail

DEVICE_OPT="${1:-cuda}"   # kept for backward-compat; each job is pinned to one GPU
THRESH="${THRESH:-1e-3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEFORE_DIR="${SCRIPT_DIR}/before_reshard"
AFTER_DIR="${SCRIPT_DIR}/after_reshard"

if [[ ! -d "${BEFORE_DIR}" || ! -d "${AFTER_DIR}" ]]; then
    echo "No ckpts found at ${BEFORE_DIR} / ${AFTER_DIR}" >&2
    exit 1
fi

# --- GPU pool -------------------------------------------------------------
if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
else
    n=$(nvidia-smi -L 2>/dev/null | wc -l)
    [[ "${n}" -lt 1 ]] && n=1
    GPU_ARR=($(seq 0 $((n - 1))))
fi
NGPU=${#GPU_ARR[@]}
JOBS="${JOBS:-${NGPU}}"
[[ "${JOBS}" -lt 1 ]] && JOBS=1

# 取交集的 iter 编号
iters=$(comm -12 \
    <(ls "${BEFORE_DIR}" | grep -E '^iter_[0-9]+$' | sort) \
    <(ls "${AFTER_DIR}"  | grep -E '^iter_[0-9]+$' | sort))

RES_DIR="$(mktemp -d)"
trap 'rm -rf "${RES_DIR}"' EXIT

echo "Verifying $(echo "${iters}" | wc -w) pair(s) over GPUs [${GPU_ARR[*]}], up to ${JOBS} concurrent ..."

# One pair = weight compare + optim compare, pinned to a single GPU. Writes
# "<weight_rc> <optim_rc>" to ${RES_DIR}/${it}.
run_pair() {
    local it="$1" gpu="$2"
    local a="${BEFORE_DIR}/${it}" b="${AFTER_DIR}/${it}"
    local log="/tmp/verify_${it}.log"
    local wrc orc
    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${SCRIPT_DIR}/compare_dcp.py" \
        "$a" "$b" --thresh "${THRESH}" --device cuda > "${log}.weight" 2>&1
    wrc=$?
    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${SCRIPT_DIR}/compare_optim_logical.py" \
        "$a" "$b" --thresh "${THRESH}" --devices 0 > "${log}.optim" 2>&1
    orc=$?
    echo "${wrc} ${orc}" > "${RES_DIR}/${it}"
    local wtag otag
    [[ ${wrc} -eq 0 ]] && wtag="✓ weight" || wtag="✗ weight (see ${log}.weight)"
    [[ ${orc} -eq 0 ]] && otag="✓ optim"  || otag="✗ optim (see ${log}.optim)"
    echo "[$it] ${wtag} | ${otag}"
}

# --- dispatch with bounded concurrency ------------------------------------
i=0
for it in ${iters}; do
    gpu="${GPU_ARR[$((i % NGPU))]}"
    run_pair "${it}" "${gpu}" &
    i=$((i + 1))
    # throttle: while running jobs >= JOBS, wait for one to finish
    while [[ "$(jobs -r | wc -l)" -ge "${JOBS}" ]]; do
        wait -n 2>/dev/null || true
    done
done
wait

# --- aggregate ------------------------------------------------------------
total=0; weight_pass=0; optim_pass=0; weight_fail=(); optim_fail=()
for it in ${iters}; do
    total=$((total + 1))
    read -r wrc orc < "${RES_DIR}/${it}"
    if [[ "${wrc}" -eq 0 ]]; then weight_pass=$((weight_pass + 1)); else weight_fail+=("${it}"); fi
    if [[ "${orc}" -eq 0 ]]; then optim_pass=$((optim_pass + 1));  else optim_fail+=("${it}"); fi
done

echo
echo "================================================================"
echo "Summary"
echo "  total iter pairs: ${total}"
echo "  weight pass:      ${weight_pass}/${total}"
echo "  optim pass:       ${optim_pass}/${total}"
if [[ ${#weight_fail[@]} -gt 0 ]]; then
    echo "  weight fail iters: ${weight_fail[*]}"
fi
if [[ ${#optim_fail[@]} -gt 0 ]]; then
    echo "  optim fail iters:  ${optim_fail[*]}"
fi

if [[ ${weight_pass} -eq ${total} && ${optim_pass} -eq ${total} ]]; then
    echo "  → ALL PASS"
    exit 0
fi
exit 1
