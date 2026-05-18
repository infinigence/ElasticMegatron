#!/usr/bin/env bash
# Batch verify all (before_reshard, after_reshard) iter pairs in tools/ckpt/.
#
# Usage:
#   tools/ckpt/verify_all.sh [device_for_optim]
#
#   device_for_optim: "cuda" or "cuda:0" (default: "cuda")
#
# 输出每对 (model weight 是否 bit-exact, optim state 是否 multiset-equal),
# 最后统计通过率。失败 case 的 detail 写到 /tmp/verify_<iter>.log。

set -euo pipefail

DEVICE_OPT="${1:-cuda}"   # for compare_optim_logical
DEVICES_OPT="${DEVICES:-0,1,2}"  # 多 GPU 并行(覆盖单 device)
THRESH="${THRESH:-1e-3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEFORE_DIR="${SCRIPT_DIR}/before_reshard"
AFTER_DIR="${SCRIPT_DIR}/after_reshard"

if [[ ! -d "${BEFORE_DIR}" || ! -d "${AFTER_DIR}" ]]; then
    echo "No ckpts found at ${BEFORE_DIR} / ${AFTER_DIR}" >&2
    exit 1
fi

# 取交集的 iter 编号
iters=$(comm -12 \
    <(ls "${BEFORE_DIR}" | grep -E '^iter_[0-9]+$' | sort) \
    <(ls "${AFTER_DIR}"  | grep -E '^iter_[0-9]+$' | sort))

total=0
weight_pass=0
optim_pass=0
weight_fail=()
optim_fail=()

for it in $iters; do
    total=$((total + 1))
    a="${BEFORE_DIR}/${it}"
    b="${AFTER_DIR}/${it}"
    log="/tmp/verify_${it}.log"
    echo "================================================================"
    echo "[$it] weight compare ..."
    if python3 "${SCRIPT_DIR}/compare_dcp.py" "$a" "$b" --thresh "${THRESH}" --device "${DEVICE_OPT}" > "${log}.weight" 2>&1; then
        echo "  ✓ weight bit-equal"
        weight_pass=$((weight_pass + 1))
    else
        echo "  ✗ weight MISMATCH (see ${log}.weight)"
        weight_fail+=("${it}")
    fi

    echo "[$it] optim compare ..."
    if python3 "${SCRIPT_DIR}/compare_optim_logical.py" "$a" "$b" --thresh "${THRESH}" --devices "${DEVICES_OPT}" > "${log}.optim" 2>&1; then
        echo "  ✓ optim multiset-equal"
        optim_pass=$((optim_pass + 1))
    else
        echo "  ✗ optim MISMATCH (see ${log}.optim)"
        optim_fail+=("${it}")
    fi
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
