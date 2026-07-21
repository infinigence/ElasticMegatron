#!/bin/bash
# -----------------------------------------------------------------------------
# resolve_data_args.sh —— 真实数据 / mock 数据自动判定(按文件存在性退化)
#
# 被各 launcher(run_dense.sh / run_qwen3_30b.sh)source 进来。
# 目的:同一个脚本既能在 A100(数据齐全)上跑 *真实* dataset + 真实 tokenizer,
# 也能在没有数据的机器(本地 dev)上自动退化到 --mock-data + NullTokenizer,而
# 不是把不存在的 --data-path 丢给 Megatron 直接崩。
#
# 判定优先级:
#   1. 调用方显式传了非空 REAL_DATA_ARGS  → 原样使用(显式覆盖最高优先级)。
#   2. 否则按文件存在性:三件套(${DATA_DIR}/${DATA_PREFIX}.bin / .idx +
#      ${TOKENIZER_DIR}/tokenizer.model)都在 → 走真实 dataset + Llama2Tokenizer。
#   3. 否则 → 退化到 --mock-data + NullTokenizer,并向 stderr 打一条醒目告警。
#   4. 安全阀:ELASTIC_REQUIRE_REAL_DATA=1 且文件缺失(且没有显式 REAL_DATA_ARGS)
#      时,直接报错 exit 1 而非退化 —— 让 A100 真实数据 run 可以硬断言。
#
# 可覆盖变量(默认对齐 A100:BASE_PATH=/mnt/hisys-data/tonic):
#   DATA_DIR       默认 ${BASE_PATH}/datasets
#   TOKENIZER_DIR  默认 ${BASE_PATH}/tokenizer
#   DATA_PREFIX    默认 wiki_llama2_text_document(--data-path 用的是不带扩展名的前缀)
#   MOCK_VOCAB_SIZE 默认 32000(Qwen3-30B-A3B launcher 覆盖为 151936)
#
# 产出:把判定好的「数据 + tokenizer」那段参数赋给 RESOLVED_DATA_ARGS(字符串),
# 由各 launcher 自行拼上各自的 --seq-length / --num-workers / --dataloader-type /
# --data-cache-path 等(那几项不在这里管,保持每个脚本原样)。
# -----------------------------------------------------------------------------

DATA_DIR=${DATA_DIR:-${BASE_PATH}/datasets}
TOKENIZER_DIR=${TOKENIZER_DIR:-${BASE_PATH}/tokenizer}
DATA_PREFIX=${DATA_PREFIX:-wiki_llama2_text_document}
MOCK_VOCAB_SIZE=${MOCK_VOCAB_SIZE:-32000}

_DATA_BIN="${DATA_DIR}/${DATA_PREFIX}.bin"
_DATA_IDX="${DATA_DIR}/${DATA_PREFIX}.idx"
_TOKENIZER_MODEL="${TOKENIZER_DIR}/tokenizer.model"

if [[ -n "${REAL_DATA_ARGS:-}" ]]; then
    # (1) 调用方显式覆盖,原样透传。
    RESOLVED_DATA_ARGS="${REAL_DATA_ARGS}"
    echo "[data] using caller-provided REAL_DATA_ARGS"
elif [[ -f "${_DATA_BIN}" && -f "${_DATA_IDX}" && -f "${_TOKENIZER_MODEL}" ]]; then
    # (2) 三件套齐全 → 真实 dataset + Llama2 tokenizer。
    RESOLVED_DATA_ARGS="--data-path ${DATA_DIR}/${DATA_PREFIX} --split 98,2,0 --tokenizer-type Llama2Tokenizer --tokenizer-model ${_TOKENIZER_MODEL}"
    echo "[data] using REAL dataset: ${DATA_DIR}/${DATA_PREFIX} + tokenizer ${_TOKENIZER_MODEL}"
elif [[ "${ELASTIC_REQUIRE_REAL_DATA:-0}" = "1" ]]; then
    # (4) 安全阀:要求真实数据但文件缺失 → 硬失败,不退化。
    echo "[data] ERROR: ELASTIC_REQUIRE_REAL_DATA=1 but real dataset/tokenizer NOT found under DATA_DIR=${DATA_DIR} / TOKENIZER_DIR=${TOKENIZER_DIR} (need ${DATA_PREFIX}.bin + ${DATA_PREFIX}.idx + tokenizer.model); refusing to fall back to --mock-data" >&2
    exit 1
else
    # (3) 退化到 mock,醒目告警(stderr)。
    RESOLVED_DATA_ARGS="--mock-data --tokenizer-type NullTokenizer --vocab-size ${MOCK_VOCAB_SIZE}"
    echo "[data] WARNING: real dataset/tokenizer NOT found under DATA_DIR=${DATA_DIR} / TOKENIZER_DIR=${TOKENIZER_DIR}; falling back to --mock-data + NullTokenizer (set the paths or drop the files to use real data)" >&2
fi
