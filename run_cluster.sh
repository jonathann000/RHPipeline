#!/bin/bash
# Run the PHI de-identification pipeline
# Usage: bash run_cluster.sh [--llm llama|mistral|qwen|gemma] [--mode full|no_bert|llm_only]
#                             [--gazetteer path/to.csv] [--judges name1 name2 ...]
#                             [--judge-max-rounds N] [--llm-backstop] [--llm-thinking]
#
# Expects:
#   - HF_TOKEN set as environment variable
#   - BERT_MODEL_PATH set to the BERT model directory
#   - setup.sh already run once

set -e

export HF_HOME="$(pwd)/.model_cache"

LLM="${LLM:-llama}"
MODE="${MODE:-full}"
GAZETTEER=""
JUDGES=()
JUDGE_MAX_ROUNDS=""
LLM_BACKSTOP=""
LLM_THINKING=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm)               LLM="$2"; shift 2 ;;
        --mode)               MODE="$2"; shift 2 ;;
        --gazetteer)          GAZETTEER="$2"; shift 2 ;;
        --judge-max-rounds)   JUDGE_MAX_ROUNDS="$2"; shift 2 ;;
        --llm-backstop)       LLM_BACKSTOP="--llm-backstop"; shift ;;
        --llm-thinking)       LLM_THINKING="--llm-thinking"; shift ;;
        --judges)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                JUDGES+=("$1")
                shift
            done
            ;;
        *) shift ;;
    esac
done

echo "=== BERT model: ${BERT_MODEL_PATH:-not set, using default} ==="
echo "=== LLM: ${LLM} | Mode: ${MODE} ==="
[[ -n "$GAZETTEER" ]] && echo "=== Gazetteer: ${GAZETTEER} ==="
[[ ${#JUDGES[@]} -gt 0 ]] && echo "=== Judges: ${JUDGES[*]} ==="

ARGS=(
    --input data/notes.txt
    --output data/redacted.txt
    --audit data/audit.json
    --mode "${MODE}"
    --llm "${LLM}"
)
[[ -n "$GAZETTEER" ]] && ARGS+=(--gazetteer "$GAZETTEER")
[[ ${#JUDGES[@]} -gt 0 ]] && ARGS+=(--judges "${JUDGES[@]}")
[[ -n "$JUDGE_MAX_ROUNDS" ]] && ARGS+=(--judge-max-rounds "$JUDGE_MAX_ROUNDS")
[[ -n "$LLM_BACKSTOP" ]] && ARGS+=("$LLM_BACKSTOP")
[[ -n "$LLM_THINKING" ]] && ARGS+=("$LLM_THINKING")

python run.py "${ARGS[@]}"

echo ""
echo "=== Done! ==="
echo "  Redacted: data/redacted.txt"
echo "  Audit:    data/audit.json"
