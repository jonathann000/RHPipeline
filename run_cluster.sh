#!/bin/bash
# Run the PHI de-identification pipeline
# Usage: bash run_cluster.sh [--input path] [--output path] [--audit path]
#                             [--llm llama|mistral|qwen|gemma] [--mode full|no_bert|llm_only]
#                             [--gazetteer path/to.csv] [--judges name1 name2 ...]
#                             [--judge-max-rounds N] [--llm-backstop] [--llm-thinking] [--quasi-only]
#
# --input/--output/--audit default to data/notes.txt, data/redacted.txt,
# data/audit.json respectively if not given.
#
# Expects:
#   - HF_TOKEN set as environment variable
#   - BERT_MODEL_PATH set to the BERT model directory
#   - setup.sh already run once

set -e

export HF_HOME="$(pwd)/.model_cache"

LLM="${LLM:-llama}"
MODE="${MODE:-full}"
INPUT="data/notes.txt"
OUTPUT="data/redacted.txt"
AUDIT="data/audit.json"
GAZETTEER=""
JUDGES=()
JUDGE_MAX_ROUNDS=""
LLM_BACKSTOP=""
LLM_THINKING=""
QUASI_ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm)               LLM="$2"; shift 2 ;;
        --mode)               MODE="$2"; shift 2 ;;
        --input)              INPUT="$2"; shift 2 ;;
        --output)             OUTPUT="$2"; shift 2 ;;
        --audit)              AUDIT="$2"; shift 2 ;;
        --gazetteer)          GAZETTEER="$2"; shift 2 ;;
        --judge-max-rounds)   JUDGE_MAX_ROUNDS="$2"; shift 2 ;;
        --llm-backstop)       LLM_BACKSTOP="--llm-backstop"; shift ;;
        --llm-thinking)       LLM_THINKING="--llm-thinking"; shift ;;
        --quasi-only)         QUASI_ONLY="--quasi-only"; shift ;;
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
echo "=== Input: ${INPUT} ==="
[[ -n "$GAZETTEER" ]] && echo "=== Gazetteer: ${GAZETTEER} ==="
[[ ${#JUDGES[@]} -gt 0 ]] && echo "=== Judges: ${JUDGES[*]} ==="

ARGS=(
    --input "${INPUT}"
    --output "${OUTPUT}"
    --audit "${AUDIT}"
    --mode "${MODE}"
    --llm "${LLM}"
)
[[ -n "$GAZETTEER" ]] && ARGS+=(--gazetteer "$GAZETTEER")
[[ ${#JUDGES[@]} -gt 0 ]] && ARGS+=(--judges "${JUDGES[@]}")
[[ -n "$JUDGE_MAX_ROUNDS" ]] && ARGS+=(--judge-max-rounds "$JUDGE_MAX_ROUNDS")
[[ -n "$LLM_BACKSTOP" ]] && ARGS+=("$LLM_BACKSTOP")
[[ -n "$LLM_THINKING" ]] && ARGS+=("$LLM_THINKING")
[[ -n "$QUASI_ONLY" ]] && ARGS+=("$QUASI_ONLY")

.venv/bin/python run.py "${ARGS[@]}"

echo ""
echo "=== Done! ==="
echo "  Redacted: ${OUTPUT}"
echo "  Audit:    ${AUDIT}"
