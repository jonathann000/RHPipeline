#!/bin/bash
# Run the PHI de-identification pipeline
# Usage: bash run_cluster.sh [--llm llama|eurollm|gpt-sw3] [--mode full|no_bert|llm_only]
#
# Expects:
#   - HF_TOKEN set as environment variable
#   - BERT_MODEL_PATH set to the BERT model directory
#   - setup.sh already run once

set -e

export HF_HOME="$(pwd)/.model_cache"

LLM="${LLM:-llama}"
MODE="${MODE:-full}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm)  LLM="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *)      shift ;;
    esac
done

echo "=== BERT model: ${BERT_MODEL_PATH:-not set, using default} ==="
echo "=== LLM: ${LLM} | Mode: ${MODE} ==="

python run.py \
    --input data/notes.txt \
    --output data/redacted.txt \
    --audit data/audit.json \
    --mode "${MODE}" \
    --llm "${LLM}"

echo ""
echo "=== Done! ==="
echo "  Redacted: data/redacted.txt"
echo "  Audit:    data/audit.json"
