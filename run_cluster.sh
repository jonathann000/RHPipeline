#!/bin/bash
# Cluster launcher for PHI de-identification pipeline
# Usage: bash run_cluster.sh [--setup-only] [--run-only]
#
# Expects:
#   - HF_TOKEN set as environment variable
#   - WORK_DIR set to the mounted bucket path containing this repo (defaults to current dir)

set -e

WORK_DIR="${WORK_DIR:-$(pwd)}"
MODEL_CACHE="${WORK_DIR}/.model_cache"

export HF_HOME="${MODEL_CACHE}"

echo "=== Work dir: ${WORK_DIR} ==="
echo "=== Model cache: ${MODEL_CACHE} ==="

# --- Setup: install deps + download model ---
setup() {
    echo ""
    echo "=== Installing dependencies ==="
    pip install -r "${WORK_DIR}/requirements.txt" --quiet

    echo ""
    echo "=== Downloading Llama 3.1 8B (skips if already cached) ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct',
    cache_dir='${MODEL_CACHE}',
)
print('Model ready.')
"
}

# --- Run the pipeline ---
run() {
    echo ""
    echo "=== Smoke test (mock LLM) ==="
    cd "${WORK_DIR}"
    python -c "
from pipeline import PIIPipeline
pipe = PIIPipeline({'mode': 'no_bert', 'llm_backend': 'mock', 'llm_model_path': '', 'bert_model_path': ''})
result = pipe.run('Test 070-123 45 67 erik@test.com')
print(f'Smoke test OK - {len(result.entities)} entities found')
"

    echo ""
    echo "=== Running full pipeline ==="
    python run.py \
        --input "${WORK_DIR}/data/notes.txt" \
        --output "${WORK_DIR}/data/redacted.txt" \
        --audit "${WORK_DIR}/data/audit.json" \
        --mode full \
        --llm llama

    echo ""
    echo "=== Done! Output: ==="
    echo "  Redacted: ${WORK_DIR}/data/redacted.txt"
    echo "  Audit:    ${WORK_DIR}/data/audit.json"
}

# --- Parse args ---
case "${1}" in
    --setup-only) setup ;;
    --run-only)   run ;;
    *)            setup && run ;;
esac
