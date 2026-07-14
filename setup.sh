#!/bin/bash
# One-time container setup for PHI de-identification pipeline
# Run this once after cloning the repo on a new container
#
# Expects: HF_TOKEN set as environment variable
#
# Installs into an isolated .venv rather than the container's system Python —
# shared GPU containers (e.g. NGC images) ship their own pinned package set
# (cudf, pyarrow, etc.) that a system-wide pip install can silently break,
# and installing as root there is a bad idea regardless.

set -e

if [ ! -d ".venv" ]; then
    echo "=== Creating virtual environment (.venv) ==="
    python3 -m venv .venv
fi

echo "=== Removing unused torchvision/torchaudio (pure text pipeline — never used, and torchvision's import chain has broken transformers on some Python builds) ==="
.venv/bin/pip uninstall -y torchaudio torchvision 2>/dev/null || true

echo "=== Installing pipeline dependencies ==="
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet

echo "=== Setup complete ==="
echo "Run the pipeline with: bash run_cluster.sh ...  (it uses .venv/bin/python automatically)"
