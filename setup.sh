#!/bin/bash
# One-time container setup for PHI de-identification pipeline
# Run this once after cloning the repo on a new container
#
# Expects: HF_TOKEN set as environment variable

set -e

echo "=== Removing unused torchvision/torchaudio (pure text pipeline — never used, and torchvision's import chain has broken transformers on some Python builds) ==="
pip uninstall -y torchaudio torchvision 2>/dev/null || true

echo "=== Installing pipeline dependencies ==="
pip install -r requirements.txt --quiet

echo "=== Setup complete ==="
