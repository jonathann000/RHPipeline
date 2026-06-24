#!/bin/bash
# One-time container setup for PHI de-identification pipeline
# Run this once after cloning the repo on a new container
#
# Expects: HF_TOKEN set as environment variable

set -e

echo "=== Fixing container dependencies ==="
pip uninstall -y torchaudio 2>/dev/null || true
pip install --upgrade torchvision --quiet 2>/dev/null || true

echo "=== Installing pipeline dependencies ==="
pip install -r requirements.txt --quiet

echo "=== Setup complete ==="
