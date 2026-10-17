#!/usr/bin/env bash
# ==============================================================================
# Master Execution Script for CUDA Enterprise Batch Image Processor
# Usage:
#   ./run.sh [arguments]
#
# Examples:
#   # Run default demo (builds if necessary, runs single and batch mode)
#   ./run.sh
#
#   # Single image mode
#   ./run.sh --input=data/sample/sample.pgm --output=results/sample/sample_50.pgm --scale=0.5
#
#   # Batch processing mode (processes 100 images in a single execution)
#   ./run.sh --batch-dir=data/batch_small --output-dir=results/batch_small --scale=0.5
#
#   # Help
#   ./run.sh --help
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Build if binary doesn't exist
if [[ ! -x "bin/image_batch_processor" ]]; then
  echo "Binary not found. Compiling via make..."
  make all
fi

# If arguments provided, pass directly to binary
if [[ $# -gt 0 ]]; then
  exec ./bin/image_batch_processor "$@"
fi

# Default demonstration workflow
echo "============================================================"
echo "CUDA Enterprise Batch Image Processor - Demonstration"
echo "============================================================"

# Ensure test sample exists
if [[ ! -f "data/sample/sample.pgm" ]]; then
  echo "Generating sample dataset..."
  python3 scripts/generate_datasets.py
fi

mkdir -p results/sample results/batch_small

echo ""
echo "--- Demo 1: Single-Image GPU Resampling (0.5x) ---"
./bin/image_batch_processor \
  --input=data/sample/sample.pgm \
  --output=results/sample/sample_50.pgm \
  --scale=0.5

echo ""
echo "--- Demo 2: Enterprise Batch Processing on 100 Small Images ---"
./bin/image_batch_processor \
  --batch-dir=data/batch_small \
  --output-dir=results/batch_small \
  --scale=0.5 \
  --benchmark-csv=results/benchmark_small_batch.csv

echo ""
echo "Demonstration complete! Results saved in results/"
