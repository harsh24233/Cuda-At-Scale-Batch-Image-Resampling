#!/usr/bin/env bash
# ==============================================================================
# Enterprise Benchmark Runner for CUDA at Scale Independent Project
# Runs:
#   1. Environment inspection and logging
#   2. Multi-scale single-image verification (0.5x, 0.75x, 1.0x, 1.5x, 2.0x)
#   3. Enterprise Batch Processing on 100 small images (single execution)
#   4. Enterprise Batch Processing on 10 large images (single execution)
#   5. Output validation across all results
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

mkdir -p logs results/sample results/batch_small results/batch_large submission_artifacts

echo "============================================================"
echo "Step 1: Capturing System and GPU Environment"
echo "============================================================"
{
  echo "=== Date and Timestamp ==="
  date -u
  echo ""
  echo "=== Host and Kernel ==="
  uname -a
  echo ""
  echo "=== NVIDIA-SMI Device Information ==="
  nvidia-smi || echo "nvidia-smi unavailable"
  echo ""
  echo "=== NVCC Compiler Information ==="
  nvcc --version || echo "nvcc unavailable"
  echo ""
  echo "=== NPP Shared Libraries ==="
  ldconfig -p | grep npp || echo "ldconfig npp check"
} > logs/environment.log 2>&1
echo "Environment logged to logs/environment.log"

echo ""
echo "============================================================"
echo "Step 2: Building CUDA Binary"
echo "============================================================"
make clean > logs/build.log 2>&1
make all >> logs/build.log 2>&1
echo "Build succeeded. Log saved to logs/build.log"

echo ""
echo "============================================================"
echo "Step 3: Ensuring Benchmark Datasets Exist"
echo "============================================================"
python3 scripts/generate_datasets.py

echo ""
echo "============================================================"
echo "Step 4: Executing Multi-Scale Single-Image Resampling"
echo "============================================================"
echo "Scale,InputW,InputH,OutputW,OutputH,GpuTimeMs,Status" > results/benchmark_multi_scale.csv
for scale in 0.5 0.75 1.0 1.5 2.0; do
  scale_label="$(echo "${scale}" | tr -d '.')"
  out_file="results/sample/sample_scale_${scale_label}.pgm"
  log_file="logs/scale_${scale_label}.log"
  
  echo "Running scale factor ${scale}x..."
  ./bin/image_batch_processor \
    --input=data/sample/sample.pgm \
    --output="${out_file}" \
    --scale="${scale}" > "${log_file}" 2>&1
    
  python3 tests/validate_outputs.py "${out_file}"
  
  gpu_time=$(grep "GPU Kernel Time:" "${log_file}" | awk '{print $4}')
  in_w=$(grep "Input Dimensions:" "${log_file}" | awk '{print $3}')
  in_h=$(grep "Input Dimensions:" "${log_file}" | awk '{print $5}')
  out_w=$(grep "Output Dimensions:" "${log_file}" | awk '{print $3}')
  out_h=$(grep "Output Dimensions:" "${log_file}" | awk '{print $5}')
  echo "${scale},${in_w},${in_h},${out_w},${out_h},${gpu_time},SUCCESS" >> results/benchmark_multi_scale.csv
done
echo "Multi-scale tests completed. Metrics in results/benchmark_multi_scale.csv"

echo ""
echo "============================================================"
echo "Step 5: Executing Enterprise Batch Processing on 100 Small Images"
echo "        (Single Continuous Software Execution)"
echo "============================================================"
./bin/image_batch_processor \
  --batch-dir=data/batch_small \
  --output-dir=results/batch_small \
  --scale=0.5 \
  --benchmark-csv=results/benchmark_small_batch.csv 2>&1 | tee logs/batch_small_100.log

python3 tests/validate_outputs.py --dir results/batch_small 100
echo "100-image batch successfully completed and verified!"

echo ""
echo "============================================================"
echo "Step 6: Executing Enterprise Batch Processing on 10 Large Images"
echo "        (Single Continuous Software Execution)"
echo "============================================================"
./bin/image_batch_processor \
  --batch-dir=data/batch_large \
  --output-dir=results/batch_large \
  --scale=0.5 \
  --benchmark-csv=results/benchmark_large_batch.csv 2>&1 | tee logs/batch_large_10.log

python3 tests/validate_outputs.py --dir results/batch_large 10
echo "10-large-image batch successfully completed and verified!"

echo ""
echo "============================================================"
echo "Step 7: Packaging Submission Proof of Execution Artifacts"
echo "============================================================"
bash scripts/package_proof.sh

echo ""
echo "============================================================"
echo "All Benchmark & Validation Runs Completed Successfully!"
echo "============================================================"
