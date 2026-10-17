# Installation and Build Guide: CUDA Enterprise Batch Image Resampling Engine

This guide details the system prerequisites, dependencies, and step-by-step procedures to build, execute, and validate the **CUDA Enterprise Batch Image Resampling Engine**.

---

## 1. System and Hardware Requirements

| Component | Minimum Specification | Tested / Verified Configuration |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU (Compute Capability $\ge 5.0$) | **NVIDIA GeForce RTX 4050 Laptop GPU** (Compute Capability 8.9, 6140 MiB GDDR6 VRAM) |
| **NVIDIA Driver** | $\ge 525.60$ | **Driver Version 592.82** (CUDA Version 13.1 compatibility) |
| **CUDA Toolkit** | CUDA 11.8+ / CUDA 12.0+ / CUDA 13.0+ | **CUDA Toolkit 12.0** (`nvcc` release 12.0, V12.0.140) |
| **C++ Host Compiler** | GCC 9.0+ / G++ with C++17 support | **GCC / G++ 13.3.0** (`-std=c++17`) |
| **Operating System** | Linux (Ubuntu 20.04/22.04/24.04 LTS) or Windows 10/11 with WSL2 | **Ubuntu 24.04 LTS on Windows Subsystem for Linux (WSL2)** |
| **Scripting Runtime** | Python 3.8+ (Standard library only; no pip dependencies required) | **Python 3.12.3** |

---

## 2. Environment Verification

Before building the application, verify that your NVIDIA GPU and CUDA compiler are properly detected:

```bash
# 1. Verify NVIDIA GPU hardware detection and driver communication
nvidia-smi

# Expected output shows GPU name, VRAM, and driver version:
# NVIDIA GeForce RTX 4050 Laptop GPU | Driver Version: 592.82 | CUDA Version: 13.1

# 2. Verify NVCC compiler availability and version
nvcc --version

# Expected output:
# Cuda compilation tools, release 12.0, V12.0.140

# 3. Verify NVIDIA Performance Primitives (NPP) headers and shared libraries
ldconfig -p | grep npp
```

---

## 3. Package Dependencies

On Debian/Ubuntu (or Ubuntu running in WSL2), ensure the required build tools and libraries are installed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential nvidia-cuda-toolkit python3
```

---

## 4. Compilation Instructions

The project provides an automated `Makefile` that automatically locates `nvcc` and links against the CUDA Runtime (`libcudart`) and NVIDIA Performance Primitives (`libnppc`, `libnppig`, `libnppicc`, `libnppial`).

```bash
# Clone or navigate to the repository directory
cd /path/to/CUDAatScaleForTheEnterpriseCourseProjectTemplate-main

# Clean any existing binaries
make clean

# Compile the optimized executable
make all
```

Upon successful compilation, the executable binary will be generated at:
```text
bin/image_batch_processor
```

---

## 5. Dataset Generation

The project includes a self-contained synthetic benchmark dataset generator (`scripts/generate_datasets.py`) with deterministic mathematical patterns (gradients, checkerboards, sinusoidal waves, and radial chirps). No external internet downloads or third-party packages are required:

```bash
# Generate 100 small images (128x128), 10 large images (2048x1536, 3.15 MP each), and sample image
make data
# Or directly:
python3 scripts/generate_datasets.py
```

Generated datasets will be populated under:
- `data/sample/sample.pgm`: $320 \times 240$ reference image.
- `data/batch_small/`: 100 images (`image_001.pgm` through `image_100.pgm`).
- `data/batch_large/`: 10 large images (`large_image_01.pgm` through `large_image_10.pgm`, ~3.15 Megapixels each).

---

## 6. Execution Examples

### A. Master Runner (`run.sh`)
The master runner provides an out-of-the-box demonstration:
```bash
./run.sh
```

### B. Single-Image Resampling Mode
Resize a single image with custom scaling:
```bash
./bin/image_batch_processor \
  --input=data/sample/sample.pgm \
  --output=results/sample/sample_50.pgm \
  --scale=0.5
```

### C. Enterprise Batch Processing Mode (Single Execution)
Process all 100 small images in a single continuous invocation:
```bash
./bin/image_batch_processor \
  --batch-dir=data/batch_small \
  --output-dir=results/batch_small \
  --scale=0.5 \
  --benchmark-csv=results/benchmark_small_batch.csv
```

Process 10 large high-resolution images in a single continuous invocation:
```bash
./bin/image_batch_processor \
  --batch-dir=data/batch_large \
  --output-dir=results/batch_large \
  --scale=0.5 \
  --benchmark-csv=results/benchmark_large_batch.csv
```

---

## 7. Testing and Verification

Run automated test validation:
```bash
# Smoke test and dimension validation
make test

# Full benchmark suite (multi-scale, 100-batch, 10-large-batch, packaging)
make benchmark
```
