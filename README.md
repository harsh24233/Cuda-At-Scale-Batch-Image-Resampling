# CUDA Enterprise Batch Image Resampling Engine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NPP](https://img.shields.io/badge/NVIDIA-NPP%20Accelerated-76B900.svg)](https://docs.nvidia.com/cuda/npp/index.html)
[![C%2B%2B](https://img.shields.io/badge/Standard-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![Google C++ Style](https://img.shields.io/badge/Style-Google%20C%2B%2B-informational.svg)](https://google.github.io/styleguide/cppguide.html)
[![Verification](https://img.shields.io/badge/Batch%20Validation-100%25%20PASS-brightgreen.svg)]()

> **Coursera Course:** CUDA at Scale for the Enterprise  
> **Assignment:** CUDA at Scale Independent Project  
> **Author:** Harsh  
> **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (Compute Capability 8.9, 6140 MiB GDDR6)  
> **Software Environment:** Ubuntu 24.04 LTS (WSL2), CUDA Toolkit 12.0, Driver 592.82, GCC 13.3.0  

---

## Table of Contents
1. [Overview and Objectives](#overview-and-objectives)
2. [Key Architectural Highlights](#key-architectural-highlights)
3. [Repository File Structure](#repository-file-structure)
4. [Prerequisites and Environment](#prerequisites-and-environment)
5. [Build Instructions](#build-instructions)
6. [Command-Line Interface (CLI) Reference](#command-line-interface-cli-reference)
7. [Dataset Description](#dataset-description)
8. [Benchmarking and Execution Proof](#benchmarking-and-execution-proof)
9. [Mathematical Formulation: Bilinear Interpolation](#mathematical-formulation-bilinear-interpolation)
10. [Validation Suite](#validation-suite)
11. [Troubleshooting and FAQ](#troubleshooting-and-faq)

---

## 1. Overview and Objectives
Digital image spatial resampling (resizing) is an essential preprocessing step across industrial computer vision pipelines, autonomous systems, deep learning training ingestion, and real-time medical imaging. When processing enterprise workloads containing hundreds of thousands of high-resolution images, conventional CPU-based resizing algorithms hit severe memory bandwidth bottlenecks and high serialized latency.

This project delivers an **Enterprise-Grade GPU-Accelerated Batch Image Resampling Engine** built on NVIDIA CUDA and the NVIDIA Performance Primitives (NPP) geometry library (`nppiResize_8u_C1R`).

The primary objectives achieved:
- **Massive Parallelism:** Concurrent execution across thousands of CUDA threads utilizing NVIDIA SIMT architecture.
- **Enterprise Batch Processing:** A single software execution processes entire datasets (both **100+ small images** and **10+ large high-resolution images**) without host-program relaunch overhead.
- **Microsecond GPU Latency:** Hardware-timed kernel resampling achieving throughput of up to **729.99 Megapixels/sec** on an NVIDIA GeForce RTX 4050 Laptop GPU.
- **Strict Compliance:** Written strictly adhering to the **Google C++ Style Guide**, robust error handling (`CheckCuda`, `CheckNpp`), RAII memory management, and zero third-party dependencies for maximum portability.

---

## 2. Key Architectural Highlights
- **Dual Operating Modes:**
  1. *Single Image Mode:* Highly flexible CLI accepting `--input`, `--output`, and custom `--scale` factors.
  2. *Enterprise Batch Mode:* Seamless single-execution directory traversal processing hundreds of inputs (`--batch-dir`, `--output-dir`), with automated performance logging and CSV generation.
- **Hardware-Accurate Profiling:** Uses asynchronous hardware CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`) to measure pure GPU execution time, isolating compute latency from PCIe transfer and disk I/O.
- **Self-Contained Netpbm PGM Engine:** Zero-dependency, standards-compliant Netpbm P5 binary parser and serializer (`include/pgm_image.h`, `src/pgm_image.cpp`) with complete header comment handling, dimensions checking, and payload verification.
- **End-to-End Validation:** Fully automated Python validation scripts (`tests/validate_outputs.py`) verify the byte integrity, header consistency, and scaled dimensions of every output image.

---

## 3. Repository File Structure

```text
.
├── bin/
│   ├── .gitkeep
│   └── image_batch_processor        # Compiled CUDA binary (generated)
├── data/
│   ├── .gitkeep
│   ├── Lena.png                     # Course template reference image
│   ├── sample/
│   │   └── sample.pgm               # 320x240 standard reference sample
│   ├── batch_small/                 # 100 small images (128x128 each)
│   └── batch_large/                 # 10 large images (2048x1536, 3.15 MP each)
├── include/
│   └── pgm_image.h                  # C++ header for PGM I/O & structures
├── logs/
│   ├── environment.log              # nvidia-smi, nvcc, OS information
│   ├── build.log                    # make clean && make build output
│   ├── batch_small_100.log          # Complete log of 100-image batch run
│   ├── batch_large_10.log           # Complete log of 10-large-image batch run
│   └── scale_*.log                  # Multi-scale individual test logs
├── results/
│   ├── benchmark_small_batch.csv    # 100 rows of measured small-batch metrics
│   ├── benchmark_large_batch.csv    # 10 rows of measured large-batch metrics
│   ├── benchmark_multi_scale.csv    # Multi-scale benchmark measurements
│   ├── sample/                      # Resized sample outputs (0.5x, 0.75x, 1.0x, 1.5x, 2.0x)
│   ├── batch_small/                 # Resized small batch images (64x64 each)
│   └── batch_large/                 # Resized large batch images (1024x768 each)
├── scripts/
│   ├── generate_datasets.py         # Deterministic synthetic dataset generator
│   ├── run_batch_benchmark.sh       # Master benchmark & test automation suite
│   └── package_proof.sh             # Packages proof ZIP with SHA256 checksums
├── src/
│   ├── image_batch_processor.cu     # CUDA/NPP batch & single processing application
│   ├── pgm_image.cpp                # Netpbm P5 binary parser and writer
│   └── imageRotationNPP.cpp         # Original course template reference
├── submission_artifacts/
│   ├── project_description.md       # Full submission text description
│   ├── proof_of_execution.zip       # Verified 341 KB archive of execution evidence
│   └── SHA256SUMS.txt               # Cryptographic hashes of all proof files
├── tests/
│   └── validate_outputs.py          # Structural & payload validation suite
├── .gitignore                       # Clean Git tracking configuration
├── INSTALL                          # Quick install pointer
├── INSTALL.md                       # Comprehensive setup and compilation guide
├── LICENSE                          # Apache 2.0 Open Source License
├── Makefile                         # GNU Makefile with auto-detecting compiler
├── PROJECT_DESCRIPTION.md           # Peer-review project description document
├── README.md                        # Primary project documentation (this file)
├── RUBRIC_AUDIT.md                  # Comprehensive self-audit against assignment rubric
└── run.sh                           # Master executable shell script
```

---

## 4. Prerequisites and Environment

The application has been compiled, executed, and benchmarked on physical NVIDIA hardware:

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (20 SMs, 2560 CUDA cores, Compute Capability 8.9)
- **VRAM:** 6140 MiB GDDR6
- **Host OS:** Ubuntu 24.04 LTS (WSL2 / Linux kernel 5.15.167.4-microsoft-standard-WSL2)
- **NVIDIA Driver:** 592.82
- **CUDA Toolkit:** Release 12.0 (V12.0.140)
- **Libraries:** CUDA Runtime (`libcudart`), NVIDIA Performance Primitives (`libnppc`, `libnppig`, `libnppicc`, `libnppial`)
- **Python:** Python 3.12.3 (Standard library; no pip installations needed)

---

## 5. Build Instructions

### Quick Build
```bash
# Clean previous builds and compile the binary
make clean
make all
```

The Makefile automatically identifies `nvcc` in your path, sets `-std=c++17`, enables optimization `-O3`, and links all required NPP shared libraries.

### Available Makefile Targets
| Target | Description |
| :--- | :--- |
| `make` / `make all` | Compiles `bin/image_batch_processor` |
| `make data` | Generates 100 small images, 10 large images, and sample image |
| `make test` | Executes smoke test and verifies output dimensions |
| `make benchmark` | Runs the full automated multi-scale and enterprise batch suites |
| `make clean` | Cleans compiled binaries in `bin/` |
| `make help` | Displays list of make commands |

---

## 6. Command-Line Interface (CLI) Reference

The executable supports structured arguments in both single-image and batch modes:

### Single-Image Mode
```bash
./bin/image_batch_processor \
  --input=data/sample/sample.pgm \
  --output=results/sample/sample_50.pgm \
  --scale=0.5
```

### Enterprise Batch Mode (Single Continuous Execution)
```bash
./bin/image_batch_processor \
  --batch-dir=data/batch_small \
  --output-dir=results/batch_small \
  --scale=0.5 \
  --benchmark-csv=results/benchmark_small_batch.csv
```

### Argument Summary
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input=<path>` | String | Required in single mode | Path to input `.pgm` image |
| `--output=<path>` | String | Required in single mode | Path to destination `.pgm` image |
| `--batch-dir=<dir>` | String | Required in batch mode | Directory containing `.pgm` batch files |
| `--output-dir=<dir>` | String | Required in batch mode | Directory to store processed `.pgm` images |
| `--scale=<factor>` | Double | `0.5` | Resampling scale factor (finite positive float) |
| `--benchmark-csv=<path>`| String | Optional | Path to CSV file for recording execution metrics |
| `--help` | Flag | N/A | Prints usage details and exits |

---

## 7. Dataset Description

In compliance with the assignment prompt requiring either **100s of small inputs** or **10s of large inputs**, this project provides **BOTH**:

1. **100 Small Images Dataset (`data/batch_small/`):**
   - 100 unique images ($128 \times 128$, 16,384 pixels each).
   - Generated deterministically via `scripts/generate_datasets.py`.
   - Contains mathematical patterns: dynamic checkerboards, concentric rings, diagonal gradients, and sinusoidal interference.
   - Total volume: **1.64 Megapixels** (1.56 MB raw).
2. **10 Large High-Resolution Dataset (`data/batch_large/`):**
   - 10 unique 3.15-Megapixel images ($2048 \times 1536$, 3,145,728 pixels each).
   - Frequency-modulated sinusoidal coordinate space.
   - Total volume: **31.46 Megapixels** (30.00 MB raw).
3. **Reference Sample (`data/sample/sample.pgm`):**
   - $320 \times 240$ Netpbm P5 binary grayscale test image.

To generate or regenerate the datasets:
```bash
python3 scripts/generate_datasets.py
```

---

## 8. Benchmarking and Execution Proof

All benchmarks were executed on physical hardware (**NVIDIA GeForce RTX 4050 Laptop GPU**) using hardware CUDA event timers.

### A. Enterprise Batch Execution: 100 Small Images (Single Execution)
- **Command:** `./bin/image_batch_processor --batch-dir=data/batch_small --output-dir=results/batch_small --scale=0.5 --benchmark-csv=results/benchmark_small_batch.csv`
- **Total Images Processed:** **100 / 100**
- **Input Data Volume:** 1.64 Megapixels (1.56 MB)
- **Output Data Volume:** 0.41 Megapixels (0.39 MB)
- **Cumulative GPU Kernel Time:** **12.05 ms**
- **Average Latency per Image:** **0.121 ms** (121 microseconds)
- **Total Wall-Clock Time:** 1,271.73 ms (includes disk I/O)
- **GPU Pixel Throughput:** **33.99 Megapixels / second**
- **Validation Status:** **100 / 100 PASSED (100% SUCCESS)**

### B. Enterprise Batch Execution: 10 Large Images (Single Execution)
- **Command:** `./bin/image_batch_processor --batch-dir=data/batch_large --output-dir=results/batch_large --scale=0.5 --benchmark-csv=results/benchmark_large_batch.csv`
- **Total Images Processed:** **10 / 10**
- **Input Data Volume:** 31.46 Megapixels (30.00 MB)
- **Output Data Volume:** 7.86 Megapixels (7.50 MB)
- **Cumulative GPU Kernel Time:** **10.77 ms**
- **Average Latency per Image:** **1.077 ms**
- **Total Wall-Clock Time:** 421.76 ms
- **GPU Resampling Throughput:** **729.99 Megapixels / second**
- **Validation Status:** **10 / 10 PASSED (100% SUCCESS)**

### C. Multi-Scale Single Image Verification
Tested across five scale factors on $320 \times 240$ input:

| Scale Factor | Input Dimensions | Output Dimensions | Output Bytes | GPU Kernel Latency | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50x** | 320 x 240 | 160 x 120 | 19,200 B | 17.05 ms | SUCCESS |
| **0.75x** | 320 x 240 | 240 x 180 | 43,200 B | 7.49 ms | SUCCESS |
| **1.00x** | 320 x 240 | 320 x 240 | 76,800 B | 27.69 ms | SUCCESS |
| **1.50x** | 320 x 240 | 480 x 360 | 172,800 B | 17.89 ms | SUCCESS |
| **2.00x** | 320 x 240 | 640 x 480 | 307,200 B | 6.54 ms | SUCCESS |

---

## 9. Mathematical Formulation: Bilinear Interpolation

The resampling algorithm implemented by `nppiResize_8u_C1R` uses continuous bilinear coordinate mapping:

1. **Coordinate Mapping:**  
   Maps discrete destination coordinates $(x_{out}, y_{out})$ into continuous source coordinate space $(x_{in}, y_{in})$:
   $$x_{in} = (x_{out} + 0.5) \times \frac{W_{in}}{W_{out}} - 0.5$$
   $$y_{in} = (y_{out} + 0.5) \times \frac{H_{in}}{H_{out}} - 0.5$$

2. **Neighbor Identification:**  
   Identifies the four surrounding discrete integer coordinate pixels:
   $$x_1 = \lfloor x_{in} \rfloor, \quad x_2 = \min(x_1 + 1, W_{in} - 1)$$
   $$y_1 = \lfloor y_{in} \rfloor, \quad y_2 = \min(y_1 + 1, H_{in} - 1)$$

3. **Fractional Distance Weights:**  
   $$\alpha = x_{in} - x_1, \quad \beta = y_{in} - y_1$$

4. **Weighted Interpolation:**  
   $$I(x_{out}, y_{out}) = (1 - \alpha)(1 - \beta)P(x_1, y_1) + \alpha(1 - \beta)P(x_2, y_1) + (1 - \alpha)\beta P(x_1, y_2) + \alpha\beta P(x_2, y_2)$$

NVIDIA NPP maps this computation over thousands of active GPU threads concurrently, utilizing texture caches and hardware memory coalescing.

---

## 10. Validation Suite

The project includes strict automated byte-level and dimension verification scripts:

```bash
# Validate single output image against expected dimensions
python3 tests/validate_outputs.py results/sample/sample_scale_05.pgm 160 120

# Validate all 100 images in small batch directory
python3 tests/validate_outputs.py --dir results/batch_small 100

# Validate all 10 images in large batch directory
python3 tests/validate_outputs.py --dir results/batch_large 10
```

---

## 11. Troubleshooting and FAQ

- **Q: `nvcc: command not found`**  
  *A:* Ensure CUDA Toolkit is installed and `/usr/local/cuda/bin` is in your `PATH`. Under Ubuntu/WSL, run `sudo apt install nvidia-cuda-toolkit`.
- **Q: `No CUDA-capable GPU detected`**  
  *A:* Verify your host GPU drivers by running `nvidia-smi`. In WSL2, ensure NVIDIA WSL Driver $\ge 525$ is installed on the Windows host.
- **Q: Where are the execution proofs located?**  
  *A:* All verified execution logs are stored in `logs/`, CSV benchmark data in `results/`, and the compact archive is at `submission_artifacts/proof_of_execution.zip`.
