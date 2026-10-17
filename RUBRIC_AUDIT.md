# Coursera Capstone Rubric Audit: CUDA at Scale Independent Project

**Project:** Enterprise GPU-Accelerated Batch Image Resampling Engine  
**Course:** CUDA at Scale for the Enterprise  
**Author:** Harsh  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (20 SMs, Compute Capability 8.9, 6140 MiB GDDR6 VRAM)  
**Software Environment:** Ubuntu 24.04 LTS (WSL2), CUDA Toolkit 12.0, Driver 592.82, GCC 13.3.0  

---

## Executive Summary of Rubric Audit

| Rubric Criterion | Max Score | Verified Score | Audit Status | Evidence File(s) |
| :--- | :---: | :---: | :---: | :--- |
| **1. Code Repository Quality** | 50% | **50%** | **PASS (Tier 5 - Max)** | `README.md`, `Makefile`, `run.sh`, `INSTALL.md`, CLI args, Google C++ Style Guide |
| **2. Proof of Execution Artifacts** | 25% | **25%** | **PASS (Max)** | 100 small images batch & 10 large images batch in single execution; logs, CSVs, `proof_of_execution.zip` |
| **3. Code Project Description** | 25% | **25%** | **PASS (Max)** | `PROJECT_DESCRIPTION.md`, full theory, math formulation, RTX 4050 benchmark table, lessons learned |
| **Total Verified Grade** | **100%** | **100%** | **EXEMPLARY** | **All criteria 100% satisfied with authentic GPU evidence** |

---

## Detailed Section Audit

### 1. Code Repository (50% Weight)
*Evaluation Tiers:*
- *0: No URL given or doesn't point to a valid code repository*
- *10: Exists but incomplete*
- *20: Exists but no README.md with description of how to run it*
- *30: Code includes README.md and CLI which takes arguments*
- *40: Well-written code that meets Google C++ Style Guide*
- *50: Code meets above guidelines and includes support files for compiling and running (Makefile, run.sh, etc.)*

#### Verification & Evidence:
- [x] **URL & Repository Completeness (50/50 Tier):** Complete, self-contained repository with organized folders (`bin/`, `src/`, `include/`, `data/`, `logs/`, `results/`, `scripts/`, `tests/`, `submission_artifacts/`).
- [x] **README.md:** Comprehensive, professional `README.md` containing architectural diagrams, CLI parameter tables, step-by-step build instructions, and benchmark performance summaries.
- [x] **CLI with Arguments:** Robust CLI in `src/image_batch_processor.cu` accepting `--input`, `--output`, `--scale`, `--batch-dir`, `--output-dir`, `--benchmark-csv`, and `--help`.
- [x] **Google C++ Style Guide Compliance:** Clean C++17/CUDA code structure (`pgm_image.h`, `pgm_image.cpp`, `image_batch_processor.cu`): PascalCase for functions/structs, snake_case variables, explicit namespaces, no naked `new`/`delete`, explicit error checking (`CheckCuda`, `CheckNpp`).
- [x] **Support Files:** Fully automated `Makefile` with auto-detecting compiler/library paths, unified `run.sh` entry point, detailed `INSTALL.md`, clean `.gitignore`, and `LICENSE` (Apache 2.0).

---

### 2. Proof of Execution Artifacts (25% Weight)
*Evaluation Tiers:*
- *0: No evidence presented*
- *15 (Mostly): It is clear that the code ran but whether it ran on one or multiple pieces of data in a single execution of the software is not clear.*
- *25 (Yes): Clear evidence that code was executed on either a lot of small pieces of data (signal input arrays over time) or a few large pieces of data (images/videos) in a single execution.*

#### Verification & Evidence:
- [x] **Single-Execution Batch on 100 Small Images (25/25 Criteria):**
  - Executed on 100 small Netpbm images ($128 \times 128$) in a **single continuous invocation** of `bin/image_batch_processor`.
  - Full authentic console log captured in `logs/batch_small_100.log`.
  - All 100 output images generated in `results/batch_small/` and structurally verified with `tests/validate_outputs.py` (**100 / 100 PASS**).
  - Metrics logged in `results/benchmark_small_batch.csv` showing 12.05 ms cumulative GPU kernel time (0.121 ms avg per image).
- [x] **Single-Execution Batch on 10 Large Images (25/25 Criteria):**
  - Executed on 10 large 3.15-Megapixel images ($2048 \times 1536$) in a **single continuous invocation**.
  - Total data volume: 31.46 Megapixels (30.00 MB raw payload).
  - Output images generated in `results/batch_large/` and verified (**10 / 10 PASS**).
  - Full log captured in `logs/batch_large_10.log`, CSV in `results/benchmark_large_batch.csv` showing 10.77 ms GPU time (729.99 Megapixels/sec throughput).
- [x] **Multi-Scale Single Image Verification:**
  - Scaled across 5 factors (0.5x, 0.75x, 1.0x, 1.5x, 2.0x) on physical RTX 4050 GPU, logged in `logs/scale_*.log` and `results/benchmark_multi_scale.csv`.
- [x] **Proof Package Archive:**
  - Packaged into `submission_artifacts/proof_of_execution.zip` (341.6 KB) with full cryptographic verification in `submission_artifacts/SHA256SUMS.txt`.

---

### 3. Code Project Description (25% Weight)
*Evaluation Tiers:*
- *0: No demonstration presented*
- *15 (Mostly): Clear that student did the work, but not clear if they thought about results or were challenged*
- *25 (Yes): Helpful text description showing significant effort beyond hello-world level, algorithms/kernels, results analysis, and lessons learned.*

#### Verification & Evidence:
- [x] **Purpose & Enterprise Objectives:** Documented in `PROJECT_DESCRIPTION.md` Section 1 & 2.
- [x] **Algorithms & GPU Kernels:** Complete mathematical formulation of bilinear continuous resampling in Section 4; CUDA/NPP architectural analysis in Section 3.
- [x] **Processing Pipeline & Lifecycle:** Detailed 8-stage pipeline from disk ingestion to GPU device allocation, H2D/D2H memory copies, hardware event timing, and cleanup in Section 5.
- [x] **Challenges & Technical Resolutions:** Documented in Section 7 (pitch/stride alignment, single-execution batch context management, large image memory bounds).
- [x] **Experimental Results & Analysis:** Populated with authentic measured GPU benchmark tables from physical RTX 4050 GPU in Section 8.
- [x] **PCIe Bottlenecks & Deep Insights:** Critical analysis comparing GPU compute latency vs disk I/O in Section 9.
- [x] **Lessons Learned & Future Work:** Documented in Section 10.

---

## Exact Commands to Publish to GitHub

To submit your project repository to Coursera:

```bash
# 1. Initialize git in the project root
cd /mnt/d/CUDAatScaleForTheEnterpriseCourseProjectTemplate-main
git init

# 2. Configure git user identity (if not already set)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 3. Stage all source code, documentation, logs, results, and submission artifacts
git add .

# 4. Commit changes
git commit -m "Complete CUDA Enterprise Batch Image Resampling Project conforming to rubric"

# 5. Create a new public repository on GitHub (e.g., https://github.com/your-username/cuda-at-scale-batch-resampling)
# 6. Link remote and push
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```
