#!/usr/bin/env bash
# ==============================================================================
# Packages genuine execution logs, benchmark CSVs, and output images into
# submission_artifacts/proof_of_execution.zip with SHA256 checksums.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

OUTPUT_ZIP="submission_artifacts/proof_of_execution.zip"
CHECKSUMS="submission_artifacts/SHA256SUMS.txt"

echo "Packaging proof of execution artifacts..."
rm -f "${OUTPUT_ZIP}" "${CHECKSUMS}"

python3 - << 'EOF'
import os
import zipfile
import hashlib

root_dir = os.getcwd()
zip_path = os.path.join(root_dir, "submission_artifacts", "proof_of_execution.zip")
checksums_path = os.path.join(root_dir, "submission_artifacts", "SHA256SUMS.txt")

os.makedirs(os.path.dirname(zip_path), exist_ok=True)

# Select files to package
files_to_pack = []

# Logs
for log in os.listdir("logs"):
    if log.endswith(".log"):
        files_to_pack.append(os.path.join("logs", log))

# Benchmark CSVs
for r in os.listdir("results"):
    if r.endswith(".csv"):
        files_to_pack.append(os.path.join("results", r))

# Multi-scale outputs
if os.path.isdir("results/sample"):
    for s in os.listdir("results/sample"):
        if s.endswith(".pgm"):
            files_to_pack.append(os.path.join("results", "sample", s))

# Representative batch small outputs (first 5)
if os.path.isdir("results/batch_small"):
    small_files = sorted([f for f in os.listdir("results/batch_small") if f.endswith(".pgm")])
    for s in small_files[:5]:
        files_to_pack.append(os.path.join("results", "batch_small", s))

# Representative batch large outputs (first 1)
if os.path.isdir("results/batch_large"):
    large_files = sorted([f for f in os.listdir("results/batch_large") if f.endswith(".pgm")])
    for l in large_files[:1]:
        files_to_pack.append(os.path.join("results", "batch_large", l))

print(f"Adding {len(files_to_pack)} verified artifacts to {zip_path}...")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for rel_path in files_to_pack:
        if os.path.isfile(rel_path):
            zf.write(rel_path, arcname=rel_path)

# Compute SHA256 checksums
with open(checksums_path, "w") as cs:
    for rel_path in sorted(files_to_pack):
        if os.path.isfile(rel_path):
            with open(rel_path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
                cs.write(f"{digest}  {rel_path}\n")

zip_size_kb = os.path.getsize(zip_path) / 1024
print(f"Successfully created {zip_path} ({zip_size_kb:.1f} KB)")
print(f"Checksums saved to {checksums_path}")
EOF
