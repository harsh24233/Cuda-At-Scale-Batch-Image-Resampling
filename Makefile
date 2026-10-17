# ==============================================================================
# Makefile for CUDA at Scale Enterprise Batch Image Processor
# Supports: native Linux and WSL2 Ubuntu environments with auto-detection
# ==============================================================================

# Compiler detection
NVCC ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CXX ?= g++

# CUDA and NPP paths
CUDA_INC ?= -Iinclude -I/usr/local/cuda/include -I/usr/include
CUDA_LIB ?= -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu

# Compiler and Linker Flags
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler "-Wall -Wextra"
LDFLAGS ?= $(CUDA_LIB) -lcudart -lnppc -lnppig -lnppicc -lnppial

# Directories
SRC_DIR = src
BIN_DIR = bin
INC_DIR = include
DATA_DIR = data
RESULTS_DIR = results
LOGS_DIR = logs

# Target Binary
TARGET = $(BIN_DIR)/image_batch_processor

# Source Files
SOURCES = $(SRC_DIR)/image_batch_processor.cu $(SRC_DIR)/pgm_image.cpp

# Default rule
.PHONY: all clean run data test benchmark help

all: $(TARGET)

$(TARGET): $(SOURCES) $(INC_DIR)/pgm_image.h
	@mkdir -p $(BIN_DIR)
	@echo "Compiling $(TARGET) with $(NVCC)..."
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	@echo "Build successful: $(TARGET)"

# Generate synthetic datasets (100 small images, 10 large images, 1 sample)
data:
	@mkdir -p $(DATA_DIR)/sample $(DATA_DIR)/batch_small $(DATA_DIR)/batch_large
	@echo "Generating synthetic benchmark datasets..."
	python3 scripts/generate_datasets.py

# Run standard test validation
test: $(TARGET)
	@mkdir -p $(RESULTS_DIR)/sample
	@echo "Running single-image smoke test..."
	$(TARGET) --input=$(DATA_DIR)/sample/sample.pgm --output=$(RESULTS_DIR)/sample/sample_50.pgm --scale=0.5
	python3 tests/validate_outputs.py $(RESULTS_DIR)/sample/sample_50.pgm 160 120

# Run full batch benchmark suite
benchmark: $(TARGET) data
	@bash scripts/run_batch_benchmark.sh

clean:
	@echo "Cleaning binaries and temporary build objects..."
	rm -rf $(BIN_DIR)/*

help:
	@echo "Available make targets:"
	@echo "  make            - Compile the CUDA batch image processor"
	@echo "  make data       - Generate synthetic benchmark datasets (100 small, 10 large)"
	@echo "  make test       - Execute smoke tests and validation"
	@echo "  make benchmark  - Run complete enterprise batch benchmark suite"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make help       - Display this help message"
