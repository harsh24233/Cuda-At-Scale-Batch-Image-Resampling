// Copyright 2026 Harsh. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "pgm_image.h"

namespace {

struct Options {
  std::string input_path;
  std::string output_path;
  std::string batch_dir;
  std::string output_dir;
  std::string benchmark_csv;
  double scale = 0.5;
  bool is_batch = false;
};

void PrintUsage(const char* program_name) {
  std::cout << "Usage:\n"
            << "  Single Image Mode:\n"
            << "    " << program_name
            << " --input=FILE.pgm --output=FILE.pgm --scale=FACTOR\n\n"
            << "  Enterprise Batch Processing Mode (Single Execution):\n"
            << "    " << program_name
            << " --batch-dir=DIR --output-dir=DIR --scale=FACTOR "
               "[--benchmark-csv=FILE.csv]\n\n"
            << "Options:\n"
            << "  --input=<path>          Path to single input PGM image\n"
            << "  --output=<path>         Path to single output PGM image\n"
            << "  --batch-dir=<path>      Directory containing input PGM images\n"
            << "  --output-dir=<path>     Directory to save processed PGM images\n"
            << "  --scale=<factor>        Resizing scale factor (> 0.0, e.g., 0.5, 2.0)\n"
            << "  --benchmark-csv=<path>  Optional CSV file to append benchmark metrics\n"
            << "  --help                  Display this help message\n";
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }

    const std::size_t equals = arg.find('=');
    if (equals == std::string::npos) {
      std::cerr << "Invalid argument (expected --name=value): " << arg << '\n';
      return false;
    }

    const std::string key = arg.substr(0, equals);
    const std::string value = arg.substr(equals + 1);

    if (key == "--input") {
      options->input_path = value;
    } else if (key == "--output") {
      options->output_path = value;
    } else if (key == "--batch-dir") {
      options->batch_dir = value;
      options->is_batch = true;
    } else if (key == "--output-dir") {
      options->output_dir = value;
    } else if (key == "--benchmark-csv") {
      options->benchmark_csv = value;
    } else if (key == "--scale") {
      try {
        std::size_t consumed = 0;
        options->scale = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(options->scale) ||
            options->scale <= 0.0) {
          std::cerr << "Invalid scale value (must be finite positive number): "
                    << value << '\n';
          return false;
        }
      } catch (const std::exception&) {
        std::cerr << "Invalid scale value: " << value << '\n';
        return false;
      }
    } else {
      std::cerr << "Unknown argument: " << key << '\n';
      return false;
    }
  }

  if (options->is_batch) {
    if (options->batch_dir.empty() || options->output_dir.empty()) {
      std::cerr << "Batch mode requires both --batch-dir and --output-dir.\n";
      return false;
    }
  } else {
    if (options->input_path.empty() || options->output_path.empty()) {
      std::cerr << "Single image mode requires both --input and --output.\n"
                << "Or specify --batch-dir and --output-dir for batch mode.\n";
      return false;
    }
  }
  return true;
}

bool CheckCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::cerr << "[CUDA Error] " << operation << " failed: "
            << cudaGetErrorString(status) << '\n';
  return false;
}

bool CheckNpp(NppStatus status, const char* operation) {
  if (status == NPP_SUCCESS) return true;
  std::cerr << "[NPP Error] " << operation << " failed with status "
            << static_cast<int>(status) << '\n';
  return false;
}

// Processes a single loaded image through the CUDA/NPP resizing pipeline.
// Measures exact GPU kernel latency via CUDA Events.
bool ResizeImageGpu(const PgmImage& source, double scale, PgmImage* destination,
                    float* kernel_time_ms) {
  const long long scaled_width =
      std::llround(static_cast<double>(source.width) * scale);
  const long long scaled_height =
      std::llround(static_cast<double>(source.height) * scale);

  if (scaled_width < 1 || scaled_height < 1 ||
      scaled_width > std::numeric_limits<int>::max() ||
      scaled_height > std::numeric_limits<int>::max()) {
    std::cerr << "Scale factor produces invalid output dimensions: "
              << scaled_width << " x " << scaled_height << '\n';
    return false;
  }

  destination->width = static_cast<int>(scaled_width);
  destination->height = static_cast<int>(scaled_height);

  const std::uint64_t output_pixels =
      static_cast<std::uint64_t>(destination->width) * destination->height;
  if (output_pixels > static_cast<std::uint64_t>(
                          std::numeric_limits<std::size_t>::max())) {
    std::cerr << "Output image dimensions exceed memory limits.\n";
    return false;
  }

  destination->pixels.resize(static_cast<std::size_t>(output_pixels));

  Npp8u* device_source = nullptr;
  Npp8u* device_destination = nullptr;
  const std::size_t source_bytes = source.pixels.size() * sizeof(Npp8u);
  const std::size_t destination_bytes =
      destination->pixels.size() * sizeof(Npp8u);

  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_source),
                            source_bytes),
                 "cudaMalloc device_source")) {
    return false;
  }

  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_destination),
                            destination_bytes),
                 "cudaMalloc device_destination")) {
    cudaFree(device_source);
    return false;
  }

  bool success = CheckCuda(
      cudaMemcpy(device_source, source.pixels.data(), source_bytes,
                 cudaMemcpyHostToDevice),
      "cudaMemcpy HostToDevice");

  cudaEvent_t start_event{};
  cudaEvent_t stop_event{};
  bool events_created = false;

  if (success) {
    events_created =
        CheckCuda(cudaEventCreate(&start_event), "cudaEventCreate start") &&
        CheckCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
    success = events_created;
  }

  if (success) {
    const NppiSize source_size{source.width, source.height};
    const NppiRect source_roi{0, 0, source.width, source.height};
    const NppiSize destination_size{destination->width, destination->height};
    const NppiRect destination_roi{0, 0, destination->width, destination->height};

    success =
        CheckCuda(cudaEventRecord(start_event, 0), "cudaEventRecord start");

    if (success) {
      success = CheckNpp(
          nppiResize_8u_C1R(
              device_source, source.width, source_size, source_roi,
              device_destination, destination->width, destination_size,
              destination_roi, NPPI_INTER_LINEAR),
          "nppiResize_8u_C1R (Bilinear Resampling)");
    }

    if (success) {
      success = CheckCuda(cudaEventRecord(stop_event, 0), "cudaEventRecord stop") &&
                CheckCuda(cudaEventSynchronize(stop_event),
                          "cudaEventSynchronize stop");
    }

    if (success && kernel_time_ms != nullptr) {
      success = CheckCuda(
          cudaEventElapsedTime(kernel_time_ms, start_event, stop_event),
          "cudaEventElapsedTime");
    }
  }

  if (events_created) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  if (success) {
    success = CheckCuda(
        cudaMemcpy(destination->pixels.data(), device_destination,
                   destination_bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy DeviceToHost");
  }

  cudaFree(device_source);
  cudaFree(device_destination);

  return success;
}

struct BatchRecord {
  int index = 0;
  std::string filename;
  int in_width = 0;
  int in_height = 0;
  double scale = 0.0;
  int out_width = 0;
  int out_height = 0;
  float gpu_time_ms = 0.0F;
  double throughput_mpps = 0.0;
  std::string status;
};

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    return 2;
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") ||
      device_count == 0) {
    std::cerr << "Fatal: No CUDA-capable GPU detected on this host.\n";
    return 1;
  }

  cudaDeviceProp properties{};
  if (!CheckCuda(cudaGetDeviceProperties(&properties, 0),
                 "cudaGetDeviceProperties")) {
    return 1;
  }

  std::cout << "============================================================\n"
            << "CUDA Enterprise Image Resampling Engine\n"
            << "============================================================\n"
            << "GPU Device:          " << properties.name << '\n'
            << "Compute Capability:  " << properties.major << '.'
            << properties.minor << '\n'
            << "Total Global Memory: "
            << properties.totalGlobalMem / (1024 * 1024) << " MiB\n"
            << "Multiprocessors:     " << properties.multiProcessorCount << '\n'
            << "Scale Factor:        " << options.scale << "x\n";

  // -------------------------------------------------------------------------
  // Mode 1: Single Image Mode
  // -------------------------------------------------------------------------
  if (!options.is_batch) {
    std::cout << "Execution Mode:      Single Image Resampling\n"
              << "Input Path:          " << options.input_path << '\n'
              << "Output Path:         " << options.output_path << '\n'
              << "------------------------------------------------------------\n";

    PgmImage source;
    std::string error;
    if (!ReadPgm(options.input_path, &source, &error)) {
      std::cerr << "Input error: " << error << '\n';
      return 1;
    }

    std::cout << "Input Dimensions:    " << source.width << " x " << source.height
              << " (" << (source.pixels.size() / 1024) << " KB)\n";

    PgmImage destination;
    float kernel_time_ms = 0.0F;
    if (!ResizeImageGpu(source, options.scale, &destination, &kernel_time_ms)) {
      std::cerr << "GPU image resizing failed.\n";
      return 1;
    }

    std::cout << "Output Dimensions:   " << destination.width << " x "
              << destination.height << " (" << (destination.pixels.size() / 1024)
              << " KB)\n"
              << "GPU Kernel Time:     " << std::fixed << std::setprecision(3)
              << kernel_time_ms << " ms\n";

    const double megapixels =
        static_cast<double>(destination.width * destination.height) / 1e6;
    const double throughput_mpps = megapixels / (kernel_time_ms / 1000.0);
    std::cout << "Throughput:          " << std::fixed << std::setprecision(2)
              << throughput_mpps << " Megapixels/sec\n";

    if (!WritePgm(options.output_path, destination, &error)) {
      std::cerr << "Output write error: " << error << '\n';
      return 1;
    }

    std::cout << "Status:              SUCCESS\n"
              << "Saved Image:         " << options.output_path << '\n'
              << "============================================================\n";

    if (!options.benchmark_csv.empty()) {
      std::ofstream csv(options.benchmark_csv, std::ios::app);
      if (csv) {
        csv << options.input_path << ',' << source.width << ',' << source.height
            << ',' << options.scale << ',' << destination.width << ','
            << destination.height << ',' << kernel_time_ms << ','
            << throughput_mpps << ",SUCCESS\n";
      }
    }
    return 0;
  }

  // -------------------------------------------------------------------------
  // Mode 2: Enterprise Batch Processing Mode (Single Execution for Entire Set)
  // -------------------------------------------------------------------------
  std::cout << "Execution Mode:      Enterprise Batch Processing (Single Run)\n"
            << "Batch Directory:     " << options.batch_dir << '\n'
            << "Output Directory:    " << options.output_dir << '\n'
            << "------------------------------------------------------------\n";

  if (!std::filesystem::exists(options.batch_dir)) {
    std::cerr << "Batch input directory does not exist: " << options.batch_dir
              << '\n';
    return 1;
  }

  std::filesystem::create_directories(options.output_dir);

  std::vector<std::filesystem::path> input_files;
  for (const auto& entry :
       std::filesystem::directory_iterator(options.batch_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
      input_files.push_back(entry.path());
    }
  }

  std::sort(input_files.begin(), input_files.end());

  if (input_files.empty()) {
    std::cerr << "No .pgm files found in batch directory: " << options.batch_dir
              << '\n';
    return 1;
  }

  std::cout << "Discovered " << input_files.size()
            << " image(s) for batch processing.\n"
            << "Beginning single-execution GPU batch pipeline...\n\n";

  std::vector<BatchRecord> records;
  records.reserve(input_files.size());

  double total_input_pixels = 0.0;
  double total_output_pixels = 0.0;
  float cumulative_kernel_ms = 0.0F;

  const auto wall_clock_start = std::chrono::high_resolution_clock::now();

  for (std::size_t i = 0; i < input_files.size(); ++i) {
    const auto& file_path = input_files[i];
    const std::string filename = file_path.filename().string();
    const std::string output_file =
        (std::filesystem::path(options.output_dir) / filename).string();

    BatchRecord rec;
    rec.index = static_cast<int>(i + 1);
    rec.filename = filename;
    rec.scale = options.scale;

    PgmImage source;
    std::string error;
    if (!ReadPgm(file_path.string(), &source, &error)) {
      std::cerr << "[Batch " << std::setw(3) << std::setfill('0') << rec.index
                << "/" << input_files.size() << "] FAILED to read " << filename
                << ": " << error << '\n';
      rec.status = "FAILED_READ";
      records.push_back(rec);
      continue;
    }

    rec.in_width = source.width;
    rec.in_height = source.height;

    PgmImage destination;
    float kernel_ms = 0.0F;
    if (!ResizeImageGpu(source, options.scale, &destination, &kernel_ms)) {
      std::cerr << "[Batch " << std::setw(3) << std::setfill('0') << rec.index
                << "/" << input_files.size() << "] GPU FAILED on " << filename
                << '\n';
      rec.status = "FAILED_GPU";
      records.push_back(rec);
      continue;
    }

    rec.out_width = destination.width;
    rec.out_height = destination.height;
    rec.gpu_time_ms = kernel_ms;

    const double out_mp =
        static_cast<double>(destination.width * destination.height) / 1e6;
    rec.throughput_mpps =
        (kernel_ms > 0.0F) ? (out_mp / (kernel_ms / 1000.0F)) : 0.0;

    if (!WritePgm(output_file, destination, &error)) {
      std::cerr << "[Batch " << std::setw(3) << std::setfill('0') << rec.index
                << "/" << input_files.size() << "] FAILED to write "
                << output_file << ": " << error << '\n';
      rec.status = "FAILED_WRITE";
      records.push_back(rec);
      continue;
    }

    rec.status = "SUCCESS";
    records.push_back(rec);

    total_input_pixels +=
        static_cast<double>(source.width) * source.height;
    total_output_pixels +=
        static_cast<double>(destination.width) * destination.height;
    cumulative_kernel_ms += kernel_ms;

    std::cout << "[Batch " << std::setw(3) << std::setfill('0') << rec.index
              << "/" << input_files.size() << "] " << std::setw(20)
              << std::left << filename << " (" << source.width << "x"
              << source.height << " -> " << destination.width << "x"
              << destination.height << ") | GPU: " << std::fixed
              << std::setprecision(2) << std::setw(6) << std::right << kernel_ms
              << " ms | " << std::setw(7) << std::setprecision(1)
              << rec.throughput_mpps << " MP/s | SUCCESS\n";
  }

  const auto wall_clock_end = std::chrono::high_resolution_clock::now();
  const double wall_clock_total_ms =
      std::chrono::duration<double, std::milli>(wall_clock_end -
                                                wall_clock_start)
          .count();

  int successful_count = 0;
  for (const auto& rec : records) {
    if (rec.status == "SUCCESS") ++successful_count;
  }

  const double avg_kernel_ms = (successful_count > 0)
                                   ? (cumulative_kernel_ms / successful_count)
                                   : 0.0;
  const double batch_images_per_sec =
      (wall_clock_total_ms > 0.0)
          ? (successful_count / (wall_clock_total_ms / 1000.0))
          : 0.0;
  const double gpu_pixel_throughput_mpps =
      (cumulative_kernel_ms > 0.0F)
          ? ((total_output_pixels / 1e6) / (cumulative_kernel_ms / 1000.0))
          : 0.0;

  std::cout << "\n============================================================\n"
            << "Enterprise GPU Batch Processing Summary\n"
            << "============================================================\n"
            << "Total Items Discovered:       " << input_files.size() << '\n'
            << "Successfully Processed:       " << successful_count << " / "
            << input_files.size() << '\n'
            << "Total Input Data Volume:      " << std::fixed
            << std::setprecision(2) << (total_input_pixels / 1e6)
            << " Megapixels (" << (total_input_pixels / (1024 * 1024))
            << " MB raw)\n"
            << "Total Output Data Volume:     " << std::fixed
            << std::setprecision(2) << (total_output_pixels / 1e6)
            << " Megapixels (" << (total_output_pixels / (1024 * 1024))
            << " MB raw)\n"
            << "Cumulative GPU Kernel Time:   " << std::fixed
            << std::setprecision(2) << cumulative_kernel_ms << " ms\n"
            << "Average GPU Latency per Image:" << std::fixed
            << std::setprecision(3) << avg_kernel_ms << " ms\n"
            << "Total Batch Wall-Clock Time:  " << std::fixed
            << std::setprecision(2) << wall_clock_total_ms << " ms\n"
            << "Batch Processing Rate:        " << std::fixed
            << std::setprecision(1) << batch_images_per_sec
            << " images / second\n"
            << "GPU Resampling Throughput:    " << std::fixed
            << std::setprecision(2) << gpu_pixel_throughput_mpps
            << " Megapixels / second\n"
            << "Overall Batch Execution Status: "
            << ((successful_count == static_cast<int>(input_files.size()))
                    ? "ALL PASSED (100% SUCCESS)"
                    : "PARTIAL SUCCESS")
            << '\n'
            << "============================================================\n";

  // Write batch results CSV if specified
  if (!options.benchmark_csv.empty()) {
    std::ofstream csv(options.benchmark_csv);
    if (csv) {
      csv << "BatchIndex,Filename,InputWidth,InputHeight,Scale,OutputWidth,"
             "OutputHeight,GpuKernelTimeMs,ThroughputMPps,Status\n";
      for (const auto& rec : records) {
        csv << rec.index << ',' << rec.filename << ',' << rec.in_width << ','
            << rec.in_height << ',' << rec.scale << ',' << rec.out_width << ','
            << rec.out_height << ',' << std::fixed << std::setprecision(3)
            << rec.gpu_time_ms << ',' << std::setprecision(2)
            << rec.throughput_mpps << ',' << rec.status << '\n';
      }
      std::cout << "Batch benchmark metrics saved to: " << options.benchmark_csv
                << "\n\n";
    }
  }

  return (successful_count == static_cast<int>(input_files.size())) ? 0 : 1;
}
