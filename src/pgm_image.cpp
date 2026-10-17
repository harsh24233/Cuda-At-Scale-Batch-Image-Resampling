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

#include "pgm_image.h"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Helper function to extract a single token from an input stream,
// correctly skipping whitespace and comment lines starting with '#'.
bool ReadToken(std::istream& input, std::string* token) {
  token->clear();
  while (input) {
    input >> std::ws;
    if (input.peek() == '#') {
      std::string comment;
      std::getline(input, comment);
      continue;
    }
    break;
  }
  return static_cast<bool>(input >> *token);
}

}  // namespace

bool ReadPgm(const std::string& path, PgmImage* image, std::string* error) {
  if (image == nullptr) {
    if (error != nullptr) *error = "Null image pointer provided.";
    return false;
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    if (error != nullptr) *error = "Could not open input file: " + path;
    return false;
  }

  std::string magic;
  std::string width_token;
  std::string height_token;
  std::string max_token;

  if (!ReadToken(input, &magic) || magic != "P5" ||
      !ReadToken(input, &width_token) ||
      !ReadToken(input, &height_token) ||
      !ReadToken(input, &max_token)) {
    if (error != nullptr) *error = "Expected a valid binary PGM (P5) header.";
    return false;
  }

  try {
    image->width = std::stoi(width_token);
    image->height = std::stoi(height_token);
    const int max_value = std::stoi(max_token);
    if (image->width <= 0 || image->height <= 0 || max_value != 255) {
      if (error != nullptr) {
        *error = "PGM must have positive dimensions and max value 255.";
      }
      return false;
    }
  } catch (const std::exception&) {
    if (error != nullptr) *error = "Invalid numeric value in PGM header.";
    return false;
  }

  // Consume the single whitespace character separating the header and pixels.
  const int separator = input.get();
  if (separator == EOF || !std::isspace(static_cast<unsigned char>(separator))) {
    if (error != nullptr) *error = "Missing PGM header/pixel separator.";
    return false;
  }

  const std::uint64_t pixel_count =
      static_cast<std::uint64_t>(image->width) * image->height;
  if (pixel_count > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
    if (error != nullptr) *error = "Image dimensions exceed host memory limit.";
    return false;
  }

  image->pixels.resize(static_cast<std::size_t>(pixel_count));
  input.read(reinterpret_cast<char*>(image->pixels.data()),
             static_cast<std::streamsize>(image->pixels.size()));
  if (input.gcount() != static_cast<std::streamsize>(image->pixels.size())) {
    if (error != nullptr) {
      *error = "Input image pixel data is incomplete or truncated.";
    }
    return false;
  }
  return true;
}

bool WritePgm(const std::string& path, const PgmImage& image,
              std::string* error) {
  if (image.width <= 0 || image.height <= 0) {
    if (error != nullptr) {
      *error = "Cannot write image with non-positive dimensions.";
    }
    return false;
  }

  const std::uint64_t expected_pixels =
      static_cast<std::uint64_t>(image.width) * image.height;
  if (image.pixels.size() != expected_pixels) {
    if (error != nullptr) {
      *error = "Image pixel buffer size does not match dimensions.";
    }
    return false;
  }

  std::ofstream output(path, std::ios::binary);
  if (!output) {
    if (error != nullptr) *error = "Could not open output file: " + path;
    return false;
  }

  output << "P5\n" << image.width << ' ' << image.height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
  if (!output) {
    if (error != nullptr) *error = "Failed while writing output image: " + path;
    return false;
  }
  return true;
}
