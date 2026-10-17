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

#ifndef PGM_IMAGE_H_
#define PGM_IMAGE_H_

#include <cstdint>
#include <string>
#include <vector>

// Represents an 8-bit single-channel grayscale Netpbm PGM (P5 binary) image.
struct PgmImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
};

// Reads a binary Netpbm PGM (P5 format) file from disk into a PgmImage struct.
// Returns true on success; returns false and populates error on failure.
bool ReadPgm(const std::string& path, PgmImage* image, std::string* error);

// Writes a PgmImage struct to disk in binary Netpbm PGM (P5 format).
// Returns true on success; returns false and populates error on failure.
bool WritePgm(const std::string& path, const PgmImage& image,
              std::string* error);

#endif  // PGM_IMAGE_H_
