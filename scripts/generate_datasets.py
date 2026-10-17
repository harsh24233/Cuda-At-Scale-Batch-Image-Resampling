#!/usr/bin/env python3
"""
Synthetic PGM Dataset Generator for CUDA Batch Processing Benchmark.
Generates:
  1. 100 distinct small images (128x128) in data/batch_small/
  2. 10 distinct large high-resolution images (2048x1536) in data/batch_large/
  3. 1 reference sample image (320x240) in data/sample/sample.pgm

All images are generated in Netpbm PGM (P5 binary) format with deterministic
synthetic patterns (gradients, checkerboards, radial chirps, sinusoidal waves).
Requires standard Python 3 with no third-party library dependencies.
"""

import math
import os
import sys

def write_pgm(path: str, width: int, height: int, pixels: bytearray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        f.write(header)
        f.write(pixels)

def generate_small_batch(output_dir: str, count: int = 100):
    width, height = 128, 128
    print(f"Generating {count} small images ({width}x{height}) in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, count + 1):
        pixels = bytearray(width * height)
        pattern_type = i % 5
        
        for y in range(height):
            y_norm = y / height
            row_offset = y * width
            for x in range(width):
                x_norm = x / width
                if pattern_type == 0:
                    # Diagonal gradient modulated by index
                    val = int((x_norm + y_norm) * 127.5 * (1.0 + 0.5 * math.sin(i))) % 256
                elif pattern_type == 1:
                    # Concentric rings / radial chirp
                    dx = x_norm - 0.5
                    dy = y_norm - 0.5
                    dist = math.sqrt(dx * dx + dy * dy)
                    val = int(127.5 * (1.0 + math.sin(dist * (10.0 + i % 15))))
                elif pattern_type == 2:
                    # Dynamic checkerboard
                    block_size = max(4, (i % 16) + 4)
                    val = 220 if ((x // block_size) + (y // block_size)) % 2 == 0 else 35
                elif pattern_type == 3:
                    # Sinusoidal interference pattern
                    val = int(127.5 * (1.0 + math.sin(x_norm * (i % 8 + 2) * math.pi) * 
                                             math.cos(y_norm * (i % 8 + 2) * math.pi)))
                else:
                    # Horizontal & vertical linear ramp
                    val = (x * 2 + y * 2 + i * 5) % 256
                
                pixels[row_offset + x] = min(255, max(0, val))
        
        file_name = f"image_{i:03d}.pgm"
        write_pgm(os.path.join(output_dir, file_name), width, height, pixels)

def generate_large_batch(output_dir: str, count: int = 10):
    width, height = 2048, 1536  # ~3.15 Megapixels each
    print(f"Generating {count} large images ({width}x{height}) in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, count + 1):
        pixels = bytearray(width * height)
        print(f"  Creating large image {i:02d}/{count:02d} ({width}x{height}, 3.15 MP)...")
        freq = 5.0 + i * 2.0
        
        for y in range(height):
            y_val = math.sin(y * math.pi * freq / height)
            row_offset = y * width
            for x in range(width):
                x_val = math.cos(x * math.pi * freq / width)
                val = int(127.5 * (1.0 + 0.5 * (x_val + y_val)))
                pixels[row_offset + x] = min(255, max(0, val))
                
        file_name = f"large_image_{i:02d}.pgm"
        write_pgm(os.path.join(output_dir, file_name), width, height, pixels)

def generate_sample_image(output_path: str):
    width, height = 320, 240
    print(f"Generating standard reference sample ({width}x{height}) at {output_path}...")
    pixels = bytearray(width * height)
    for y in range(height):
        row_offset = y * width
        for x in range(width):
            dx = (x / width) - 0.5
            dy = (y / height) - 0.5
            dist = math.sqrt(dx * dx + dy * dy)
            val = int(127.5 * (1.0 + math.cos(dist * 25.0)))
            pixels[row_offset + x] = min(255, max(0, val))
    write_pgm(output_path, width, height, pixels)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_path = os.path.join(base_dir, "data", "sample", "sample.pgm")
    small_batch_dir = os.path.join(base_dir, "data", "batch_small")
    large_batch_dir = os.path.join(base_dir, "data", "batch_large")
    
    generate_sample_image(sample_path)
    generate_small_batch(small_batch_dir, count=100)
    generate_large_batch(large_batch_dir, count=10)
    print("All datasets generated successfully!")
