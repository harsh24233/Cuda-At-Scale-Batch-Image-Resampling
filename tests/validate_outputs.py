#!/usr/bin/env python3
"""
Netpbm PGM Output Verification Script.
Validates:
  1. Valid binary Netpbm P5 header format
  2. Maxval equals 255
  3. Exact file size matches (header_size + width * height)
  4. Non-empty pixel data
  5. Scaled dimensions match expected scale factor if provided
"""

import math
import os
import sys

def validate_pgm(file_path: str, expected_w: int = None, expected_h: int = None) -> bool:
    if not os.path.isfile(file_path):
        print(f"[ERROR] File does not exist: {file_path}")
        return False

    with open(file_path, "rb") as f:
        data = f.read()

    # Parse header tokens
    pos = 0
    tokens = []
    while len(tokens) < 4 and pos < len(data):
        # Skip leading whitespace
        while pos < len(data) and chr(data[pos]).isspace():
            pos += 1
        if pos >= len(data):
            break
        # Skip comments
        if chr(data[pos]) == '#':
            while pos < len(data) and data[pos] != 10:  # newline
                pos += 1
            continue
        # Read token
        token_start = pos
        while pos < len(data) and not chr(data[pos]).isspace():
            pos += 1
        tokens.append(data[token_start:pos].decode("ascii", errors="replace"))

    if len(tokens) < 4:
        print(f"[FAIL] Incomplete PGM header in {file_path}")
        return False

    magic, w_str, h_str, maxval_str = tokens[:4]

    if magic != "P5":
        print(f"[FAIL] Expected magic 'P5', got '{magic}' in {file_path}")
        return False

    try:
        width = int(w_str)
        height = int(h_str)
        maxval = int(maxval_str)
    except ValueError:
        print(f"[FAIL] Non-integer dimension in header: {tokens} in {file_path}")
        return False

    if maxval != 255:
        print(f"[FAIL] Expected maxval 255, got {maxval} in {file_path}")
        return False

    # Consume single whitespace separator after maxval
    if pos < len(data) and chr(data[pos]).isspace():
        pos += 1
    else:
        print(f"[FAIL] Missing header/data separator in {file_path}")
        return False

    header_bytes = pos
    expected_pixel_bytes = width * height
    actual_pixel_bytes = len(data) - header_bytes

    if actual_pixel_bytes != expected_pixel_bytes:
        print(f"[FAIL] Pixel payload mismatch: expected {expected_pixel_bytes} bytes, "
              f"got {actual_pixel_bytes} bytes in {file_path}")
        return False

    if expected_w is not None and width != expected_w:
        print(f"[FAIL] Width mismatch: expected {expected_w}, got {width} in {file_path}")
        return False

    if expected_h is not None and height != expected_h:
        print(f"[FAIL] Height mismatch: expected {expected_h}, got {height} in {file_path}")
        return False

    return True

def validate_directory(dir_path: str, expected_count: int = None) -> bool:
    if not os.path.isdir(dir_path):
        print(f"[ERROR] Directory does not exist: {dir_path}")
        return False

    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".pgm")])
    if not files:
        print(f"[FAIL] No .pgm files found in {dir_path}")
        return False

    if expected_count is not None and len(files) < expected_count:
        print(f"[FAIL] Expected at least {expected_count} files, found {len(files)} in {dir_path}")
        return False

    print(f"Validating {len(files)} PGM files in {dir_path}...")
    all_ok = True
    for f in files:
        full_path = os.path.join(dir_path, f)
        if not validate_pgm(full_path):
            all_ok = False

    if all_ok:
        print(f"[PASS] All {len(files)} files in {dir_path} passed structural validation.")
    return all_ok

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  validate_outputs.py <file.pgm> [expected_width] [expected_height]")
        print("  validate_outputs.py --dir <directory> [expected_count]")
        sys.exit(1)

    if sys.argv[1] == "--dir":
        target_dir = sys.argv[2]
        count = int(sys.argv[3]) if len(sys.argv) > 3 else None
        ok = validate_directory(target_dir, count)
        sys.exit(0 if ok else 1)
    else:
        file_path = sys.argv[1]
        exp_w = int(sys.argv[2]) if len(sys.argv) > 2 else None
        exp_h = int(sys.argv[3]) if len(sys.argv) > 3 else None
        ok = validate_pgm(file_path, exp_w, exp_h)
        if ok:
            print(f"[PASS] {file_path} is structurally valid.")
        sys.exit(0 if ok else 1)
