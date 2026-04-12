#!/bin/bash
echo "Running Kernel Benchmark"
cd "$(dirname "$0")/.."

RUSTFLAGS="-C target-cpu=native" cargo criterion --bench kernel --features="avx2"
