#!/bin/bash
echo "Running Matrix Benchmark"
cd "$(dirname "$0")/.."

RAYON_NUM_THREADS=4 cargo criterion --bench matrix --features="avx2"
# cargo criterion --bench matrix --features="avx2"
