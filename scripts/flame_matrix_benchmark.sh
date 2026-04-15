#!/bin/bash
echo "Running Matrix Benchmark"
cd "$(dirname "$0")/.."

RAYON_NUM_THREADS=4 cargo criterion --bench matrix_flame --features="avx2"
open -a "Firefox"  "./diagnostics/flamegraph_matmul.svg"
