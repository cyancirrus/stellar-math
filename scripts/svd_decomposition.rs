#!/bin/bash
echo "Running SVD Benchmark"
cd "$(dirname "$0")/.."

cargo criterion --bench svd
