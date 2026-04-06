#!/bin/bash
echo "Running Matrix Benchmark"
cd "$(dirname "$0")/.."

cargo criterion --bench matrix
