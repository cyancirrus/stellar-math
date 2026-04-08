#!/bin/bash
echo "Running Kernel Benchmark"
cd "$(dirname "$0")/.."

cargo criterion --bench kernel
