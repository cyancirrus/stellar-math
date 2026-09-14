#!/bin/bash
echo "Running LQ Benchmark"
cd "$(dirname "$0")/.."

cargo criterion --bench lq
