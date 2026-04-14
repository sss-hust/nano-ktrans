#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/benchmarks/pim_microbench/build"
mkdir -p "${BUILD_DIR}"

TASKLETS="${TASKLETS:-16}"
MAX_BYTES_PER_DPU="${MAX_BYTES_PER_DPU:-8388608}"
BLOCK_BYTES="${BLOCK_BYTES:-1024}"
STACK_SIZE_DEFAULT="${STACK_SIZE_DEFAULT:-2048}"

echo "Building DPU kernel..." >&2
dpu-upmem-dpurte-clang \
  -O3 \
  -DNR_TASKLETS="${TASKLETS}" \
  -DMAX_BYTES_PER_DPU="${MAX_BYTES_PER_DPU}" \
  -DBLOCK_BYTES="${BLOCK_BYTES}" \
  -DSTACK_SIZE_DEFAULT="${STACK_SIZE_DEFAULT}" \
  -o "${BUILD_DIR}/pim_kernel" \
  "${ROOT_DIR}/benchmarks/pim_microbench/pim_kernel.c"

echo "Building host benchmark..." >&2
gcc \
  -O3 \
  -std=c11 \
  -DDPU_BINARY_PATH=\"${BUILD_DIR}/pim_kernel\" \
  "${ROOT_DIR}/benchmarks/pim_microbench/host.c" \
  -o "${BUILD_DIR}/pim_microbench_host" \
  $(pkg-config --cflags --libs dpu)

echo "Built:" >&2
echo "  ${BUILD_DIR}/pim_kernel" >&2
echo "  ${BUILD_DIR}/pim_microbench_host" >&2
