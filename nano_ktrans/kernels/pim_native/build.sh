#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
mkdir -p "${BUILD_DIR}"

TASKLETS="${TASKLETS:-16}"
BLOCK_FLOATS="${BLOCK_FLOATS:-64}"
MAX_WEIGHT_FLOATS="${MAX_WEIGHT_FLOATS:-2097152}"
MAX_QWEIGHT_WORDS="${MAX_QWEIGHT_WORDS:-2097152}"
MAX_SCALE_FLOATS="${MAX_SCALE_FLOATS:-65536}"
MAX_INPUT_FLOATS="${MAX_INPUT_FLOATS:-65536}"
MAX_OUTPUT_FLOATS="${MAX_OUTPUT_FLOATS:-65536}"
STACK_SIZE_DEFAULT="${STACK_SIZE_DEFAULT:-2048}"
MAX_INTERMEDIATE_DIM="${MAX_INTERMEDIATE_DIM:-2048}"

echo "Building DPU linear kernel..." >&2
dpu-upmem-dpurte-clang \
  -O3 \
  -DNR_TASKLETS="${TASKLETS}" \
  -DBLOCK_FLOATS="${BLOCK_FLOATS}" \
  -DMAX_INTERMEDIATE_DIM="${MAX_INTERMEDIATE_DIM}" \
  -DMAX_WEIGHT_FLOATS="${MAX_WEIGHT_FLOATS}" \
  -DMAX_INPUT_FLOATS="${MAX_INPUT_FLOATS}" \
  -DMAX_OUTPUT_FLOATS="${MAX_OUTPUT_FLOATS}" \
  -DSTACK_SIZE_DEFAULT="${STACK_SIZE_DEFAULT}" \
  -o "${BUILD_DIR}/pim_linear_kernel" \
  "${ROOT_DIR}/dpu_linear_kernel.c"

echo "Building host bridge..." >&2
gcc \
  -O3 \
  -std=c11 \
  -shared \
  -fPIC \
  -DBLOCK_FLOATS="${BLOCK_FLOATS}" \
  -DMAX_INTERMEDIATE_DIM="${MAX_INTERMEDIATE_DIM}" \
  -DMAX_WEIGHT_FLOATS="${MAX_WEIGHT_FLOATS}" \
  -DMAX_INPUT_FLOATS="${MAX_INPUT_FLOATS}" \
  -DMAX_OUTPUT_FLOATS="${MAX_OUTPUT_FLOATS}" \
  "${ROOT_DIR}/host_bridge.c" \
  -o "${BUILD_DIR}/libpim_linear_bridge.so" \
  $(pkg-config --cflags --libs dpu)

echo "Building DPU quantized kernel..." >&2
dpu-upmem-dpurte-clang \
  -O3 \
  -DNR_TASKLETS="${TASKLETS}" \
  -DBLOCK_FLOATS="${BLOCK_FLOATS}" \
  -DMAX_QWEIGHT_WORDS="${MAX_QWEIGHT_WORDS}" \
  -DMAX_SCALE_FLOATS="${MAX_SCALE_FLOATS}" \
  -DMAX_INPUT_FLOATS="${MAX_INPUT_FLOATS}" \
  -DMAX_OUTPUT_FLOATS="${MAX_OUTPUT_FLOATS}" \
  -DSTACK_SIZE_DEFAULT="${STACK_SIZE_DEFAULT}" \
  -o "${BUILD_DIR}/pim_quantized_kernel" \
  "${ROOT_DIR}/dpu_quantized_kernel.c"

echo "Building quantized host bridge..." >&2
gcc \
  -O3 \
  -std=c11 \
  -shared \
  -fPIC \
  -DBLOCK_FLOATS="${BLOCK_FLOATS}" \
  -DMAX_QWEIGHT_WORDS="${MAX_QWEIGHT_WORDS}" \
  -DMAX_SCALE_FLOATS="${MAX_SCALE_FLOATS}" \
  -DMAX_INPUT_FLOATS="${MAX_INPUT_FLOATS}" \
  -DMAX_OUTPUT_FLOATS="${MAX_OUTPUT_FLOATS}" \
  "${ROOT_DIR}/host_quantized_bridge.c" \
  -o "${BUILD_DIR}/libpim_quantized_bridge.so" \
  $(pkg-config --cflags --libs dpu)

echo "Building DPU expert kernel..." >&2
dpu-upmem-dpurte-clang \
  -O3 \
  -DNR_TASKLETS="${TASKLETS}" \
  -DBLOCK_FLOATS="${BLOCK_FLOATS}" \
  -DMAX_INTERMEDIATE_DIM="${MAX_INTERMEDIATE_DIM}" \
  -DMAX_WEIGHT_FLOATS="${MAX_WEIGHT_FLOATS}" \
  -DMAX_INPUT_FLOATS="${MAX_INPUT_FLOATS}" \
  -DMAX_OUTPUT_FLOATS="${MAX_OUTPUT_FLOATS}" \
  -DSTACK_SIZE_DEFAULT="${STACK_SIZE_DEFAULT}" \
  -o "${BUILD_DIR}/pim_expert_kernel" \
  "${ROOT_DIR}/dpu_expert_kernel.c"

echo "Building expert host bridge..." >&2
gcc \
  -O3 \
  -std=c11 \
  -shared \
  -fPIC \
  -DBLOCK_FLOATS="${BLOCK_FLOATS}" \
  -DMAX_INTERMEDIATE_DIM="${MAX_INTERMEDIATE_DIM}" \
  -DMAX_WEIGHT_FLOATS="${MAX_WEIGHT_FLOATS}" \
  -DMAX_INPUT_FLOATS="${MAX_INPUT_FLOATS}" \
  -DMAX_OUTPUT_FLOATS="${MAX_OUTPUT_FLOATS}" \
  "${ROOT_DIR}/host_expert_bridge.c" \
  -o "${BUILD_DIR}/libpim_expert_bridge.so" \
  $(pkg-config --cflags --libs dpu)

echo "Built:" >&2
echo "  ${BUILD_DIR}/pim_linear_kernel" >&2
echo "  ${BUILD_DIR}/libpim_linear_bridge.so" >&2
echo "  ${BUILD_DIR}/pim_quantized_kernel" >&2
echo "  ${BUILD_DIR}/libpim_quantized_bridge.so" >&2
echo "  ${BUILD_DIR}/pim_expert_kernel" >&2
echo "  ${BUILD_DIR}/libpim_expert_bridge.so" >&2
