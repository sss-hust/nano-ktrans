#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"${ROOT_DIR}/benchmarks/pim_microbench/build.sh"
"${ROOT_DIR}/benchmarks/pim_microbench/build/pim_microbench_host" "$@"
