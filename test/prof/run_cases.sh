#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
while [[ "${ROOT_DIR}" != "/" && ! -f "${ROOT_DIR}/CMakeLists.txt" ]]; do
    ROOT_DIR="$(dirname "${ROOT_DIR}")"
done
if [[ ! -f "${ROOT_DIR}/CMakeLists.txt" ]]; then
    echo "error: could not locate repo root (CMakeLists.txt)" >&2
    exit 1
fi

RESULTS_DIR="${ROOT_DIR}/test/prof/results"

NPU_OFFLOAD_XCLBIN="${ROOT_DIR}/build/NPU/bitonic.xclbin" \
    python3 "${ROOT_DIR}/test/prof/runner.py" cpu gpu npu \
    --types int uint float double fp16 --q-max 24 \
    --cooldown 15 --min 0 --max 1000000000 \
    --output-json "${RESULTS_DIR}/cases_output.json" \
    --output-raw "${RESULTS_DIR}/cases_output.txt" "$@"
