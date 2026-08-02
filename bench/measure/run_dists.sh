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

RESULTS_DIR="${ROOT_DIR}/bench/results"
mkdir -p "${RESULTS_DIR}/raw/dists"

NPU_OFFLOAD_XCLBIN="${ROOT_DIR}/build/NPU/bitonic.xclbin" \
    python3 "${ROOT_DIR}/bench/lib/runner.py" cpu gpu npu \
    --types float half --q-min 20 --q-max 24 --k 8 131072 \
    --dists uniform trimodal adversarial sorted reverse --seeds 2 \
    --cooldown 15 --min 0 --max 1000000000 \
    --output-json "${RESULTS_DIR}/raw/dists/output.json" \
    --output-raw "${RESULTS_DIR}/raw/dists/output.txt" "$@"
