#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/find_root.sh"
ROOT_DIR="$(find_root_dir "${SCRIPT_DIR}")" || exit 1

RESULTS_DIR="${ROOT_DIR}/bench/results"
mkdir -p "${RESULTS_DIR}/raw/dists"

NPU_OFFLOAD_XCLBIN="${ROOT_DIR}/build/NPU/bitonic.xclbin" \
    python3 "${ROOT_DIR}/bench/lib/runner.py" cpu gpu npu \
    --types float half --q-min 20 --q-max 24 --k 8 131072 \
    --dists uniform trimodal adversarial sorted reverse --seeds 2 \
    --cooldown 15 --min 0 --max 1000000000 \
    --output-json "${RESULTS_DIR}/raw/dists/output.json" \
    --output-raw "${RESULTS_DIR}/raw/dists/output.txt" "$@"
