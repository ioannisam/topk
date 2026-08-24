#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/find_root.sh"
ROOT_DIR="$(find_root_dir "${SCRIPT_DIR}")" || exit 1

RESULTS_DIR="${ROOT_DIR}/bench/results"
mkdir -p "${RESULTS_DIR}/raw/roofline"

python3 "${ROOT_DIR}/bench/lib/roofline.py" cpu gpu npu \
    --exp all --bytes 256M --cooldown 10 \
    --output-json "${RESULTS_DIR}/raw/roofline/roofline.json" \
    --output-raw "${RESULTS_DIR}/raw/roofline/roofline.txt" "$@"
