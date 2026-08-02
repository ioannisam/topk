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
mkdir -p "${RESULTS_DIR}/raw/roofline"

python3 "${ROOT_DIR}/bench/lib/roofline.py" cpu gpu npu \
    --exp all --bytes 256M --cooldown 10 \
    --output-json "${RESULTS_DIR}/raw/roofline/roofline.json" \
    --output-raw "${RESULTS_DIR}/raw/roofline/roofline.txt" "$@"
