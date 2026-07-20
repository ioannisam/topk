#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
while [[ "${ROOT_DIR}" != "/" && ! -f "${ROOT_DIR}/CMakeLists.txt" ]]; do
    ROOT_DIR="$(dirname "${ROOT_DIR}")"
done
if [[ ! -f "${ROOT_DIR}/CMakeLists.txt" ]]; then
    echo "error: could not locate repo root (CMakeLists.txt)" >&2
    exit 1
fi

BENCH_DIR="${ROOT_DIR}/bench"
RESULTS_DIR="${BENCH_DIR}/results"

echo "Cleaning benchmark artifacts..."

if [[ -d "${RESULTS_DIR}" ]]; then
    for sub in raw derived plots; do
        if [[ -d "${RESULTS_DIR}/${sub}" ]]; then
            echo "  -> Emptying ${sub}/"
            rm -rf "${RESULTS_DIR:?}/${sub}/"*
        fi
    done
else
    echo "  -> No results directory found at ${RESULTS_DIR}, skipping."
fi

echo "  -> Removing Python __pycache__ directories in ${BENCH_DIR}"
find "${BENCH_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "Benchmark cleanup complete!"
