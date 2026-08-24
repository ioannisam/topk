#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/find_root.sh"
ROOT_DIR="$(find_root_dir "${SCRIPT_DIR}")" || exit 1

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
