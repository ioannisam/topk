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

PROF_DIR="${ROOT_DIR}/test/prof"
RESULTS_DIR="${PROF_DIR}/results"

echo "Cleaning profiler artifacts..."
if [[ -d "${RESULTS_DIR}" ]]; then
    echo "  -> Removing log files and reports from ${RESULTS_DIR}"
    rm -f "${RESULTS_DIR}"/*.txt
    rm -f "${RESULTS_DIR}"/*.md
    rm -f "${RESULTS_DIR}"/*.json

    if [[ -d "${RESULTS_DIR}/plots" ]]; then
        echo "  -> Emptying plots directory"
        rm -rf "${RESULTS_DIR}/plots/"*
    fi

    if [[ -d "${RESULTS_DIR}/measurements" ]]; then
        echo "  -> Emptying measurements directory"
        rm -rf "${RESULTS_DIR}/measurements/"*
    fi

    if [[ -d "${RESULTS_DIR}/energy" ]]; then
        echo "  -> Emptying energy directory"
        rm -rf "${RESULTS_DIR}/energy/"*
    fi
else
    echo "  -> No results directory found at ${RESULTS_DIR}, skipping."
fi

echo "  -> Removing Python __pycache__ directories in ${PROF_DIR}"
find "${PROF_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "Profiler cleanup complete!"
