#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROF_DIR="${ROOT_DIR}/test/prof"
RESULTS_DIR="${PROF_DIR}/results"

echo "Cleaning profiler artifacts..."
if [[ -d "${RESULTS_DIR}" ]]; then
    echo "  -> Removing log files and reports from ${RESULTS_DIR}"
    rm -f "${RESULTS_DIR}"/*.txt
    rm -f "${RESULTS_DIR}"/*.md

    if [[ -d "${RESULTS_DIR}/plots" ]]; then
        echo "  -> Emptying plots directory"
        rm -rf "${RESULTS_DIR}/plots/"*
    fi

    if [[ -d "${RESULTS_DIR}/measurements" ]]; then
        echo "  -> Emptying measurements directory"
        rm -rf "${RESULTS_DIR}/measurements/"*
    fi
else
    echo "  -> No results directory found at ${RESULTS_DIR}, skipping."
fi

echo "  -> Removing Python __pycache__ directories in ${PROF_DIR}"
find "${PROF_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "Profiler cleanup complete!"
