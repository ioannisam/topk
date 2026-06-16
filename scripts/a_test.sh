#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/lib.sh"

usage() {
    echo "Usage: $(basename "$0") [cpu|gpu|npu] [bitonic|map_reduce|gt|all]"
    echo "  Reports the performance of the current codebase for the specified backend and algorithm."
    echo "  Defaults: backend=cpu, algorithm=all."
}

parse_backend_algo "${1:-cpu}" "${2:-all}"

ROOT_BUILD_DIR="${ROOT_DIR}/build"
EXECUTABLE="${ROOT_BUILD_DIR}/${BACKEND_DIR}/topk"
TIMEOUT_SECONDS="${TOPK_TEST_TIMEOUT_SECONDS:-300}"

cd "${ROOT_DIR}" || { echo "Error: Cannot cd to ${ROOT_DIR}" >&2; exit 1; }

TMP_OUT="$(mktemp)"

cleanup() {
    rm -f "${TMP_OUT}"
    echo -e "\n>> Restoring root build to all backends..."
    configure_root_build "ON" "ON" "ON" >/dev/null 2>&1 || echo "Warning: Failed to restore root build settings." >&2
}
trap cleanup EXIT

echo -e "Backend: ${BACKEND_DIR} | Target: ${TARGET_ALGO}\nExecutable: ${EXECUTABLE}"

if ! run_suite "Current" "${TMP_OUT}"; then exit 1; fi

echo -e "\n=== Performance Report ==="
printf "| %-15s | %-12s | %-15s | %-6s | %-14s | %-15s |\n" \
    "Case" "Size (q/k)" "Algorithm" "Type" "E2E (ms)" "Algo (ms)"
printf "|-----------------|--------------|-----------------|--------|----------------|-----------------|\n"

while read -r case_name q k algo dtype e2e algo_ms; do
    printf "| %-15s | %-12s | %-15s | %-6s | %-14s | %-15s |\n" \
        "${case_name}" "q=${q}/k=${k}" "${algo}" "${dtype}" "${e2e}" "${algo_ms}"
done < "${TMP_OUT}"
