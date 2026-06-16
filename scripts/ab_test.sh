#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/lib.sh"

usage() {
    echo "Usage: $(basename "$0") [cpu|gpu|npu] [bitonic|map_reduce|gt|all]"
    echo "  Compares the performance of the current uncommitted changes"
    echo "  against the HEAD commit for the specified backend and algorithm."
    echo "  Defaults: backend=cpu, algorithm=all."
}

parse_backend_algo "${1:-cpu}" "${2:-all}"

ROOT_BUILD_DIR="${ROOT_DIR}/build"
EXECUTABLE="${ROOT_BUILD_DIR}/${BACKEND_DIR}/topk"
TIMEOUT_SECONDS="${TOPK_TEST_TIMEOUT_SECONDS:-300}"

cd "${ROOT_DIR}" || { echo "Error: Cannot cd to ${ROOT_DIR}" >&2; exit 1; }
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then echo "Error: Not a git repo." >&2; exit 1; fi
if git diff --quiet && git diff --cached --quiet; then echo "Error: No uncommitted changes to test against HEAD!" >&2; exit 1; fi

TMP_NEW="$(mktemp)"; TMP_BASE="$(mktemp)"; STASHED=0

cleanup() {
    rm -f "${TMP_NEW}" "${TMP_BASE}"
    [[ ${STASHED} -eq 1 ]] && { echo -e "\nRestoring stashed changes..."; git stash pop -q || echo "Warning: Failed to pop git stash." >&2; }
    echo -e "\n>> Restoring root build to all backends..."
    configure_root_build "ON" "ON" "ON" >/dev/null 2>&1 || echo "Warning: Failed to restore root build settings." >&2
}
trap cleanup EXIT

echo -e "Backend: ${BACKEND_DIR} | Target: ${TARGET_ALGO}\nExecutable: ${EXECUTABLE}"

echo -e "\n=== Phase 1: Testing CURRENT uncommitted changes ==="
if ! run_suite "Current Changes" "${TMP_NEW}"; then exit 1; fi

echo -e "\n=== Phase 2: Testing BASELINE (HEAD) ==="
echo "Stashing current changes..."
git stash push -q -u -m "perf_ab_test_temp"
STASHED=1

if ! run_suite "Baseline (HEAD)" "${TMP_BASE}"; then exit 1; fi

echo -e "\n=== Performance Report ==="
printf "| %-10s | %-12s | %-6s | %-12s | %-14s | %-15s | %-12s | %-13s | %-12s | %-13s |\n" \
    "Case" "Size (q/k)" "Type" "Algorithm" "Base E2E (ms)" "Base Algo (ms)" "New E2E (ms)" "New Algo (ms)" "Speedup E2E" "Speedup Algo"
printf "|------------|--------------|--------|--------------|----------------|-----------------|--------------|---------------|--------------|---------------|\n"

exec 3<"${TMP_BASE}"; exec 4<"${TMP_NEW}"
while read -u 3 base_case base_q base_k base_algo base_dtype base_e2e base_algo_ms && \
      read -u 4 new_case new_q new_k new_algo new_dtype new_e2e new_algo_ms; do
    speedup_e2e="" speedup_algo=""
    if [[ "${base_e2e}" == "ERROR" || "${new_e2e}" == "ERROR" ]]; then
        speedup_e2e="N/A"
    else
        if awk "BEGIN {exit !(${new_e2e} <= 0.000)}"; then speedup_e2e="INFx"
        else speedup_e2e="$(awk -v base="${base_e2e}" -v new="${new_e2e}" 'BEGIN { printf "%.2fx", base/new }')"; fi
    fi

    if [[ "${base_algo_ms}" == "ERROR" || "${new_algo_ms}" == "ERROR" ]]; then
        speedup_algo="N/A"
    else
        if awk "BEGIN {exit !(${new_algo_ms} <= 0.000)}"; then speedup_algo="INFx"
        else speedup_algo="$(awk -v base="${base_algo_ms}" -v new="${new_algo_ms}" 'BEGIN { printf "%.2fx", base/new }')"; fi
    fi

    printf "| %-10s | %-12s | %-6s | %-12s | %-14s | %-15s | %-12s | %-13s | %-12s | %-13s |\n" \
        "${base_case}" "q=${base_q}/k=${base_k}" "${base_dtype}" "${base_algo}" "${base_e2e}" "${base_algo_ms}" "${new_e2e}" "${new_algo_ms}" "${speedup_e2e}" "${speedup_algo}"
done
exec 3<&-; exec 4<&-
