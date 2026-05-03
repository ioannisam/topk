#!/usr/bin/env bash
set -uo pipefail

# Configuration
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: $(basename "$0") [cpu|gpu|npu] [bitonic|map_reduce|both]"
    echo "  Compares the performance of the current uncommitted changes"
    echo "  against the HEAD commit for the specified backend and algorithm."
    echo "  Defaults: backend=cpu, algorithm=both."
}

BACKEND_INPUT="${1:-cpu}"
BACKEND_LC="$(echo "${BACKEND_INPUT}" | tr '[:upper:]' '[:lower:]')"
case "${BACKEND_LC}" in
    cpu) BACKEND_DIR="CPU" ;;
    gpu) BACKEND_DIR="GPU" ;;
    npu) BACKEND_DIR="NPU" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: Unknown backend '${BACKEND_INPUT}'." >&2; usage; exit 2 ;;
esac

ALGO_INPUT="${2:-both}"
ALGO_LC="$(echo "${ALGO_INPUT}" | tr '[:upper:]' '[:lower:]')"
case "${ALGO_LC}" in
    bitonic|map_reduce|both) TARGET_ALGO="${ALGO_LC}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: Unknown algorithm '${ALGO_INPUT}'." >&2; usage; exit 2 ;;
esac

BUILD_DIR="${ROOT_DIR}/${BACKEND_DIR}/build"
EXECUTABLE="${BUILD_DIR}/topk"

cases=(
    "Small  10 16   bitonic"
    "Medium 20 256  bitonic"
    "Large  23 1024 bitonic"
    "Small  10 16   map_reduce"
    "Medium 20 256  map_reduce"
    "Large  23 1024 map_reduce"
)

# Checks
cd "${ROOT_DIR}" || { echo "Error: Cannot cd to ${ROOT_DIR}" >&2; exit 1; }
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then echo "Error: Not a git repo." >&2; exit 1; fi
if git diff --quiet && git diff --cached --quiet; then echo "Error: No uncommitted changes to test against HEAD!" >&2; exit 1; fi

# Run Tests
run_suite() {
    local PHASE_NAME="$1" OUT_FILE="$2"

    echo ">> Recompiling for ${PHASE_NAME}..."
    if [[ ! -d "${BUILD_DIR}" ]]; then echo "Error: Build dir not found: ${BUILD_DIR}" >&2; return 1; fi
    if ! cmake --build "${BUILD_DIR}" --config Release >/dev/null 2>&1; then echo "Error: Build failed!" >&2; return 1; fi
    if [[ ! -x "${EXECUTABLE}" ]]; then echo "Error: Executable not found: ${EXECUTABLE}" >&2; return 1; fi

    > "${OUT_FILE}"

    for case_info in "${cases[@]}"; do
        read -r size_name q k algo <<< "${case_info}"

        [[ "${TARGET_ALGO}" != "both" && "${algo}" != "${TARGET_ALGO}" ]] && continue

        local cmd=("${EXECUTABLE}" "q=${q}" "k=${k}" "algo=${algo}")
        [[ "${algo}" == "bitonic" ]] && cmd+=("run=both")

        local min_e2e=""
        local min_algo=""
        local found_e2e=0
        local found_algo=0
        local runs=3
        local exit_code=0
        
        for ((i=1; i<=runs; i++)); do
            local ERR_LOG; ERR_LOG="$(mktemp)"
            local RAW_OUTPUT
            RAW_OUTPUT=$("${cmd[@]}" 2>>"${ERR_LOG}") || exit_code=$?

            if [[ ${exit_code} -ne 0 ]]; then
                found_e2e=0
                found_algo=0
                min_e2e=""
                min_algo=""
                echo "   [!] ${algo} failed for q=${q}. See log: ${ERR_LOG}"
                break
            fi

            local duration_e2e duration_algo
            duration_e2e=$(echo "${RAW_OUTPUT}" | grep -E "(Trunc bitonic|Map-reduce top-k) end-to-end time \(ms\)" | awk -F':' '{print $2}' | tr -d ' ')
            duration_algo=$(echo "${RAW_OUTPUT}" | grep -E "(Trunc bitonic|Map-reduce top-k) algorithmic time \(ms\)" | awk -F':' '{print $2}' | tr -d ' ')

            if [[ -n "${duration_e2e}" && "${duration_e2e}" != "skipped" ]]; then
                if [[ ${found_e2e} -eq 0 ]]; then
                    min_e2e="${duration_e2e}"
                    found_e2e=1
                else
                    min_e2e=$(awk -v current="${duration_e2e}" -v min="${min_e2e}" 'BEGIN { print (current < min) ? current : min }')
                fi
            fi
            if [[ -n "${duration_algo}" && "${duration_algo}" != "skipped" ]]; then
                if [[ ${found_algo} -eq 0 ]]; then
                    min_algo="${duration_algo}"
                    found_algo=1
                else
                    min_algo=$(awk -v current="${duration_algo}" -v min="${min_algo}" 'BEGIN { print (current < min) ? current : min }')
                fi
            fi

            rm -f "${ERR_LOG}"
        done

        local duration_e2e_out duration_algo_out
        if [[ ${found_e2e} -eq 1 ]]; then
            duration_e2e_out="$(awk -v ms="${min_e2e}" 'BEGIN { printf "%.3f", ms }')"
        else
            duration_e2e_out="ERROR"
        fi
        if [[ ${found_algo} -eq 1 ]]; then
            duration_algo_out="$(awk -v ms="${min_algo}" 'BEGIN { printf "%.3f", ms }')"
        else
            duration_algo_out="ERROR"
        fi

        echo "${size_name} ${q} ${k} ${algo} ${duration_e2e_out} ${duration_algo_out}" >> "${OUT_FILE}"
    done
    echo ">> ${PHASE_NAME} run complete."
}

# Workflow Setup
TMP_NEW="$(mktemp)"; TMP_BASE="$(mktemp)"; STASHED=0

cleanup() {
    rm -f "${TMP_NEW}" "${TMP_BASE}"
    [[ ${STASHED} -eq 1 ]] && { echo -e "\nRestoring stashed changes..."; git stash pop -q || echo "Warning: Failed to pop git stash." >&2; }
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

# Output
echo -e "\n=== Performance Report ==="
printf "| %-15s | %-12s | %-15s | %-14s | %-15s | %-12s | %-13s | %-12s | %-13s |\n" \
    "Case" "Size (q/k)" "Algorithm" "Base E2E (ms)" "Base Algo (ms)" "New E2E (ms)" "New Algo (ms)" "Speedup E2E" "Speedup Algo"
printf "|-----------------|--------------|-----------------|----------------|-----------------|--------------|---------------|--------------|---------------|\n"

exec 3<"${TMP_BASE}"; exec 4<"${TMP_NEW}"
while read -u 3 base_case base_q base_k base_algo base_e2e base_algo_ms && read -u 4 new_case new_q new_k new_algo new_e2e new_algo_ms; do
    speedup_e2e=""
    speedup_algo=""
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

    printf "| %-15s | %-12s | %-15s | %-14s | %-15s | %-12s | %-13s | %-12s | %-13s |\n" \
        "${base_case}" "q=${base_q}/k=${base_k}" "${base_algo}" "${base_e2e}" "${base_algo_ms}" "${new_e2e}" "${new_algo_ms}" "${speedup_e2e}" "${speedup_algo}"
done
exec 3<&-; exec 4<&-
