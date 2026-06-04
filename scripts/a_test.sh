#!/usr/bin/env bash
set -uo pipefail

# Configuration
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: $(basename "$0") [cpu|gpu|npu] [bitonic|map_reduce|both]"
    echo "  Reports the performance of the current codebase for the specified backend and algorithm."
    echo "  Defaults: backend=cpu, algorithm=both."
}

BACKEND_INPUT="${1:-cpu}"
BACKEND_LC="$(echo "${BACKEND_INPUT}" | tr '[:upper:]' '[:lower:]')"
case "${BACKEND_LC}" in
    cpu) BACKEND_DIR="CPU"; BACKEND_TARGET="cpu_topk" ;;
    gpu) BACKEND_DIR="GPU"; BACKEND_TARGET="gpu_topk" ;;
    npu) BACKEND_DIR="NPU"; BACKEND_TARGET="npu_topk" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: Unknown backend '${BACKEND_INPUT}'." >&2; usage; exit 2 ;;
esac

if [[ "${BACKEND_LC}" == "npu" && -z "${XILINX_XRT:-}" ]]; then
    if [[ -f "/opt/xilinx/xrt/setup.sh" ]]; then
        source "/opt/xilinx/xrt/setup.sh"
    else
        echo "Warning: XILINX_XRT is not set and /opt/xilinx/xrt/setup.sh not found. Execution may fail." >&2
    fi
fi

ALGO_INPUT="${2:-both}"
ALGO_LC="$(echo "${ALGO_INPUT}" | tr '[:upper:]' '[:lower:]')"
case "${ALGO_LC}" in
    bitonic|map_reduce|both) TARGET_ALGO="${ALGO_LC}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: Unknown algorithm '${ALGO_INPUT}'." >&2; usage; exit 2 ;;
esac

ROOT_BUILD_DIR="${ROOT_DIR}/build"
EXECUTABLE="${ROOT_BUILD_DIR}/${BACKEND_DIR}/topk"
TIMEOUT_SECONDS="${TOPK_TEST_TIMEOUT_SECONDS:-300}"

cases=(
    "Small  10 16   bitonic float"
    "Small  10 16   bitonic fp16"
    "Small  10 16   bitonic double"
    "Medium 20 256  bitonic float"
    "Medium 20 256  bitonic fp16"
    "Medium 20 256  bitonic double"
    "Large  23 1024 bitonic float"
    "Large  23 1024 bitonic fp16"
    "Large  23 1024 bitonic double"
    "Small  10 16   map_reduce float"
    "Small  10 16   map_reduce fp16"
    "Small  10 16   map_reduce double"
    "Medium 20 256  map_reduce float"
    "Medium 20 256  map_reduce fp16"
    "Medium 20 256  map_reduce double"
    "Large  23 1024 map_reduce float"
    "Large  23 1024 map_reduce fp16"
    "Large  23 1024 map_reduce double"
)

# Checks
cd "${ROOT_DIR}" || { echo "Error: Cannot cd to ${ROOT_DIR}" >&2; exit 1; }

# Run Tests
configure_root_build() {
    local cpu_opt="$1"
    local gpu_opt="$2"
    local npu_opt="$3"

    echo ">> Configuring root build (CPU=${cpu_opt} GPU=${gpu_opt} NPU=${npu_opt})..."
    if ! cmake -S "${ROOT_DIR}" -B "${ROOT_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
        -DTOPK_BUILD_CPU="${cpu_opt}" -DTOPK_BUILD_GPU="${gpu_opt}" -DTOPK_BUILD_NPU="${npu_opt}" \
        >/dev/null 2>&1; then
        echo "Error: CMake configure failed!" >&2
        return 1
    fi
}

run_suite() {
    local OUT_FILE="$1"
    local cpu_opt="OFF"
    local gpu_opt="OFF"
    local npu_opt="OFF"

    case "${BACKEND_LC}" in
        cpu) cpu_opt="ON" ;;
        gpu) gpu_opt="ON" ;;
        npu) npu_opt="ON" ;;
    esac

    echo ">> Building ${BACKEND_DIR}..."
    if ! configure_root_build "${cpu_opt}" "${gpu_opt}" "${npu_opt}"; then return 1; fi
    if ! cmake --build "${ROOT_BUILD_DIR}" --config Release --target "${BACKEND_TARGET}" >/dev/null 2>&1; then
        echo "Error: Build failed!" >&2
        return 1
    fi
    if [[ ! -x "${EXECUTABLE}" ]]; then echo "Error: Executable not found: ${EXECUTABLE}" >&2; return 1; fi

    > "${OUT_FILE}"

    for case_info in "${cases[@]}"; do
        read -r size_name q k algo dtype <<< "${case_info}"

        [[ "${TARGET_ALGO}" != "both" && "${algo}" != "${TARGET_ALGO}" ]] && continue

        if [[ "${BACKEND_LC}" == "npu" ]]; then
            local xclbin_path="${ROOT_BUILD_DIR}/${BACKEND_DIR}/${algo}.xclbin"
            if [[ ! -f "${xclbin_path}" ]]; then
                echo "   [!] Skipping: Missing xclbin for ${algo} at ${xclbin_path}"
                continue
            fi
            export NPU_OFFLOAD_XCLBIN="${xclbin_path}"
        fi

        local cmd=("${EXECUTABLE}" "q=${q}" "k=${k}" "algo=${algo}" "dtype=${dtype}")
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
            RAW_OUTPUT=$(timeout --preserve-status "${TIMEOUT_SECONDS}"s "${cmd[@]}" 2>>"${ERR_LOG}") || exit_code=$?

            if [[ ${exit_code} -ne 0 ]]; then
                found_e2e=0
                found_algo=0
                min_e2e=""
                min_algo=""
                if [[ ${exit_code} -eq 124 ]]; then
                    echo "   [!] ${algo} (${dtype}) timed out after ${TIMEOUT_SECONDS}s for q=${q}. See log: ${ERR_LOG}"
                else
                    echo "   [!] ${algo} (${dtype}) failed for q=${q}. See log: ${ERR_LOG}"
                fi
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

        echo "${size_name} ${q} ${k} ${algo} ${dtype} ${duration_e2e_out} ${duration_algo_out}" >> "${OUT_FILE}"
    done
}

TMP_OUT="$(mktemp)"

cleanup() {
    rm -f "${TMP_OUT}"
    echo -e "\n>> Restoring root build to all backends..."
    configure_root_build "ON" "ON" "ON" >/dev/null 2>&1 || echo "Warning: Failed to restore root build settings." >&2
}
trap cleanup EXIT

echo -e "Backend: ${BACKEND_DIR} | Target: ${TARGET_ALGO}\nExecutable: ${EXECUTABLE}"

if ! run_suite "${TMP_OUT}"; then exit 1; fi

# Output
echo -e "\n=== Performance Report ==="
printf "| %-15s | %-12s | %-15s | %-6s | %-14s | %-15s |\n" \
    "Case" "Size (q/k)" "Algorithm" "Type" "E2E (ms)" "Algo (ms)"
printf "|-----------------|--------------|-----------------|--------|----------------|-----------------|\n"

while read -r case_name q k algo dtype e2e algo_ms; do
    printf "| %-15s | %-12s | %-15s | %-6s | %-14s | %-15s |\n" \
        "${case_name}" "q=${q}/k=${k}" "${algo}" "${dtype}" "${e2e}" "${algo_ms}"
done < "${TMP_OUT}"
