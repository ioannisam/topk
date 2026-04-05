#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GT_STATUS="x"
CPU_STATUS="x"
GPU_STATUS="x"
NPU_STATUS="x"

GT_REASON="Not run"
CPU_REASON="Not run"
GPU_REASON="Not run"
NPU_REASON="Not run"

run_build() {
    local BACKEND="$1"
    local SOURCE_DIR="$2"
    local BUILD_DIR="$3"

    local STATUS="x"
    local REASON=""

    local LOG_FILE
    LOG_FILE="$(mktemp)"

    if [[ ! -d "${SOURCE_DIR}" ]]; then
        REASON="source directory missing"
    elif [[ ! -f "${SOURCE_DIR}/CMakeLists.txt" ]]; then
        REASON="CMakeLists.txt not found"
    elif ! cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" >"${LOG_FILE}" 2>&1; then
        REASON="configure failed"
    elif ! cmake --build "${BUILD_DIR}" >>"${LOG_FILE}" 2>&1; then
        REASON="build failed"
    else
        STATUS="✓"
        REASON="ok"
    fi

    if [[ "${STATUS}" == "x" ]]; then
        echo
        echo "[${BACKEND}] ${REASON}"
        echo "Last 20 log lines:"
        tail -n 20 "${LOG_FILE}" || true
    fi

    rm -f "${LOG_FILE}"

    case "${BACKEND}" in
        GT)
            GT_STATUS="${STATUS}"
            GT_REASON="${REASON}"
            ;;
        CPU)
            CPU_STATUS="${STATUS}"
            CPU_REASON="${REASON}"
            ;;
        GPU)
            GPU_STATUS="${STATUS}"
            GPU_REASON="${REASON}"
            ;;
        NPU)
            NPU_STATUS="${STATUS}"
            NPU_REASON="${REASON}"
            ;;
    esac
}

echo "Building backends from ${ROOT_DIR}"

run_build "GT" "${ROOT_DIR}/test/perf/ground_truth" "${ROOT_DIR}/test/perf/ground_truth/build"
run_build "CPU" "${ROOT_DIR}/CPU" "${ROOT_DIR}/CPU/build"
run_build "NPU" "${ROOT_DIR}/NPU" "${ROOT_DIR}/NPU/build"
run_build "GPU" "${ROOT_DIR}/GPU" "${ROOT_DIR}/GPU/build"

echo
echo "Build Summary:"
printf "|  %s  |  %s  |  %s  |  %s  |\n" "GT " "CPU" "NPU" "GPU"
printf "|-%-5s-|-%-5s-|-%-5s-|-%-5s-|\n" "-----" "-----" "-----" "-----"
printf "|   %s   |   %s   |   %s   |   %s   |\n" "${GT_STATUS}" "${CPU_STATUS}" "${NPU_STATUS}" "${GPU_STATUS}"

echo
echo "Details:"
printf "%-4s %s\n" "GT:" "${GT_REASON}"
printf "%-4s %s\n" "CPU:" "${CPU_REASON}"
printf "%-4s %s\n" "NPU:" "${NPU_REASON}"
printf "%-4s %s\n" "GPU:" "${GPU_REASON}"

if [[ "${CPU_STATUS}" == "✓" && "${GT_STATUS}" == "✓" && "${NPU_STATUS}" == "✓" && "${GPU_STATUS}" == "✓" ]]; then
    exit 0
fi

exit 1
