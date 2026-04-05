#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="${ROOT_DIR}/test/smoke"

CPU_STATUS="x"
NPU_STATUS="x"
GPU_STATUS="x"
CPU_REASON="Not run"
NPU_REASON="Not run"
GPU_REASON="Not run"

run_backend() {
    local BACKEND="$1"
    local expected_line="$2"
    local STATUS="x"
    local REASON=""

    local BACKEND_DIR="${SMOKE_ROOT}/${BACKEND}"
    local BUILD_DIR="${BACKEND_DIR}/build"
    local BIN_PATH="${BUILD_DIR}/smoke"

    local LOG_FILE
    LOG_FILE="$(mktemp)"

    if ! cmake -S "${BACKEND_DIR}" -B "${BUILD_DIR}" >"${LOG_FILE}" 2>&1; then
        REASON="configure failed"
    elif ! cmake --build "${BUILD_DIR}" >>"${LOG_FILE}" 2>&1; then
        REASON="build failed"
    elif ! "${BIN_PATH}" >>"${LOG_FILE}" 2>&1; then
        REASON="run failed"
    elif ! grep -Fq "${expected_line}" "${LOG_FILE}"; then
        REASON="PASS marker not found"
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
        CPU)
            CPU_STATUS="${STATUS}"
            CPU_REASON="${REASON}"
            ;;
        NPU)
            NPU_STATUS="${STATUS}"
            NPU_REASON="${REASON}"
            ;;
        GPU)
            GPU_STATUS="${STATUS}"
            GPU_REASON="${REASON}"
            ;;
    esac
}

echo "Running smoke tests from ${SMOKE_ROOT}"

run_backend "CPU" "CPU top-k smoke test: PASS"
run_backend "NPU" "NPU top-k smoke test: PASS"
run_backend "GPU" "GPU top-k smoke test: PASS"

echo
echo "Smoke Test Summary:"
printf "|  %s  |  %s  |  %s  |\n" "CPU" "NPU" "GPU"
printf "|-%-5s-|-%-5s-|-%-5s-|\n" "-----" "-----" "-----"
printf "|   %s   |   %s   |   %s   |\n" "${CPU_STATUS}" "${NPU_STATUS}" "${GPU_STATUS}"

echo
echo "Details:"
printf "%-4s %s\n" "CPU:" "${CPU_REASON}"
printf "%-4s %s\n" "NPU:" "${NPU_REASON}"
printf "%-4s %s\n" "GPU:" "${GPU_REASON}"

if [[ "${CPU_STATUS}" == "✓" && "${NPU_STATUS}" == "✓" && "${GPU_STATUS}" == "✓" ]]; then
    exit 0
fi

exit 1
