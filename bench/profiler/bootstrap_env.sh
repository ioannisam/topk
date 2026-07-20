#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
while [[ "${ROOT_DIR}" != "/" && ! -f "${ROOT_DIR}/CMakeLists.txt" ]]; do
    ROOT_DIR="$(dirname "${ROOT_DIR}")"
done
if [[ ! -f "${ROOT_DIR}/CMakeLists.txt" ]]; then
    echo "error: could not locate repo root (CMakeLists.txt)" >&2
    exit 1
fi

VENV_DIR="${ROOT_DIR}/bench/profiler/.venv"
REQ_FILE="${ROOT_DIR}/bench/profiler/requirements.txt"
PROFILER="${ROOT_DIR}/bench/profiler/profiler.py"
TEST_INPUT="${ROOT_DIR}/bench/results/raw/cases/output.txt"
OUT_DIR="${ROOT_DIR}/bench/results/plots"
CSV_OUT="${ROOT_DIR}/bench/results/derived/cases.csv"
MEAS_CSV_OUT="${ROOT_DIR}/bench/results/derived/energy.csv"

pick_python() {
    if [[ -x "${ROOT_DIR}/NPU/npu_env/bin/python" ]]; then
        echo "${ROOT_DIR}/NPU/npu_env/bin/python"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    return 1
}

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "Missing requirements file: ${REQ_FILE}" >&2
    exit 1
fi

BASE_PYTHON="$(pick_python || true)"
if [[ -z "${BASE_PYTHON}" ]]; then
    echo "Could not find a usable Python interpreter." >&2
    exit 1
fi

echo "Using base Python: ${BASE_PYTHON}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating profiler venv: ${VENV_DIR}"
    "${BASE_PYTHON}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${REQ_FILE}"

echo
echo "Profiler environment is ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"

if [[ "${1:-}" == "--run-plots" ]]; then
    echo
    echo "Running profiler to generate plots..."
    "${VENV_DIR}/bin/python" "${PROFILER}" \
        --input "${TEST_INPUT}" \
        --plot all \
        --timing-csv-out "${CSV_OUT}" \
        --energy-csv-out "${MEAS_CSV_OUT}" \
        --output-dir "${OUT_DIR}"
fi
