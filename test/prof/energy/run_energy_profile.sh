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

MEASURE_DIR="${ROOT_DIR}/test/prof/results/measurements"
PLOTS_DIR="${ROOT_DIR}/test/prof/results/plots"
MEAS_CSV="${ROOT_DIR}/test/prof/results/profile_measurements.csv"
CASE_PATH="${ROOT_DIR}/test/prof/cases/bitonic/int/q10_k8_max.case"
TOPK_BIN="${ROOT_DIR}/build/CPU/topk"
SLEEP_SECONDS="1"
GPU_INDEX="0"
GPU_INTERVAL_MS="100"
RAPL_PATH=""
SKIP_GPU="no"
SKIP_RAPL="no"
SKIP_PLOTS="no"

usage() {
    cat <<'EOF'
Usage:
    test/prof/energy/run_energy_profile.sh [options]

Options:
  --case <path>              Testcase path for topk run
                             (default: test/prof/cases/bitonic/int/q10_k8_max.case)
  --topk-bin <path>          Top-k binary to execute for workload run
                             (default: build/CPU/topk)
  --sleep-seconds <sec>      Sleep duration for baseline runs (default: 1)
  --gpu-index <idx>          GPU index for measure_smi.sh (default: 0)
  --gpu-interval-ms <ms>     Sampling interval in ms for GPU script (default: 100)
  --rapl-path <path>         Optional explicit RAPL energy_uj path
  --skip-gpu                 Skip GPU measurements
  --skip-rapl                Skip RAPL measurement
  --skip-plots               Skip CSV/plot refresh step
  --help, -h                 Show this help

Outputs:
  test/prof/results/measurements/*.txt
  test/prof/results/profile_measurements.csv
  test/prof/results/plots/energy_by_source.png
  test/prof/results/plots/power_by_source.png

Notes:
  - RAPL collection is executed with sudo.
  - The RAPL report is piped through tee so output files remain user-owned.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            CASE_PATH="$2"
            shift 2
            ;;
        --topk-bin)
            TOPK_BIN="$2"
            shift 2
            ;;
        --sleep-seconds)
            SLEEP_SECONDS="$2"
            shift 2
            ;;
        --gpu-index)
            GPU_INDEX="$2"
            shift 2
            ;;
        --gpu-interval-ms)
            GPU_INTERVAL_MS="$2"
            shift 2
            ;;
        --rapl-path)
            RAPL_PATH="$2"
            shift 2
            ;;
        --skip-gpu)
            SKIP_GPU="yes"
            shift
            ;;
        --skip-rapl)
            SKIP_RAPL="yes"
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS="yes"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "error: --gpu-index must be a non-negative integer" >&2
    exit 2
fi

if [[ ! "${GPU_INTERVAL_MS}" =~ ^[0-9]+$ ]] || [[ "${GPU_INTERVAL_MS}" -le 0 ]]; then
    echo "error: --gpu-interval-ms must be a positive integer" >&2
    exit 2
fi

if [[ ! "${SLEEP_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: --sleep-seconds must be numeric" >&2
    exit 2
fi

if [[ "${SKIP_GPU}" == "no" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "error: nvidia-smi not found but GPU measurement is enabled" >&2
        echo "hint: use --skip-gpu to bypass GPU measurement" >&2
        exit 1
    }
fi

MEASURE_SMI_SH="${ROOT_DIR}/test/prof/energy/measure_smi.sh"
MEASURE_RAPL_SH="${ROOT_DIR}/test/prof/energy/measure_rapl.sh"

if [[ ! -x "${MEASURE_SMI_SH}" ]]; then
    echo "error: missing executable ${MEASURE_SMI_SH}" >&2
    exit 1
fi

if [[ ! -x "${MEASURE_RAPL_SH}" ]]; then
    echo "error: missing executable ${MEASURE_RAPL_SH}" >&2
    exit 1
fi

if [[ ! -x "${TOPK_BIN}" ]]; then
    echo "error: topk binary is not executable: ${TOPK_BIN}" >&2
    exit 1
fi

if [[ ! -f "${CASE_PATH}" ]]; then
    echo "error: testcase not found: ${CASE_PATH}" >&2
    exit 1
fi

mkdir -p "${MEASURE_DIR}" "${PLOTS_DIR}"

run_stamp="$(date +%Y%m%d_%H%M%S)"

gpu_sleep_out="${MEASURE_DIR}/gpu_sleep_${run_stamp}.txt"
gpu_topk_out="${MEASURE_DIR}/gpu_topk_${run_stamp}.txt"
rapl_sleep_out="${MEASURE_DIR}/rapl_sleep_${run_stamp}.txt"

if [[ "${SKIP_GPU}" == "no" ]]; then
    echo "[1/3] GPU baseline measurement (sleep ${SLEEP_SECONDS}s)"
    "${MEASURE_SMI_SH}" \
        --gpu-index "${GPU_INDEX}" \
        --interval-ms "${GPU_INTERVAL_MS}" \
        --out "${gpu_sleep_out}" \
        -- sleep "${SLEEP_SECONDS}"

    echo "[2/3] GPU workload measurement (${TOPK_BIN} ${CASE_PATH})"
    "${MEASURE_SMI_SH}" \
        --gpu-index "${GPU_INDEX}" \
        --interval-ms "${GPU_INTERVAL_MS}" \
        --out "${gpu_topk_out}" \
        -- "${TOPK_BIN}" "${CASE_PATH}"
fi

if [[ "${SKIP_RAPL}" == "no" ]]; then
    echo "[3/3] RAPL baseline measurement with sudo (sleep ${SLEEP_SECONDS}s)"
    rapl_args=("${MEASURE_RAPL_SH}")
    if [[ -n "${RAPL_PATH}" ]]; then
        rapl_args+=(--path "${RAPL_PATH}")
    fi
    rapl_args+=(-- sleep "${SLEEP_SECONDS}")

    # Keep output file owned by the invoking user.
    sudo "${rapl_args[@]}" | tee "${rapl_sleep_out}"
fi

if [[ "${SKIP_PLOTS}" == "no" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        if command -v python3 >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v python3)"
        else
            echo "error: python3 not found" >&2
            exit 1
        fi
    fi

    echo "Refreshing measurement CSV and energy plots"
    "${PYTHON_BIN}" "${ROOT_DIR}/test/prof/profiler/profiler.py" \
        --plot energy-by-source power-by-source \
        --measurement-csv-out "${MEAS_CSV}" \
        --measurement-glob "${MEASURE_DIR}/*.txt" \
        --output-dir "${PLOTS_DIR}"
fi

echo
echo "Run complete."
echo "Measurements dir: ${MEASURE_DIR}"
echo "Measurement CSV: ${MEAS_CSV}"
echo "Plots dir: ${PLOTS_DIR}"
