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

usage() {
    echo "Usage: test/prof/cases/run_testcases.sh [options] <backends> [types]"
    echo "Examples:"
    echo "  test/prof/cases/run_testcases.sh cpu"
    echo "  test/prof/cases/run_testcases.sh cpu int --output /tmp/out.txt"
    echo "  test/prof/cases/run_testcases.sh gt"
    echo "  test/prof/cases/run_testcases.sh cpu float,int,uint"
    echo "  test/prof/cases/run_testcases.sh '{cpu,gt,npu}' '{float,int,uint}'"
    echo "  test/prof/cases/run_testcases.sh --energy auto '{cpu,gpu,gt}' int"
    echo "Options:"
    echo "  --energy <none|auto|rapl|gpu>   Enable optional energy measurement wrappers (default: none)"
    echo "  --energy-out-dir <path>         Directory for measurement output files"
    echo "                                  (default: test/prof/results/measurements)"
    echo "  --rapl-path <path>              Optional explicit RAPL energy_uj path for rapl mode"
    echo "  --gpu-index <idx>               GPU index for gpu mode (default: 0)"
    echo "  --gpu-interval-ms <ms>          Sample interval for gpu mode (default: 100)"
    echo "  --q-min <int>                   Minimum q to profile (default: 1)"
    echo "  --q-max <int>                   Maximum q to profile (default: 21)"
    echo "  --k <int>                       Requested top-k (default: 8)"
    echo "  --mode <min|max>                Top-k mode (default: max)"
    echo "  --run <trunc|full|both>         Network run mode (default: trunc)"
    echo "  --threads <int>                 Threads argument for binaries (default: 8)"
    echo "  --seed-base <int>               Per-q seed base; seed=seed-base+q (default: 100)"
    echo "  --min <int>                     Random generator minimum (default: 0)"
    echo "  --max <int>                     Random generator maximum (default: 1000)"
    echo "  --verify <true|false>           Enable CPU sorted reference check (default: true)"
    echo "  --output <file>                 Detailed output file (default: test/prof/results/test_output.txt)"
    echo "  --help, -h                      Show this help"
}

normalize_list() {
    local raw="$1"
    local cleaned
    cleaned="$(printf '%s' "${raw}" | tr -d '{}' | tr -d '[:space:]')"
    printf '%s' "${cleaned}"
}

resolve_binary_path() {
    local backend="$1"
    case "${backend}" in
        gt)
            printf '%s' "${ROOT_DIR}/build/test/ground_truth/topk"
            ;;
        cpu)
            printf '%s' "${ROOT_DIR}/build/CPU/topk"
            ;;
        gpu)
            printf '%s' "${ROOT_DIR}/build/GPU/topk"
            ;;
        npu)
            printf '%s' "${ROOT_DIR}/build/NPU/topk"
            ;;
        *)
            return 1
            ;;
    esac
}

is_backend_token() {
    local token
    token="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    [[ "${token}" == "cpu" || "${token}" == "gt" || "${token}" == "gpu" || "${token}" == "npu" ]]
}

resolve_energy_mode() {
    local backend="$1"
    local mode="$2"

    case "${mode}" in
        none)
            printf '%s' "none"
            ;;
        rapl)
            printf '%s' "rapl"
            ;;
        gpu)
            printf '%s' "gpu"
            ;;
        auto)
            if [[ "${backend}" == "gpu" ]]; then
                printf '%s' "gpu"
            else
                printf '%s' "rapl"
            fi
            ;;
        *)
            return 1
            ;;
    esac
}

ENERGY_MODE="none"
ENERGY_OUT_DIR="${ROOT_DIR}/test/prof/results/measurements"
RAPL_PATH=""
GPU_INDEX="0"
GPU_INTERVAL_MS="100"
RESULT_FILE="${ROOT_DIR}/test/prof/results/test_output.txt"

Q_MIN=1
Q_MAX=21
DEFAULT_K=8
DEFAULT_MODE="max"
DEFAULT_RUN="trunc"
DEFAULT_DEBUG="false"
DEFAULT_THREADS="8"
SEED_BASE=100
DEFAULT_MIN=0
DEFAULT_MAX=1000
VERIFY_OUTPUT="true"

POSITIONALS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --energy|--energy-out-dir|--rapl-path|--gpu-index|--gpu-interval-ms|--output|--q-min|--q-max|--k|--mode|--run|--threads|--seed-base|--min|--max|--verify)
            if [[ $# -lt 2 ]]; then
                echo "error: $1 needs a value"
                usage
                exit 2
            fi
            case "$1" in
                --energy)
                    ENERGY_MODE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
                    ;;
                --energy-out-dir)
                    ENERGY_OUT_DIR="$2"
                    ;;
                --rapl-path)
                    RAPL_PATH="$2"
                    ;;
                --gpu-index)
                    GPU_INDEX="$2"
                    ;;
                --gpu-interval-ms)
                    GPU_INTERVAL_MS="$2"
                    ;;
                --output)
                    RESULT_FILE="$2"
                    ;;
                --q-min)
                    Q_MIN="$2"
                    ;;
                --q-max)
                    Q_MAX="$2"
                    ;;
                --k)
                    DEFAULT_K="$2"
                    ;;
                --mode)
                    DEFAULT_MODE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
                    ;;
                --run)
                    DEFAULT_RUN="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
                    ;;
                --threads)
                    DEFAULT_THREADS="$2"
                    ;;
                --seed-base)
                    SEED_BASE="$2"
                    ;;
                --min)
                    DEFAULT_MIN="$2"
                    ;;
                --max)
                    DEFAULT_MAX="$2"
                    ;;
                --verify)
                    VERIFY_OUTPUT="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
                    ;;
            esac
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                POSITIONALS+=("$1")
                shift
            done
            ;;
        *)
            POSITIONALS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONALS[@]}"

if [[ "${ENERGY_MODE}" != "none" && "${ENERGY_MODE}" != "auto" && "${ENERGY_MODE}" != "rapl" && "${ENERGY_MODE}" != "gpu" ]]; then
    echo "error: invalid --energy value '${ENERGY_MODE}' (expected: none|auto|rapl|gpu)"
    usage
    exit 2
fi

if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "error: --gpu-index must be a non-negative integer"
    exit 2
fi

if [[ ! "${GPU_INTERVAL_MS}" =~ ^[0-9]+$ ]] || [[ "${GPU_INTERVAL_MS}" -le 0 ]]; then
    echo "error: --gpu-interval-ms must be a positive integer"
    exit 2
fi

if [[ ! "${Q_MIN}" =~ ^[0-9]+$ ]] || [[ ! "${Q_MAX}" =~ ^[0-9]+$ ]]; then
    echo "error: --q-min/--q-max must be non-negative integers"
    exit 2
fi
if [[ "${Q_MIN}" -gt "${Q_MAX}" ]]; then
    echo "error: --q-min must be <= --q-max"
    exit 2
fi
if [[ ! "${DEFAULT_K}" =~ ^[0-9]+$ ]] || [[ "${DEFAULT_K}" -le 0 ]]; then
    echo "error: --k must be a positive integer"
    exit 2
fi
if [[ "${DEFAULT_MODE}" != "max" && "${DEFAULT_MODE}" != "min" ]]; then
    echo "error: --mode must be min or max"
    exit 2
fi
if [[ "${DEFAULT_RUN}" != "trunc" && "${DEFAULT_RUN}" != "full" && "${DEFAULT_RUN}" != "both" ]]; then
    echo "error: --run must be trunc, full, or both"
    exit 2
fi
if [[ ! "${DEFAULT_THREADS}" =~ ^[0-9]+$ ]] || [[ "${DEFAULT_THREADS}" -le 0 ]]; then
    echo "error: --threads must be a positive integer"
    exit 2
fi
if [[ ! "${SEED_BASE}" =~ ^[0-9]+$ ]]; then
    echo "error: --seed-base must be a non-negative integer"
    exit 2
fi
if [[ ! "${DEFAULT_MIN}" =~ ^-?[0-9]+$ ]] || [[ ! "${DEFAULT_MAX}" =~ ^-?[0-9]+$ ]]; then
    echo "error: --min/--max must be integers"
    exit 2
fi
if [[ "${DEFAULT_MAX}" -lt "${DEFAULT_MIN}" ]]; then
    echo "error: --max must be >= --min"
    exit 2
fi
if [[ "${VERIFY_OUTPUT}" != "true" && "${VERIFY_OUTPUT}" != "false" ]]; then
    echo "error: --verify must be true or false"
    exit 2
fi

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

EXPECTED_PASS_MARKER="Top-k correctness vs CPU sorted reference: OK"
RESULTS_DIR="$(dirname "${RESULT_FILE}")"

ARGS=("$@")
ARGC=${#ARGS[@]}
IDX=0

BACKEND_TOKENS=()
TYPE_TOKENS=()

if is_backend_token "${ARGS[0]}"; then
    while [[ ${IDX} -lt ${ARGC} ]] && is_backend_token "${ARGS[${IDX}]}"; do
        BACKEND_TOKENS+=("$(printf '%s' "${ARGS[${IDX}]}" | tr '[:upper:]' '[:lower:]')")
        IDX=$((IDX + 1))
    done

    while [[ ${IDX} -lt ${ARGC} ]]; do
        TYPE_TOKENS+=("${ARGS[${IDX}]}")
        IDX=$((IDX + 1))
    done

    BACKENDS_RAW="$(IFS=,; printf '%s' "${BACKEND_TOKENS[*]}")"
    TYPES_RAW=""
    if [[ ${#TYPE_TOKENS[@]} -gt 0 ]]; then
        TYPE_EXPR="$(IFS=,; printf '%s' "${TYPE_TOKENS[*]}")"
        TYPES_RAW="$(normalize_list "${TYPE_EXPR}")"
    fi
else
    BACKEND_EXPR="${ARGS[0]}"
    if [[ "${BACKEND_EXPR}" == \{* ]]; then
        IDX=1
        while [[ "${BACKEND_EXPR}" != *\} && ${IDX} -lt ${ARGC} ]]; do
            BACKEND_EXPR+=" ${ARGS[${IDX}]}"
            IDX=$((IDX + 1))
        done
        if [[ "${BACKEND_EXPR}" != *\} ]]; then
            echo "Malformed backend list: missing closing }"
            usage
            exit 2
        fi
    else
        IDX=1
    fi

    TYPE_EXPR=""
    if [[ ${IDX} -lt ${ARGC} ]]; then
        TYPE_EXPR="${ARGS[${IDX}]}"
        IDX=$((IDX + 1))
        while [[ ${IDX} -lt ${ARGC} ]]; do
            TYPE_EXPR+=" ${ARGS[${IDX}]}"
            IDX=$((IDX + 1))
        done
    fi

    BACKENDS_RAW="$(normalize_list "${BACKEND_EXPR}")"
    TYPES_RAW=""
    if [[ -n "${TYPE_EXPR}" ]]; then
        TYPES_RAW="$(normalize_list "${TYPE_EXPR}")"
    fi
fi

IFS=',' read -r -a BACKENDS <<< "${BACKENDS_RAW}"
if [[ ${#BACKENDS[@]} -eq 0 || -z "${BACKENDS[0]}" ]]; then
    echo "No backends provided."
    usage
    exit 2
fi

if [[ -n "${TYPES_RAW}" ]]; then
    IFS=',' read -r -a REQUESTED_TYPES <<< "${TYPES_RAW}"
else
    REQUESTED_TYPES=(double float fp16 int uint)
fi

TOTAL_PASS=0
TOTAL_FAIL=0
RUN_ID="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULTS_DIR}"
if [[ "${ENERGY_MODE}" != "none" ]]; then
    mkdir -p "${ENERGY_OUT_DIR}"
fi
{
    echo "Testcase Run Output"
    echo "Backends: ${BACKENDS_RAW}"
    if [[ -n "${TYPES_RAW}" ]]; then
        echo "Types   : ${TYPES_RAW}"
    else
        echo "Types   : default (double,float,fp16,int,uint)"
    fi
    echo "Q range : ${Q_MIN}..${Q_MAX}"
    echo "Defaults: k=min(${DEFAULT_K},N) mode=${DEFAULT_MODE} run=${DEFAULT_RUN} threads=${DEFAULT_THREADS} verify=${VERIFY_OUTPUT}"
    echo "Random range: ${DEFAULT_MIN}..${DEFAULT_MAX}"
    echo "Energy  : ${ENERGY_MODE}"
    if [[ "${ENERGY_MODE}" != "none" ]]; then
        echo "Energy out dir: ${ENERGY_OUT_DIR}"
    fi
    echo
} >"${RESULT_FILE}"

echo "Running testcase suite"
echo "Backends: ${BACKENDS_RAW}"
if [[ -n "${TYPES_RAW}" ]]; then
    echo "Types   : ${TYPES_RAW}"
else
    echo "Types   : default (double,float,fp16,int,uint)"
fi
echo "Q range : ${Q_MIN}..${Q_MAX}"
echo "Defaults: k=min(${DEFAULT_K},N) mode=${DEFAULT_MODE} run=${DEFAULT_RUN} threads=${DEFAULT_THREADS} verify=${VERIFY_OUTPUT}"
echo "Random range: ${DEFAULT_MIN}..${DEFAULT_MAX}"
echo "Energy  : ${ENERGY_MODE}"
if [[ "${ENERGY_MODE}" != "none" ]]; then
    echo "Energy out dir: ${ENERGY_OUT_DIR}"
fi

for backend_raw in "${BACKENDS[@]}"; do
    BACKEND="$(printf '%s' "${backend_raw}" | tr '[:upper:]' '[:lower:]')"
    BINARY_PATH="$(resolve_binary_path "${BACKEND}")" || {
        echo
        echo "Unsupported backend: ${backend_raw}"
        usage
        exit 2
    }

    if [[ ! -x "${BINARY_PATH}" ]]; then
        echo
        echo "Backend binary not found or not executable: ${BINARY_PATH}"
        echo "Build backend '${BACKEND}' first, then rerun this script."
        {
            echo "== Backend: ${BACKEND} =="
            echo "Binary not found: ${BINARY_PATH}"
            echo
        } >>"${RESULT_FILE}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        continue
    fi

    BACKEND_PASS=0
    BACKEND_FAIL=0

    TYPES=("${REQUESTED_TYPES[@]}")

    echo
    echo "Backend: ${BACKEND}"
    echo "Binary : ${BINARY_PATH}"
    {
        echo "== Backend: ${BACKEND} =="
        echo "Binary: ${BINARY_PATH}"
        echo
    } >>"${RESULT_FILE}"

    for type_raw in "${TYPES[@]}"; do
        TYPE="$(printf '%s' "${type_raw}" | tr '[:upper:]' '[:lower:]')"
        TYPE_PASS=0
        TYPE_FAIL=0

        echo "  Type ${TYPE}: dynamic cases"
        {
            echo "-- Type: ${TYPE} --"
            echo "Cases: dynamic"
            echo
        } >>"${RESULT_FILE}"

        ALGORITHMS=(bitonic map_reduce)
        if [[ "${BACKEND}" == "npu" || "${BACKEND}" == "gt" ]]; then
            ALGORITHMS=(bitonic)
        fi

        for ALGO in "${ALGORITHMS[@]}"; do
            for ((q=${Q_MIN}; q<=${Q_MAX}; q++)); do
                n=$((1 << q))
                k_eff=${DEFAULT_K}
                if [[ ${k_eff} -gt ${n} ]]; then
                    k_eff=${n}
                fi

                NAME="$(printf 'q%02d_k%d_%s' "${q}" "${k_eff}" "${DEFAULT_MODE}")"

            OUTPUT="$(mktemp)"
            CASE_STATUS="FAIL"
            CASE_REASON="non-zero exit"
            ENERGY_CASE_FILE=""

            CASE_ENERGY_MODE="$(resolve_energy_mode "${BACKEND}" "${ENERGY_MODE}")" || {
                echo "error: unsupported energy mode '${ENERGY_MODE}'"
                exit 2
            }

            if [[ "${CASE_ENERGY_MODE}" == "rapl" || "${CASE_ENERGY_MODE}" == "gpu" ]]; then
                ENERGY_CASE_FILE="${ENERGY_OUT_DIR}/${RUN_ID}_${BACKEND}_${TYPE}_${ALGO}_${NAME}_${CASE_ENERGY_MODE}.txt"
            fi

            SEED=$((SEED_BASE + q))
            CASE_ARGS=(
                "q=${q}"
                "k=${k_eff}"
                "mode=${DEFAULT_MODE}"
                "dtype=${TYPE}"
                "algo=${ALGO}"
                "run=${DEFAULT_RUN}"
                "debug=${DEFAULT_DEBUG}"
                "threads=${DEFAULT_THREADS}"
                "seed=${SEED}"
                "verify=${VERIFY_OUTPUT}"
                "min=${DEFAULT_MIN}"
                "max=${DEFAULT_MAX}"
            )

            if [[ "${CASE_ENERGY_MODE}" == "none" ]]; then
                RUN_CMD=("${BINARY_PATH}" "${CASE_ARGS[@]}")
            elif [[ "${CASE_ENERGY_MODE}" == "rapl" ]]; then
                RUN_CMD=("${ROOT_DIR}/test/prof/energy/measure_rapl.sh" "--out" "${ENERGY_CASE_FILE}")
                if [[ -n "${RAPL_PATH}" ]]; then
                    RUN_CMD+=("--path" "${RAPL_PATH}")
                fi
                RUN_CMD+=("--" "${BINARY_PATH}" "${CASE_ARGS[@]}")
            else
                RUN_CMD=("${ROOT_DIR}/test/prof/energy/measure_smi.sh" "--out" "${ENERGY_CASE_FILE}" "--gpu-index" "${GPU_INDEX}" "--interval-ms" "${GPU_INTERVAL_MS}" "--" "${BINARY_PATH}" "${CASE_ARGS[@]}")
            fi

            if "${RUN_CMD[@]}" >"${OUTPUT}" 2>&1; then
                if [[ "${VERIFY_OUTPUT}" == "true" ]] && ! grep -Fq "${EXPECTED_PASS_MARKER}" "${OUTPUT}"; then
                    echo "    [FAIL] ${NAME} (PASS marker missing)"
                    tail -n 20 "${OUTPUT}" || true
                    TYPE_FAIL=$((TYPE_FAIL + 1))
                    CASE_REASON="PASS marker missing"
                else
                    echo "    [PASS] ${NAME}"
                    TYPE_PASS=$((TYPE_PASS + 1))
                    CASE_STATUS="PASS"
                    CASE_REASON="ok"
                fi
            else
                echo "    [FAIL] ${NAME} (non-zero exit)"
                tail -n 20 "${OUTPUT}" || true
                TYPE_FAIL=$((TYPE_FAIL + 1))
            fi

            {
                echo "### Case: ${NAME}"
                echo "Status: ${CASE_STATUS}"
                echo "Reason: ${CASE_REASON}"
                echo "Command: ${RUN_CMD[*]}"
                if [[ -n "${ENERGY_CASE_FILE}" ]]; then
                    echo "Measurement file: ${ENERGY_CASE_FILE}"
                fi
                echo "Output:"
                cat "${OUTPUT}"
                echo
            } >>"${RESULT_FILE}"

            rm -f "${OUTPUT}"
            done
        done

        printf "    Type %s summary: pass=%d fail=%d\n" "${TYPE}" "${TYPE_PASS}" "${TYPE_FAIL}"
        BACKEND_PASS=$((BACKEND_PASS + TYPE_PASS))
        BACKEND_FAIL=$((BACKEND_FAIL + TYPE_FAIL))
    done

    printf "  Backend %s summary: pass=%d fail=%d\n" "${BACKEND}" "${BACKEND_PASS}" "${BACKEND_FAIL}"
    TOTAL_PASS=$((TOTAL_PASS + BACKEND_PASS))
    TOTAL_FAIL=$((TOTAL_FAIL + BACKEND_FAIL))
done

echo
echo "Total Summary"
printf "  Passed: %d\n" "${TOTAL_PASS}"
printf "  Failed: %d\n" "${TOTAL_FAIL}"

if [[ ${TOTAL_FAIL} -gt 0 ]]; then
    exit 1
fi

if [[ ${TOTAL_PASS} -eq 0 ]]; then
    echo "No testcases were executed."
    exit 4
fi

echo "Wrote detailed output to: ${RESULT_FILE}"

exit 0
