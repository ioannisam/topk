#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: scripts/run_testcases.sh <backends> [types]"
    echo "       scripts/run_testcases.sh [options] <backends> [types]"
    echo "Examples:"
    echo "  scripts/run_testcases.sh cpu"
    echo "  scripts/run_testcases.sh gt"
    echo "  scripts/run_testcases.sh cpu float,int,uint"
    echo "  scripts/run_testcases.sh '{cpu,gt,npu}' '{float,int,uint}'"
    echo "  scripts/run_testcases.sh --energy auto '{cpu,gpu,gt}' int"
    echo "Options:"
    echo "  --cases-dir <path>              Root directory of testcase .case files"
    echo "                                  (default: test/cases)"
    echo "  --energy <none|auto|rapl|gpu>   Enable optional energy measurement wrappers (default: none)"
    echo "  --energy-out-dir <path>         Directory for measurement output files"
    echo "                                  (default: test/prof/results/measurements)"
    echo "  --rapl-path <path>              Optional explicit RAPL energy_uj path for rapl mode"
    echo "  --gpu-index <idx>               GPU index for gpu mode (default: 0)"
    echo "  --gpu-interval-ms <ms>          Sample interval for gpu mode (default: 100)"
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
            if [[ -x "${ROOT_DIR}/build/test/perf/ground_truth/topk" ]]; then
                printf '%s' "${ROOT_DIR}/build/test/perf/ground_truth/topk"
            else
                printf '%s' "${ROOT_DIR}/test/perf/ground_truth/build/topk"
            fi
            ;;
        cpu)
            printf '%s' "${ROOT_DIR}/CPU/build/topk"
            ;;
        gpu)
            printf '%s' "${ROOT_DIR}/GPU/build/topk"
            ;;
        npu)
            printf '%s' "${ROOT_DIR}/NPU/build/topk"
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

extract_case_algo() {
    local case_file="$1"
    local line
    local algo=""
    line="$(grep -m1 -E '^[[:space:]]*[^#[:space:]]' "${case_file}" || true)"
    if [[ "${line}" =~ (^|[[:space:]])algo=([^[:space:]]+) ]]; then
        algo="${BASH_REMATCH[2]}"
    fi
    if [[ -z "${algo}" ]]; then
        algo="bitonic"
    elif [[ "${algo}" == "mapreduce" ]]; then
        algo="map_reduce"
    fi
    printf '%s' "${algo}"
}

CASES_DIR="${ROOT_DIR}/test/cases"
ENERGY_MODE="none"
ENERGY_OUT_DIR="${ROOT_DIR}/test/prof/results/measurements"
RAPL_PATH=""
GPU_INDEX="0"
GPU_INTERVAL_MS="100"
RESULT_FILE="${ROOT_DIR}/test/prof/results/test_output.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --energy)
            if [[ $# -lt 2 ]]; then
                echo "error: --energy needs a value"
                usage
                exit 2
            fi
            ENERGY_MODE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --cases-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --cases-dir needs a value"
                usage
                exit 2
            fi
            CASES_DIR="$2"
            shift 2
            ;;
        --energy-out-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --energy-out-dir needs a value"
                usage
                exit 2
            fi
            ENERGY_OUT_DIR="$2"
            shift 2
            ;;
        --rapl-path)
            if [[ $# -lt 2 ]]; then
                echo "error: --rapl-path needs a value"
                usage
                exit 2
            fi
            RAPL_PATH="$2"
            shift 2
            ;;
        --gpu-index)
            if [[ $# -lt 2 ]]; then
                echo "error: --gpu-index needs a value"
                usage
                exit 2
            fi
            GPU_INDEX="$2"
            shift 2
            ;;
        --gpu-interval-ms)
            if [[ $# -lt 2 ]]; then
                echo "error: --gpu-interval-ms needs a value"
                usage
                exit 2
            fi
            GPU_INTERVAL_MS="$2"
            shift 2
            ;;
        --output)
            if [[ $# -lt 2 ]]; then
                echo "error: --output needs a value"
                usage
                exit 2
            fi
            RESULT_FILE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "error: unknown option: $1"
            usage
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

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

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

EXPECTED_PASS_MARKER="Top-k correctness vs testcase answer: OK"
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
    REQUESTED_TYPES=()
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
        echo "Types   : auto-detected from ${CASES_DIR}"
    fi
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
    echo "Types   : auto-detected from ${CASES_DIR}"
fi
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

    if [[ ${#REQUESTED_TYPES[@]} -gt 0 ]]; then
        TYPES=("${REQUESTED_TYPES[@]}")
    else
        # Find unique type directories inside the algorithm subdirectories
        mapfile -t TYPES < <(find "${CASES_DIR}" -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort -u)
    fi

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

        # Find all .case files for this TYPE across all algorithm directories
        mapfile -t CASES < <(find "${CASES_DIR}" -mindepth 2 -maxdepth 2 -type d -name "${TYPE}" -exec find {} -maxdepth 1 -type f -name '*.case' \; | sort)

        if [[ ${#CASES[@]} -eq 0 ]]; then
            echo "  Type ${TYPE}: skipped (no .case files found in ${CASES_DIR}/*/${TYPE})"
            {
                echo "-- Type: ${TYPE} --"
                echo "Skipped: no .case files found"
                echo
            } >>"${RESULT_FILE}"
            continue
        fi

        echo "  Type ${TYPE}: ${#CASES[@]} case(s)"
        {
            echo "-- Type: ${TYPE} --"
            echo "Cases: ${#CASES[@]}"
            echo
        } >>"${RESULT_FILE}"

        for CASE in "${CASES[@]}"; do
            NAME="$(basename "${CASE}")"
            # Optional: to make the output clearer in the log file, we can extract the parent directory name
            # ALGO_DIR="$(basename "$(dirname "$(dirname "${CASE}")")")"
            OUTPUT="$(mktemp)"
            CASE_STATUS="FAIL"
            CASE_REASON="non-zero exit"
            ENERGY_CASE_FILE=""

            CASE_ENERGY_MODE="$(resolve_energy_mode "${BACKEND}" "${ENERGY_MODE}")" || {
                echo "error: unsupported energy mode '${ENERGY_MODE}'"
                exit 2
            }

            if [[ "${CASE_ENERGY_MODE}" == "rapl" || "${CASE_ENERGY_MODE}" == "gpu" ]]; then
                CASE_STEM="${NAME%.case}"
                CASE_ALGO="$(extract_case_algo "${CASE}")"
                ENERGY_CASE_FILE="${ENERGY_OUT_DIR}/${RUN_ID}_${BACKEND}_${TYPE}_${CASE_ALGO}_${CASE_STEM}_${CASE_ENERGY_MODE}.txt"
            fi

            if [[ "${CASE_ENERGY_MODE}" == "none" ]]; then
                RUN_CMD=("${BINARY_PATH}" "${CASE}")
            elif [[ "${CASE_ENERGY_MODE}" == "rapl" ]]; then
                RUN_CMD=("${ROOT_DIR}/test/perf/measure_rapl.sh" "--out" "${ENERGY_CASE_FILE}")
                if [[ -n "${RAPL_PATH}" ]]; then
                    RUN_CMD+=("--path" "${RAPL_PATH}")
                fi
                RUN_CMD+=("--" "${BINARY_PATH}" "${CASE}")
            else
                RUN_CMD=("${ROOT_DIR}/test/perf/measure_smi.sh" "--out" "${ENERGY_CASE_FILE}" "--gpu-index" "${GPU_INDEX}" "--interval-ms" "${GPU_INTERVAL_MS}" "--" "${BINARY_PATH}" "${CASE}")
            fi

            if "${RUN_CMD[@]}" >"${OUTPUT}" 2>&1; then
                if grep -Fq "${EXPECTED_PASS_MARKER}" "${OUTPUT}"; then
                    echo "    [PASS] ${NAME}"
                    TYPE_PASS=$((TYPE_PASS + 1))
                    CASE_STATUS="PASS"
                    CASE_REASON="ok"
                else
                    echo "    [FAIL] ${NAME} (PASS marker missing)"
                    tail -n 20 "${OUTPUT}" || true
                    TYPE_FAIL=$((TYPE_FAIL + 1))
                    CASE_REASON="PASS marker missing"
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
