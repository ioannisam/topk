#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: scripts/run_testcases.sh <backends> [types]"
    echo "Examples:"
    echo "  scripts/run_testcases.sh cpu"
    echo "  scripts/run_testcases.sh cpu float,int,uint"
    echo "  scripts/run_testcases.sh '{cpu,npu}' '{float,int,uint}'"
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

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

EXPECTED_PASS_MARKER="Top-k correctness vs testcase answer: OK"
RESULTS_DIR="${ROOT_DIR}/test/results"
RESULT_FILE="${RESULTS_DIR}/test_output.txt"

ARGS=("$@")
ARGC=${#ARGS[@]}
IDX=0

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

mkdir -p "${RESULTS_DIR}"
{
    echo "Testcase Run Output"
    echo "Backends: ${BACKENDS_RAW}"
    if [[ -n "${TYPES_RAW}" ]]; then
        echo "Types   : ${TYPES_RAW}"
    else
        echo "Types   : auto-detected from test/cases"
    fi
    echo
} >"${RESULT_FILE}"

echo "Running testcase suite"
echo "Backends: ${BACKENDS_RAW}"
if [[ -n "${TYPES_RAW}" ]]; then
    echo "Types   : ${TYPES_RAW}"
else
    echo "Types   : auto-detected from test/cases"
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
        mapfile -t TYPES < <(find "${ROOT_DIR}/test/cases" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
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

        CASE_DIR_BACKEND_TYPE="${ROOT_DIR}/test/cases/${BACKEND}/${TYPE}"
        CASE_DIR_TYPE="${ROOT_DIR}/test/cases/${TYPE}"

        if [[ -d "${CASE_DIR_BACKEND_TYPE}" ]]; then
            CASE_DIR="${CASE_DIR_BACKEND_TYPE}"
        elif [[ -d "${CASE_DIR_TYPE}" ]]; then
            CASE_DIR="${CASE_DIR_TYPE}"
        else
            echo "  Type ${TYPE}: skipped (no directory found)"
            {
                echo "-- Type: ${TYPE} --"
                echo "Skipped: no directory found"
                echo
            } >>"${RESULT_FILE}"
            continue
        fi

        mapfile -t CASES < <(find "${CASE_DIR}" -maxdepth 1 -type f -name '*.case' | sort)
        if [[ ${#CASES[@]} -eq 0 ]]; then
            echo "  Type ${TYPE}: skipped (no .case files in ${CASE_DIR})"
            {
                echo "-- Type: ${TYPE} --"
                echo "Directory: ${CASE_DIR}"
                echo "Skipped: no .case files"
                echo
            } >>"${RESULT_FILE}"
            continue
        fi

        echo "  Type ${TYPE}: ${#CASES[@]} case(s)"
        {
            echo "-- Type: ${TYPE} --"
            echo "Directory: ${CASE_DIR}"
            echo "Cases: ${#CASES[@]}"
            echo
        } >>"${RESULT_FILE}"

        for CASE in "${CASES[@]}"; do
            NAME="$(basename "${CASE}")"
            OUTPUT="$(mktemp)"
            CASE_STATUS="FAIL"
            CASE_REASON="non-zero exit"

            if "${BINARY_PATH}" "${CASE}" >"${OUTPUT}" 2>&1; then
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
                echo "Command: ${BINARY_PATH} ${CASE}"
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
