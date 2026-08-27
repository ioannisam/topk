#!/usr/bin/env bash

cases=(
    "Small  10 16   bitonic float"
    "Small  10 16   bitonic half"
    "Small  10 16   bitonic double"
    "Medium 20 256  bitonic float"
    "Medium 20 256  bitonic half"
    "Medium 20 256  bitonic double"
    "Large  23 1024 bitonic float"
    "Large  23 1024 bitonic half"
    "Large  23 1024 bitonic double"
    "Small  10 16   map_reduce float"
    "Small  10 16   map_reduce half"
    "Small  10 16   map_reduce double"
    "Medium 20 256  map_reduce float"
    "Medium 20 256  map_reduce half"
    "Medium 20 256  map_reduce double"
    "Large  23 1024 map_reduce float"
    "Large  23 1024 map_reduce half"
    "Large  23 1024 map_reduce double"
    "Small  10 16   gt float"
    "Small  10 16   gt half"
    "Small  10 16   gt double"
    "Medium 20 256  gt float"
    "Medium 20 256  gt half"
    "Medium 20 256  gt double"
    "Large  23 1024 gt float"
    "Large  23 1024 gt half"
    "Large  23 1024 gt double"
)

configure_root_build() {
    local cpu_opt="$1" gpu_opt="$2" npu_opt="$3"
    echo ">> Configuring root build (CPU=${cpu_opt} GPU=${gpu_opt} NPU=${npu_opt})..."
    if ! cmake -S "${ROOT_DIR}" -B "${ROOT_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
        -DTOPK_BUILD_CPU="${cpu_opt}" -DTOPK_BUILD_GPU="${gpu_opt}" -DTOPK_BUILD_NPU="${npu_opt}" \
        >/dev/null 2>&1; then
        echo "Error: CMake configure failed!" >&2
        return 1
    fi
}

# run_suite <phase_label> <out_file>
# Writes one result line per case to <out_file>.
# Uses globals: BACKEND_LC, BACKEND_DIR, BACKEND_TARGET, TARGET_ALGO,
#               EXECUTABLE, ROOT_BUILD_DIR, TIMEOUT_SECONDS, cases[].
run_suite() {
    local phase_label="$1" out_file="$2"
    local cpu_opt="OFF" gpu_opt="OFF" npu_opt="OFF"

    case "${BACKEND_LC}" in
        cpu) cpu_opt="ON" ;;
        gpu) gpu_opt="ON" ;;
        npu) npu_opt="ON" ;;
    esac

    echo ">> Building ${BACKEND_DIR} (${phase_label})..."
    if ! configure_root_build "${cpu_opt}" "${gpu_opt}" "${npu_opt}"; then return 1; fi
    if ! cmake --build "${ROOT_BUILD_DIR}" --config Release --target "${BACKEND_TARGET}" >/dev/null 2>&1; then
        echo "Error: Build failed!" >&2
        return 1
    fi
    if [[ ! -x "${EXECUTABLE}" ]]; then
        echo "Error: Executable not found: ${EXECUTABLE}" >&2
        return 1
    fi

    > "${out_file}"

    for case_info in "${cases[@]}"; do
        read -r size_name q k algo dtype <<< "${case_info}"

        [[ "${TARGET_ALGO}" != "all" && "${algo}" != "${TARGET_ALGO}" ]] && continue

        if [[ "${BACKEND_LC}" == "npu" && "${algo}" != "gt" ]]; then
            local xclbin_path="${ROOT_BUILD_DIR}/${BACKEND_DIR}/${algo}.xclbin"
            if [[ ! -f "${xclbin_path}" ]]; then
                echo "   [!] Skipping: Missing xclbin for ${algo} at ${xclbin_path}"
                echo "${size_name} ${q} ${k} ${algo} ${dtype} ERROR ERROR" >> "${out_file}"
                continue
            fi
            export NPU_OFFLOAD_XCLBIN="${xclbin_path}"
        fi

        local cmd=("${EXECUTABLE}" "q=${q}" "k=${k}" "algo=${algo}" "dtype=${dtype}")
        [[ "${algo}" == "bitonic" ]] && cmd+=("run=both")

        local min_e2e="" min_algo="" found_e2e=0 found_algo=0 runs=3 exit_code=0

        for ((i=1; i<=runs; i++)); do
            local ERR_LOG; ERR_LOG="$(mktemp)"
            local RAW_OUTPUT
            RAW_OUTPUT=$(timeout "${TIMEOUT_SECONDS}"s "${cmd[@]}" 2>>"${ERR_LOG}") || exit_code=$?

            if [[ ${exit_code} -ne 0 ]]; then
                found_e2e=0; found_algo=0; min_e2e=""; min_algo=""
                if [[ ${exit_code} -eq 124 ]]; then
                    echo "   [!] ${algo} (${dtype}) timed out after ${TIMEOUT_SECONDS}s for q=${q}."
                else
                    echo "   [!] ${algo} (${dtype}) failed for q=${q}."
                fi
                if [[ -s "${ERR_LOG}" ]]; then
                    sed 's/^/       /' "${ERR_LOG}"
                fi
                rm -f "${ERR_LOG}"
                break
            fi

            local duration_e2e duration_algo
            duration_e2e=$(echo "${RAW_OUTPUT}" | grep -iE "end-to-end time \(ms\)" | awk -F':' '{print $2}' | tr -d ' ' | tail -n 1)
            duration_algo=$(echo "${RAW_OUTPUT}" | grep -iE "algorithmic time \(ms\)" | awk -F':' '{print $2}' | tr -d ' ' | tail -n 1)

            if [[ -n "${duration_e2e}" && "${duration_e2e}" != "skipped" ]]; then
                if [[ ${found_e2e} -eq 0 ]]; then
                    min_e2e="${duration_e2e}"; found_e2e=1
                else
                    min_e2e=$(awk -v current="${duration_e2e}" -v min="${min_e2e}" 'BEGIN { print (current < min) ? current : min }')
                fi
            fi
            if [[ -n "${duration_algo}" && "${duration_algo}" != "skipped" ]]; then
                if [[ ${found_algo} -eq 0 ]]; then
                    min_algo="${duration_algo}"; found_algo=1
                else
                    min_algo=$(awk -v current="${duration_algo}" -v min="${min_algo}" 'BEGIN { print (current < min) ? current : min }')
                fi
            fi

            rm -f "${ERR_LOG}"
        done

        local e2e_out algo_out
        if [[ ${found_e2e} -eq 1 ]]; then
            e2e_out="$(awk -v ms="${min_e2e}" 'BEGIN { printf "%.3f", ms }')"
        else
            e2e_out="ERROR"
        fi
        if [[ ${found_algo} -eq 1 ]]; then
            algo_out="$(awk -v ms="${min_algo}" 'BEGIN { printf "%.3f", ms }')"
        else
            algo_out="ERROR"
        fi

        echo "${size_name} ${q} ${k} ${algo} ${dtype} ${e2e_out} ${algo_out}" >> "${out_file}"
    done
    echo ">> ${phase_label} run complete."
}

# parse_backend_algo <backend_arg> <algo_arg>
# Sets BACKEND_LC, BACKEND_DIR, BACKEND_TARGET, TARGET_ALGO in the caller's scope.
# Calls usage() on --help or invalid input (usage must be defined by the caller).
parse_backend_algo() {
    local backend_input="${1:-cpu}" algo_input="${2:-all}"

    BACKEND_LC="$(echo "${backend_input}" | tr '[:upper:]' '[:lower:]')"
    case "${BACKEND_LC}" in
        cpu) BACKEND_DIR="CPU"; BACKEND_TARGET="cpu_topk" ;;
        gpu) BACKEND_DIR="GPU"; BACKEND_TARGET="gpu_topk" ;;
        npu) BACKEND_DIR="NPU"; BACKEND_TARGET="npu_topk" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: Unknown backend '${backend_input}'." >&2; usage; exit 2 ;;
    esac

    if [[ "${BACKEND_LC}" == "npu" && -z "${XILINX_XRT:-}" ]]; then
        if [[ -f "/opt/xilinx/xrt/setup.sh" ]]; then
            source "/opt/xilinx/xrt/setup.sh"
        else
            echo "Warning: XILINX_XRT is not set and /opt/xilinx/xrt/setup.sh not found. Execution may fail." >&2
        fi
    fi

    ALGO_LC="$(echo "${algo_input}" | tr '[:upper:]' '[:lower:]')"
    case "${ALGO_LC}" in
        bitonic|map_reduce|gt|all) TARGET_ALGO="${ALGO_LC}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: Unknown algorithm '${algo_input}'." >&2; usage; exit 2 ;;
    esac
}
