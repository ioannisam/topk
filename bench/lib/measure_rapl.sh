#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
    sudo ./bench/lib/measure_rapl.sh [--path <energy_uj_path>] -- <command> [args...]
    sudo ./bench/lib/measure_rapl.sh [--out <file>] [--path <energy_uj_path>] -- <command> [args...]
    sudo ./bench/lib/measure_rapl.sh --list-paths

Options:
    --out <file>               Also write report to file (for profiler ingestion).
    --baseline-watts <w>       Idle power (W) to subtract; emits net_energy_joules/net_average_watts.
    --core-baseline-watts <w>  Idle core power (W) to subtract; emits net_core_energy_joules.
    --path <energy_uj_path>    Use a specific RAPL energy counter file.
    --list-paths               List discovered energy_uj paths and readability.
    --help, -h                 Show this help message.
    --                         End script options; remaining args are the command to run.

Examples:
    sudo ./bench/lib/measure_rapl.sh -- ./build/CPU/topk q=20 k=256 dtype=int algo=bitonic
    sudo ./bench/lib/measure_rapl.sh --out ./bench/results/raw/energy/measurements/cpu_run1.txt -- ./build/CPU/topk q=20 k=256 dtype=int algo=bitonic
    sudo ./bench/lib/measure_rapl.sh --path /sys/class/powercap/intel-rapl:0/energy_uj -- sleep 1

Notes:
    - Uses Linux RAPL energy_uj counters (microjoules) (may need sudo).
    - Reports elapsed time, consumed energy and average power.
    - Handles counter wraparound when max_energy_range_uj is available; polls every
      RAPL_POLL_INTERVAL_S seconds (default 0.5) so multi-rollover runs stay correct too.
EOF
}

POWERCAP_ROOT="${MEASURE_RAPL_ROOT:-/sys/class/powercap}"

discover_energy_paths() {
    find -L "${POWERCAP_ROOT}" -maxdepth 2 -name energy_uj -print 2>/dev/null | sort
}

find_all_package_energy_paths() {
    local candidate domain_name found=0
    while IFS= read -r candidate; do
        domain_name="$(cat "$(dirname "${candidate}")/name" 2>/dev/null || true)"
        if [[ -r "${candidate}" && "${domain_name}" == package-* ]]; then
            echo "${candidate}"
            found=1
        fi
    done < <(discover_energy_paths)
    [[ ${found} -eq 1 ]]
}

list_energy_paths() {
    local found=0
    local candidate domain_name
    while IFS= read -r candidate; do
        found=1
        domain_name="$(cat "$(dirname "${candidate}")/name" 2>/dev/null || echo '?')"
        if [[ -r "${candidate}" ]]; then
            echo "readable ${candidate} (${domain_name})"
        else
            echo "not-readable ${candidate} (${domain_name})"
        fi
    done < <(discover_energy_paths)

    if [[ ${found} -eq 0 ]]; then
        echo "no energy_uj paths found under /sys/class/powercap" >&2
        return 1
    fi

    return 0
}

ENERGY_PATH=""
LIST_ONLY="no"
OUT_FILE=""
BASELINE_WATTS=""
CORE_BASELINE_WATTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            if [[ $# -lt 2 ]]; then
                echo "error: --out needs a value" >&2
                usage
                exit 2
            fi
            OUT_FILE="$2"
            shift 2
            ;;
        --baseline-watts)
            if [[ $# -lt 2 ]]; then
                echo "error: --baseline-watts needs a value" >&2
                usage
                exit 2
            fi
            BASELINE_WATTS="$2"
            shift 2
            ;;
        --core-baseline-watts)
            if [[ $# -lt 2 ]]; then
                echo "error: --core-baseline-watts needs a value" >&2
                usage
                exit 2
            fi
            CORE_BASELINE_WATTS="$2"
            shift 2
            ;;
        --path)
            if [[ $# -lt 2 ]]; then
                echo "error: --path needs a value" >&2
                usage
                exit 2
            fi
            ENERGY_PATH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --list-paths)
            LIST_ONLY="yes"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ "${LIST_ONLY}" == "yes" ]]; then
    list_energy_paths
    exit 0
fi

if [[ $# -eq 0 ]]; then
    echo "error: missing command to run" >&2
    usage
    exit 2
fi

CMD_STR="$*"

ENERGY_PATHS=()
if [[ -n "${ENERGY_PATH}" ]]; then
    ENERGY_PATHS=("${ENERGY_PATH}")
else
    while IFS= read -r _p; do
        ENERGY_PATHS+=("${_p}")
    done < <(find_all_package_energy_paths)
    if [[ ${#ENERGY_PATHS[@]} -eq 0 ]]; then
        echo "error: could not find a readable energy_uj path under /sys/class/powercap" >&2
        echo "hint: run ./bench/lib/measure_rapl.sh --list-paths to inspect availability" >&2
        echo "hint: pass --path explicitly, or run with sufficient permissions (for example with sudo)" >&2
        exit 1
    fi
fi

for _p in "${ENERGY_PATHS[@]}"; do
    if [[ ! -r "${_p}" ]]; then
        echo "error: energy path is not readable: ${_p}" >&2
        exit 1
    fi
done

# On a multi-socket host, every package-* domain found is tracked and summed (see
# find_all_package_energy_paths); --path overrides this with a single explicit counter.
MAX_RANGE_UJS=()
CORE_ENERGY_PATHS=()
CORE_MAX_RANGE_UJS=()
for _p in "${ENERGY_PATHS[@]}"; do
    _max_range_path="$(dirname "${_p}")/max_energy_range_uj"
    if [[ -r "${_max_range_path}" ]]; then
        MAX_RANGE_UJS+=("$(<"${_max_range_path}")")
    else
        MAX_RANGE_UJS+=("")
    fi

    _core_path=""
    _core_max_range=""
    _pkg_dir="$(dirname "${_p}")"
    for _sub in "${_pkg_dir}"/intel-rapl:*; do
        if [[ -r "${_sub}/name" && "$(<"${_sub}/name")" == "core" && -r "${_sub}/energy_uj" ]]; then
            _core_path="${_sub}/energy_uj"
            [[ -r "${_sub}/max_energy_range_uj" ]] && _core_max_range="$(<"${_sub}/max_energy_range_uj")"
            break
        fi
    done
    CORE_ENERGY_PATHS+=("${_core_path}")
    CORE_MAX_RANGE_UJS+=("${_core_max_range}")
done

POLL_INTERVAL_S="${RAPL_POLL_INTERVAL_S:-0.5}"

ACCUM_UJ=0
CORE_ACCUM_UJ=0
WRAP_EVENTS=0

accumulate_tick() {
    local i cur_uj delta_uj
    for i in "${!ENERGY_PATHS[@]}"; do
        [[ -r "${ENERGY_PATHS[$i]}" ]] || continue
        cur_uj="$(<"${ENERGY_PATHS[$i]}")"
        delta_uj=$((cur_uj - PREV_UJS[i]))
        if [[ ${delta_uj} -lt 0 ]]; then
            if [[ -n "${MAX_RANGE_UJS[$i]}" && ${MAX_RANGE_UJS[$i]} -gt 0 ]]; then
                delta_uj=$((delta_uj + MAX_RANGE_UJS[i]))
                WRAP_EVENTS=$((WRAP_EVENTS + 1))
            else
                delta_uj=0
            fi
        fi
        ACCUM_UJ=$((ACCUM_UJ + delta_uj))
        PREV_UJS[i]="${cur_uj}"
    done
    for i in "${!CORE_ENERGY_PATHS[@]}"; do
        [[ -n "${CORE_ENERGY_PATHS[$i]}" && -r "${CORE_ENERGY_PATHS[$i]}" ]] || continue
        cur_uj="$(<"${CORE_ENERGY_PATHS[$i]}")"
        delta_uj=$((cur_uj - CORE_PREV_UJS[i]))
        if [[ ${delta_uj} -lt 0 ]]; then
            if [[ -n "${CORE_MAX_RANGE_UJS[$i]}" && ${CORE_MAX_RANGE_UJS[$i]} -gt 0 ]]; then
                delta_uj=$((delta_uj + CORE_MAX_RANGE_UJS[i]))
            else
                delta_uj=0
            fi
        fi
        CORE_ACCUM_UJ=$((CORE_ACCUM_UJ + delta_uj))
        CORE_PREV_UJS[i]="${cur_uj}"
    done
}

PREV_UJS=()
for _p in "${ENERGY_PATHS[@]}"; do
    PREV_UJS+=("$(<"${_p}")")
done
CORE_PREV_UJS=()
for _p in "${CORE_ENERGY_PATHS[@]}"; do
    if [[ -n "${_p}" ]]; then
        CORE_PREV_UJS+=("$(<"${_p}")")
    else
        CORE_PREV_UJS+=("0")
    fi
done
START_TS="$(date +%s.%N)"

# `set -e` would abort here before CMD_STATUS is captured, dropping the whole report (and the
# --out file) for exactly the runs whose failure the profiler needs to see.
set +e
"$@" &
CMD_PID=$!
while kill -0 "${CMD_PID}" 2>/dev/null; do
    sleep "${POLL_INTERVAL_S}" &
    SLEEP_PID=$!
    wait -n "${CMD_PID}" "${SLEEP_PID}" 2>/dev/null
    if kill -0 "${SLEEP_PID}" 2>/dev/null; then
        kill "${SLEEP_PID}" 2>/dev/null
    fi
    wait "${SLEEP_PID}" 2>/dev/null
    kill -0 "${CMD_PID}" 2>/dev/null && accumulate_tick
done
wait "${CMD_PID}"
CMD_STATUS=$?
set -e

accumulate_tick
END_TS="$(date +%s.%N)"

ENERGY_PATHS_JOINED="$(IFS=,; echo "${ENERGY_PATHS[*]}")"
CORE_PATH_PRESENT="no"
for _p in "${CORE_ENERGY_PATHS[@]}"; do
    [[ -n "${_p}" ]] && CORE_PATH_PRESENT="yes"
done

set +e
REPORT="$(awk \
    -v path="${ENERGY_PATHS_JOINED}" \
    -v accum_uj="${ACCUM_UJ}" \
    -v wrap_events="${WRAP_EVENTS}" \
    -v start_ts="${START_TS}" \
    -v end_ts="${END_TS}" \
    -v cmd_status="${CMD_STATUS}" \
    -v baseline_watts="${BASELINE_WATTS}" \
    -v core_baseline_watts="${CORE_BASELINE_WATTS}" \
    -v core_path_present="${CORE_PATH_PRESENT}" \
    -v core_accum_uj="${CORE_ACCUM_UJ}" \
    -v cmd_str="${CMD_STR}" \
'BEGIN {
    delta_uj = accum_uj
    wrapped = (wrap_events > 0) ? "yes" : "no"

    dt = end_ts - start_ts
    if (dt <= 0) {
        print "error: non-positive elapsed time" > "/dev/stderr"
        exit 4
    }

    joules = delta_uj / 1000000.0
    watts = joules / dt

    core_available = "no"
    core_joules = 0.0
    core_watts = 0.0
    if (core_path_present == "yes") {
        core_joules = core_accum_uj / 1000000.0
        core_watts = core_joules / dt
        core_available = "yes"
    }

    print "RAPL measurement"
    if (cmd_str != "") print "Command: " cmd_str
    print "- energy_path: " path
    print "- wrapped: " wrapped
    print "- core_available: " core_available
    printf("- elapsed_seconds: %.6f\n", dt)
    printf("- energy_joules: %.6f\n", joules)
    printf("- average_watts: %.6f\n", watts)
    printf("- core_energy_joules: %.6f\n", core_joules)
    printf("- core_average_watts: %.6f\n", core_watts)
    if (baseline_watts != "") {
        net_j = joules - baseline_watts * dt
        net_w = watts - baseline_watts
        if (net_j < 0) net_j = 0
        if (net_w < 0) net_w = 0
        printf("- baseline_watts: %.6f\n", baseline_watts)
        printf("- net_energy_joules: %.6f\n", net_j)
        printf("- net_average_watts: %.6f\n", net_w)
    }
    if (core_baseline_watts != "") {
        net_core_j = core_joules - core_baseline_watts * dt
        if (net_core_j < 0) net_core_j = 0
        printf("- core_baseline_watts: %.6f\n", core_baseline_watts)
        printf("- net_core_energy_joules: %.6f\n", net_core_j)
    }
    print "- command_exit_code: " cmd_status
}
'
 )"
AWK_STATUS=$?
set -e

if [[ ${AWK_STATUS} -ne 0 ]]; then
    REPORT="RAPL measurement
Command: ${CMD_STR}
- energy_path: ${ENERGY_PATHS_JOINED}
- error: energy accounting failed (awk exit ${AWK_STATUS})
- command_exit_code: ${CMD_STATUS}"
fi

if [[ -n "${OUT_FILE}" ]]; then
    mkdir -p "$(dirname "${OUT_FILE}")"
    printf '%s\n' "${REPORT}" | tee "${OUT_FILE}"
else
    printf '%s\n' "${REPORT}"
fi

exit "${CMD_STATUS}"
