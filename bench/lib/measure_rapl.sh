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
    - Handles counter wraparound when max_energy_range_uj is available.
EOF
}

# -L to follow the class symlinks, but depth 2 so the per-domain `device/` back-links do not
# re-surface the same counters under aliased paths.
discover_energy_paths() {
    find -L /sys/class/powercap -maxdepth 2 -name energy_uj -print 2>/dev/null | sort
}

# Match the in-process counter in common/src/energy.cpp: integrate the package domain, not
# whichever energy_uj happens to sort first (psys and intel-rapl-mmio also expose one).
find_default_energy_path() {
    local candidate domain_name
    while IFS= read -r candidate; do
        domain_name="$(cat "$(dirname "${candidate}")/name" 2>/dev/null || true)"
        if [[ -r "${candidate}" && "${domain_name}" == package-* ]]; then
            echo "${candidate}"
            return 0
        fi
    done < <(discover_energy_paths)
    return 1
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

if [[ -z "${ENERGY_PATH}" ]]; then
    if ! ENERGY_PATH="$(find_default_energy_path)"; then
        echo "error: could not find a readable energy_uj path under /sys/class/powercap" >&2
        echo "hint: run ./bench/lib/measure_rapl.sh --list-paths to inspect availability" >&2
        echo "hint: pass --path explicitly, or run with sufficient permissions (for example with sudo)" >&2
        exit 1
    fi
fi

if [[ ! -r "${ENERGY_PATH}" ]]; then
    echo "error: energy path is not readable: ${ENERGY_PATH}" >&2
    exit 1
fi

MAX_RANGE_PATH="$(dirname "${ENERGY_PATH}")/max_energy_range_uj"
MAX_RANGE_UJ=""
if [[ -r "${MAX_RANGE_PATH}" ]]; then
    MAX_RANGE_UJ="$(<"${MAX_RANGE_PATH}")"
fi

CORE_ENERGY_PATH=""
CORE_MAX_RANGE_UJ=""
_pkg_dir="$(dirname "${ENERGY_PATH}")"
for _sub in "${_pkg_dir}"/intel-rapl:*; do
    if [[ -r "${_sub}/name" && "$(<"${_sub}/name")" == "core" && -r "${_sub}/energy_uj" ]]; then
        CORE_ENERGY_PATH="${_sub}/energy_uj"
        [[ -r "${_sub}/max_energy_range_uj" ]] && CORE_MAX_RANGE_UJ="$(<"${_sub}/max_energy_range_uj")"
        break
    fi
done

START_UJ="$(<"${ENERGY_PATH}")"
CORE_START_UJ=""
[[ -n "${CORE_ENERGY_PATH}" ]] && CORE_START_UJ="$(<"${CORE_ENERGY_PATH}")"
START_TS="$(date +%s.%N)"

# `set -e` would abort here before CMD_STATUS is captured, dropping the whole report (and the
# --out file) for exactly the runs whose failure the profiler needs to see.
set +e
"$@"
CMD_STATUS=$?
set -e

END_TS="$(date +%s.%N)"
END_UJ="$(<"${ENERGY_PATH}")"
CORE_END_UJ=""
[[ -n "${CORE_ENERGY_PATH}" ]] && CORE_END_UJ="$(<"${CORE_ENERGY_PATH}")"

REPORT="$(awk \
    -v path="${ENERGY_PATH}" \
    -v start_uj="${START_UJ}" \
    -v end_uj="${END_UJ}" \
    -v start_ts="${START_TS}" \
    -v end_ts="${END_TS}" \
    -v max_range_uj="${MAX_RANGE_UJ}" \
    -v cmd_status="${CMD_STATUS}" \
    -v baseline_watts="${BASELINE_WATTS}" \
    -v core_baseline_watts="${CORE_BASELINE_WATTS}" \
    -v core_path="${CORE_ENERGY_PATH}" \
    -v core_start_uj="${CORE_START_UJ}" \
    -v core_end_uj="${CORE_END_UJ}" \
    -v core_max_range_uj="${CORE_MAX_RANGE_UJ}" \
    -v cmd_str="${CMD_STR}" \
'BEGIN {
    delta_uj = end_uj - start_uj
    wrapped = "no"

    if (delta_uj < 0) {
        if (max_range_uj != "" && max_range_uj > 0) {
            delta_uj += max_range_uj
            wrapped = "yes"
        } else {
            print "error: negative energy delta and max_energy_range_uj unavailable" > "/dev/stderr"
            exit 3
        }
    }

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
    if (core_path != "" && core_start_uj != "" && core_end_uj != "") {
        core_delta_uj = core_end_uj - core_start_uj
        if (core_delta_uj < 0 && core_max_range_uj != "" && core_max_range_uj > 0) {
            core_delta_uj += core_max_range_uj
        }
        if (core_delta_uj >= 0) {
            core_joules = core_delta_uj / 1000000.0
            core_watts = core_joules / dt
            core_available = "yes"
        }
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

if [[ -n "${OUT_FILE}" ]]; then
    mkdir -p "$(dirname "${OUT_FILE}")"
    printf '%s\n' "${REPORT}" | tee "${OUT_FILE}"
else
    printf '%s\n' "${REPORT}"
fi

exit "${CMD_STATUS}"