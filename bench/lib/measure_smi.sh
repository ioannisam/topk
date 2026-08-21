#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
    ./bench/lib/measure_smi.sh [--gpu-index <idx>] [--interval-ms <ms>] -- <command> [args...]
    ./bench/lib/measure_smi.sh [--out <file>] [--gpu-index <idx>] [--interval-ms <ms>] -- <command> [args...]
    ./bench/lib/measure_smi.sh --list-gpus

Options:
    --out <file>                Also write report to file (for profiler ingestion).
    --baseline-watts <w>        Idle power (W) to subtract; emits net_energy_joules/net_average_watts.
    --board-baseline-watts <w>  Idle GPU-board power (W) to subtract; emits net_board_energy_joules.
    --gpu-index <idx>           Select GPU index to sample (default: 0).
    --interval-ms <ms>          Sampling interval in milliseconds (default: 100).
    --list-gpus                 List available GPUs and power-management info.
    --help, -h                  Show this help message.
    --                          End script options; remaining args are the command to run.

Examples:
    ./bench/lib/measure_smi.sh -- ./build/GPU/topk q=20 k=256 dtype=int algo=bitonic
    ./bench/lib/measure_smi.sh --out ./bench/results/raw/energy/measurements/gpu_run1.txt -- ./build/GPU/topk q=20 k=256 dtype=int algo=bitonic
    ./bench/lib/measure_smi.sh --gpu-index 0 --interval-ms 100 -- sleep 1

Notes:
    - Uses nvidia-smi power.draw telemetry (Watts).
    - Integrates sampled power over time to estimate energy (Joules).
    - Reports elapsed time, estimated energy, average power and command exit code.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        exit 1
    fi
}

read_gpu_power_w() {
    local idx="$1"
    local raw
    raw="$(nvidia-smi --id="${idx}" --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    if [[ -z "${raw}" ]]; then
        return 1
    fi
    if [[ "${raw}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "${raw}"
        return 0
    fi
    return 1
}

GPU_INDEX="0"
INTERVAL_MS="100"
LIST_ONLY="no"
OUT_FILE=""
BASELINE_WATTS=""
BOARD_BASELINE_WATTS=""

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
        --board-baseline-watts)
            if [[ $# -lt 2 ]]; then
                echo "error: --board-baseline-watts needs a value" >&2
                usage
                exit 2
            fi
            BOARD_BASELINE_WATTS="$2"
            shift 2
            ;;
        --gpu-index)
            if [[ $# -lt 2 ]]; then
                echo "error: --gpu-index needs a value" >&2
                usage
                exit 2
            fi
            GPU_INDEX="$2"
            shift 2
            ;;
        --interval-ms)
            if [[ $# -lt 2 ]]; then
                echo "error: --interval-ms needs a value" >&2
                usage
                exit 2
            fi
            INTERVAL_MS="$2"
            shift 2
            ;;
        --list-gpus)
            LIST_ONLY="yes"
            shift
            ;;
        --help|-h)
            usage
            exit 0
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

require_cmd nvidia-smi

if [[ "${LIST_ONLY}" == "yes" ]]; then
    nvidia-smi --query-gpu=index,name,driver_version,power.management,power.limit --format=csv,noheader
    exit 0
fi

if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "error: --gpu-index must be a non-negative integer" >&2
    exit 2
fi

if [[ ! "${INTERVAL_MS}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL_MS}" -le 0 ]]; then
    echo "error: --interval-ms must be a positive integer" >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    echo "error: missing command to run" >&2
    usage
    exit 2
fi

CMD_STR="$*"

if ! read_gpu_power_w "${GPU_INDEX}" >/dev/null; then
    echo "error: could not read GPU power.draw for index ${GPU_INDEX}" >&2
    echo "hint: verify nvidia-smi is available and power telemetry is supported" >&2
    exit 1
fi

POWERCAP_ROOT="${MEASURE_RAPL_ROOT:-/sys/class/powercap}"
HOST_ENERGY_PATHS=()
HOST_MAX_RANGE_UJS=()
_domain=""
while IFS= read -r _candidate; do
    _domain="$(cat "$(dirname "${_candidate}")/name" 2>/dev/null || true)"
    if [[ -r "${_candidate}" && "${_domain}" == package-* ]]; then
        HOST_ENERGY_PATHS+=("${_candidate}")
        _max_range_path="$(dirname "${_candidate}")/max_energy_range_uj"
        if [[ -r "${_max_range_path}" ]]; then
            HOST_MAX_RANGE_UJS+=("$(<"${_max_range_path}")")
        else
            HOST_MAX_RANGE_UJS+=("")
        fi
    fi
done < <(find -L "${POWERCAP_ROOT}" -maxdepth 2 -name energy_uj -print 2>/dev/null | sort)
HOST_ENERGY_PATH=""
[[ ${#HOST_ENERGY_PATHS[@]} -gt 0 ]] && HOST_ENERGY_PATH="$(IFS=,; echo "${HOST_ENERGY_PATHS[*]}")"

samples_file="$(mktemp)"
trap 'rm -f "${samples_file}"' EXIT

HOST_PREV_UJS=()
for _p in "${HOST_ENERGY_PATHS[@]}"; do
    HOST_PREV_UJS+=("$(<"${_p}")")
done
HOST_ACCUM_UJ=0
HOST_WRAP_EVENTS=0

sample_once() {
    local ts power i host_cur_uj host_delta_uj
    ts="$(date +%s.%N)"
    if power="$(read_gpu_power_w "${GPU_INDEX}")"; then
        printf '%s %s\n' "${ts}" "${power}" >>"${samples_file}"
    fi
    for i in "${!HOST_ENERGY_PATHS[@]}"; do
        [[ -r "${HOST_ENERGY_PATHS[$i]}" ]] || continue
        host_cur_uj="$(<"${HOST_ENERGY_PATHS[$i]}")"
        host_delta_uj=$((host_cur_uj - HOST_PREV_UJS[i]))
        if [[ ${host_delta_uj} -lt 0 ]]; then
            if [[ -n "${HOST_MAX_RANGE_UJS[$i]}" && ${HOST_MAX_RANGE_UJS[$i]} -gt 0 ]]; then
                host_delta_uj=$((host_delta_uj + HOST_MAX_RANGE_UJS[i]))
                HOST_WRAP_EVENTS=$((HOST_WRAP_EVENTS + 1))
            else
                host_delta_uj=0
            fi
        fi
        HOST_ACCUM_UJ=$((HOST_ACCUM_UJ + host_delta_uj))
        HOST_PREV_UJS[i]="${host_cur_uj}"
    done
}

start_ts="$(date +%s.%N)"
"$@" &
cmd_pid=$!

sample_once

interval_s="$(awk -v ms="${INTERVAL_MS}" 'BEGIN { printf "%.6f", ms / 1000.0 }')"
while kill -0 "${cmd_pid}" 2>/dev/null; do
    sleep "${interval_s}" &
    sleep_pid=$!
    wait -n "${cmd_pid}" "${sleep_pid}" 2>/dev/null || true
    if kill -0 "${sleep_pid}" 2>/dev/null; then
        kill "${sleep_pid}" 2>/dev/null
    fi
    wait "${sleep_pid}" 2>/dev/null || true
    kill -0 "${cmd_pid}" 2>/dev/null && sample_once
done

set +e
wait "${cmd_pid}"
cmd_status=$?
set -e

sample_once
end_ts="$(date +%s.%N)"

set +e
REPORT="$(awk \
    -v start_ts="${start_ts}" \
    -v end_ts="${end_ts}" \
    -v gpu_index="${GPU_INDEX}" \
    -v interval_ms="${INTERVAL_MS}" \
    -v cmd_status="${cmd_status}" \
    -v baseline_watts="${BASELINE_WATTS}" \
    -v board_baseline_watts="${BOARD_BASELINE_WATTS}" \
    -v cmd_str="${CMD_STR}" \
    -v host_path="${HOST_ENERGY_PATH}" \
    -v host_accum_uj="${HOST_ACCUM_UJ}" \
    -v host_wrap_events="${HOST_WRAP_EVENTS}" \
'BEGIN {
    n = 0
}
{
    t[++n] = $1
    p[n] = $2
}
END {
    dt_total = end_ts - start_ts
    if (dt_total <= 0) {
        print "error: non-positive elapsed time" > "/dev/stderr"
        exit 3
    }

    if (n == 0) {
        print "error: no GPU power samples collected" > "/dev/stderr"
        exit 4
    }

    board_energy = 0.0
    if (n == 1) {
        board_energy = p[1] * dt_total
    } else {
        for (i = 1; i < n; i++) {
            dt = t[i+1] - t[i]
            if (dt > 0) {
                board_energy += ((p[i] + p[i+1]) / 2.0) * dt
            }
        }
    }

    host_energy = 0.0
    host_available = "no"
    if (host_path != "") {
        host_energy = host_accum_uj / 1000000.0
        host_available = "yes"
    }

    energy = board_energy + host_energy
    avg_w = energy / dt_total

    print "GPU measurement"
    if (cmd_str != "") print "Command: " cmd_str
    print "- gpu_index: " gpu_index
    print "- sample_interval_ms: " interval_ms
    print "- sample_count: " n
    print "- host_rapl_available: " host_available
    printf("- elapsed_seconds: %.6f\n", dt_total)
    printf("- board_energy_joules: %.6f\n", board_energy)
    printf("- board_average_watts: %.6f\n", board_energy / dt_total)
    printf("- host_energy_joules: %.6f\n", host_energy)
    printf("- energy_joules: %.6f\n", energy)
    printf("- average_watts: %.6f\n", avg_w)
    if (board_baseline_watts != "") {
        net_board_j = board_energy - board_baseline_watts * dt_total
        if (net_board_j < 0) net_board_j = 0
        printf("- board_baseline_watts: %.6f\n", board_baseline_watts)
        printf("- net_board_energy_joules: %.6f\n", net_board_j)
    }
    if (baseline_watts != "") {
        net_j = energy - baseline_watts * dt_total
        net_w = avg_w - baseline_watts
        if (net_j < 0) net_j = 0
        if (net_w < 0) net_w = 0
        printf("- baseline_watts: %.6f\n", baseline_watts)
        printf("- net_energy_joules: %.6f\n", net_j)
        printf("- net_average_watts: %.6f\n", net_w)
    }
    print "- command_exit_code: " cmd_status
}
' "${samples_file}"
 )"
AWK_STATUS=$?
set -e

if [[ ${AWK_STATUS} -ne 0 ]]; then
    REPORT="GPU measurement
Command: ${CMD_STR}
- gpu_index: ${GPU_INDEX}
- error: energy accounting failed (awk exit ${AWK_STATUS})
- command_exit_code: ${cmd_status}"
fi

if [[ -n "${OUT_FILE}" ]]; then
    mkdir -p "$(dirname "${OUT_FILE}")"
    printf '%s\n' "${REPORT}" | tee "${OUT_FILE}"
else
    printf '%s\n' "${REPORT}"
fi

exit "${cmd_status}"
