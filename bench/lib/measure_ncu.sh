#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
    ./bench/lib/measure_ncu.sh [--out <file>] [--kernel <regex>] -- <command> [args...]

Options:
    --out <file>       Write the metric report to this file as well as stdout.
    --kernel <regex>   Only profile kernels matching this regex (default: all).
    --help, -h         Show this help message.
    --                 End script options; remaining args are the command to run.

Examples:
    ./bench/lib/measure_ncu.sh -- ./build/GPU/topk q=24 k=256 dtype=int algo=map_reduce verify=false
    ./bench/lib/measure_ncu.sh --kernel topk_map_kernel -- ./build/GPU/topk q=24 k=8 dtype=int algo=map_reduce

Metrics collected (the three that settle "bandwidth-bound vs latency-bound"):
    - dram__throughput.avg.pct_of_peak_sustained_elapsed
        How much of peak DRAM bandwidth the kernel actually uses.
        High  => memory-bandwidth bound.  Low => something else is the wall.
    - l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
        Sectors moved per load request. 1.0 is perfectly coalesced;
        higher means the access pattern wastes bandwidth (8.0 = 32B moved per 4B used).
    - smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        Share of stalls waiting on memory latency. High => latency bound.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        echo "hint: Nsight Compute ships with the CUDA toolkit; check that ncu is on PATH" >&2
        exit 1
    fi
}

METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
gpu__time_duration.sum,\
launch__occupancy_limit_registers,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

OUT_FILE=""
KERNEL_REGEX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            if [[ $# -lt 2 ]]; then
                echo "error: --out needs a value" >&2
                exit 2
            fi
            OUT_FILE="$2"
            shift 2
            ;;
        --kernel)
            if [[ $# -lt 2 ]]; then
                echo "error: --kernel needs a value" >&2
                exit 2
            fi
            KERNEL_REGEX="$2"
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
        *)
            echo "error: unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "error: missing command to run" >&2
    usage
    exit 2
fi

require_cmd ncu

NCU_ARGS=(--metrics "${METRICS}" --target-processes all --print-summary per-kernel)
if [[ -n "${KERNEL_REGEX}" ]]; then
    NCU_ARGS+=(--kernel-name "regex:${KERNEL_REGEX}")
fi

REPORT="$(ncu "${NCU_ARGS[@]}" "$@" 2>&1)"
STATUS=$?

if [[ -n "${OUT_FILE}" ]]; then
    mkdir -p "$(dirname "${OUT_FILE}")"
    printf '%s\n' "${REPORT}" | tee "${OUT_FILE}"
else
    printf '%s\n' "${REPORT}"
fi

exit "${STATUS}"
