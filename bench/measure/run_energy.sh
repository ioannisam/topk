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

RESULTS_DIR="${ROOT_DIR}/bench/results"
mkdir -p "${RESULTS_DIR}/raw/energy/measurements"

# Start each sweep clean so the profiler does not mix it with prior runs.
rm -f "${RESULTS_DIR}/raw/energy/measurements/"*.txt 2>/dev/null || true

# Grant RAPL read for this session so the workload runs unprivileged
# (NPU needs its own XRT env, which sudo would strip).
if command -v sudo >/dev/null 2>&1; then
    sudo -n chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj \
        /sys/class/powercap/intel-rapl:*/*/energy_uj 2>/dev/null || true
fi

rapl_readable=0
for f in /sys/class/powercap/intel-rapl:*/energy_uj; do
    [[ -r "${f}" ]] && rapl_readable=1 && break
done
if [[ "${rapl_readable}" != "1" ]]; then
    echo "error: no readable /sys/class/powercap/intel-rapl:*/energy_uj - energy data would be empty" >&2
    echo "       fix: sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*/*/energy_uj" >&2
    exit 1
fi

NPU_OFFLOAD_XCLBIN="${ROOT_DIR}/build/NPU/bitonic.xclbin" \
    python3 "${ROOT_DIR}/bench/lib/runner.py" cpu gpu npu --energy auto \
    --types int uint float double half --q-min 16 --q-max 24 --repeats 3 --verify false \
    --cooldown 15 --min 0 --max 1000000000 \
    --output-json "${RESULTS_DIR}/raw/energy/output.json" \
    --output-raw "${RESULTS_DIR}/raw/energy/output.txt" "$@"
