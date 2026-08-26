#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/find_root.sh"
ROOT_DIR="$(find_root_dir "${SCRIPT_DIR}")" || exit 1

RESULTS_DIR="${ROOT_DIR}/bench/results"
mkdir -p "${RESULTS_DIR}/raw/energy/measurements"

rm -f "${RESULTS_DIR}/raw/energy/measurements/"*.txt 2>/dev/null || true

POWERCAP_ROOT="${MEASURE_RAPL_ROOT:-/sys/class/powercap}"
if command -v sudo >/dev/null 2>&1; then
    sudo -n chmod a+r "${POWERCAP_ROOT}"/intel-rapl:*/energy_uj \
        "${POWERCAP_ROOT}"/intel-rapl:*/*/energy_uj 2>/dev/null || true
fi

rapl_readable=0
for f in "${POWERCAP_ROOT}"/intel-rapl:*/energy_uj; do
    [[ -r "${f}" ]] && rapl_readable=1 && break
done
if [[ "${rapl_readable}" != "1" ]]; then
    echo "error: no readable ${POWERCAP_ROOT}/intel-rapl:*/energy_uj - energy data would be empty" >&2
    echo "       fix: sudo chmod a+r ${POWERCAP_ROOT}/intel-rapl:*/energy_uj ${POWERCAP_ROOT}/intel-rapl:*/*/energy_uj" >&2
    exit 1
fi

NPU_OFFLOAD_XCLBIN="${ROOT_DIR}/build/NPU/bitonic.xclbin" \
    python3 "${ROOT_DIR}/bench/lib/runner.py" cpu gpu npu --energy auto \
    --types int uint float double half --q-min 16 --q-max 24 --repeats 3 --verify false \
    --cooldown 15 --min 0 --max 1000000000 \
    --output-json "${RESULTS_DIR}/raw/energy/output.json" \
    --output-raw "${RESULTS_DIR}/raw/energy/output.txt" "$@"
