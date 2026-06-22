#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/test/prof/results"
mkdir -p "${OUT_DIR}"

OUT_FILE="${OUT_DIR}/specs_validation.md"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

run() {
    local cmd="$1"
    if command -v "${cmd}" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

capture() {
    local NAME="$1"
    shift
    local out_file="${TMP_DIR}/${NAME}.txt"
    if "$@" >"${out_file}" 2>&1; then
        true
    else
        echo "[command failed] $*" >>"${out_file}"
    fi
}

section() {
    echo "## $1"
    echo
}

block() {
    local TITLE="$1"
    local FILE="$2"
    local MISSING="$3"
    echo "### ${TITLE}"
    echo '```text'
    if [[ -f "${FILE}" ]]; then
        cat "${FILE}"
    else
        echo "${MISSING}"
    fi
    echo '```'
    echo
}

# System
capture hostnamectl hostnamectl
capture uname uname -r
capture os_release cat /etc/os-release
if run plasmashell; then
    capture plasma plasmashell --version
fi

# Memory
capture free free -h
if run dmidecode; then
    if [[ "${EUID}" -eq 0 ]]; then
        capture dmidecode_memory dmidecode -t memory
    elif run sudo && sudo -v; then
        capture dmidecode_memory sudo dmidecode -t memory
    else
        echo "dmidecode requires root; re-run with sudo to validate memory channel topology" >"${TMP_DIR}/dmidecode_memory.txt"
    fi
fi

# Storage
capture lsblk lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,MODEL

# CPU
capture lscpu lscpu
capture powercap ls -la /sys/class/powercap
capture energy_paths bash -lc 'find -L /sys/class/powercap -maxdepth 6 \( -name energy_uj -o -name max_energy_range_uj \) -print 2>/dev/null || true'

# GPU
if run nvidia-smi; then
    capture nvidia_query nvidia-smi --query-gpu=name,driver_version,memory.total,power.default_limit,power.max_limit --format=csv,noheader
    capture nvidia_banner nvidia-smi
fi

# NPU
capture lsmod lsmod
capture accel ls -la /dev/accel
capture drm ls -la /dev/dri
if run xrt-smi; then
    capture xrt xrt-smi examine
fi

capture session bash -lc 'loginctl show-session "$XDG_SESSION_ID" -p Type -p Desktop -p Name'

SESSION_FILE="${TMP_DIR}/desktop_session.txt"
{
    if [[ -f "${TMP_DIR}/plasma.txt" ]]; then
        cat "${TMP_DIR}/plasma.txt"
    else
        echo "plasmashell not found"
    fi
    cat "${TMP_DIR}/session.txt"
} >"${SESSION_FILE}"

POWERCAP_FILE="${TMP_DIR}/powercap_with_paths.txt"
{
    cat "${TMP_DIR}/powercap.txt"
    echo
    echo "-- energy paths --"
    cat "${TMP_DIR}/energy_paths.txt"
} >"${POWERCAP_FILE}"

NPU_DRIVER_FILE="${TMP_DIR}/npu_driver_nodes.txt"
{
    grep -E '^amdxdna' "${TMP_DIR}/lsmod.txt" || echo "amdxdna not loaded"
    echo
    echo "/dev/accel"
    cat "${TMP_DIR}/accel.txt"
    echo
    echo "/dev/dri"
    cat "${TMP_DIR}/drm.txt"
} >"${NPU_DRIVER_FILE}"

NPU_EVIDENCE_FILE="${TMP_DIR}/npu_memory_evidence.txt"
{
    if [[ -f "${TMP_DIR}/xrt.txt" ]]; then
        echo "xrt-smi memory-related lines:"
        grep -Ei 'memory|hbm|ddr|vram' "${TMP_DIR}/xrt.txt" || echo "no memory-specific lines found in xrt-smi output"
    else
        echo "xrt-smi not found"
    fi
    echo

    npu_bdf=""
    if [[ -f "${TMP_DIR}/xrt.txt" ]]; then
        npu_bdf="$(grep -Eo '\[[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]\]' "${TMP_DIR}/xrt.txt" | head -n 1 | tr -d '[]')"
    fi

    if run lspci && [[ -n "${npu_bdf}" ]]; then
        echo "lspci BAR regions for NPU (${npu_bdf}):"
        lspci -s "${npu_bdf#0000:}" -vv 2>/dev/null | grep -E '^[[:space:]]*Region [0-9]+:' || echo "no BAR region lines found"
    elif run lspci; then
        echo "could not infer NPU BDF from xrt-smi output"
    else
        echo "lspci not found"
    fi
} >"${NPU_EVIDENCE_FILE}"

{
    HOST_NAME="unknown"
    if [[ -r /proc/sys/kernel/hostname ]]; then
        HOST_NAME="$(cat /proc/sys/kernel/hostname)"
    fi

    echo "# Machine Specs Validation Report"
    echo
    echo "Generated: $(date -Iseconds)"
    echo "Host: ${HOST_NAME}"
    echo

    # System
    section "System"
    block "hostnamectl" "${TMP_DIR}/hostnamectl.txt" "hostnamectl output unavailable"
    block "Kernel" "${TMP_DIR}/uname.txt" "kernel output unavailable"
    block "OS Release" "${TMP_DIR}/os_release.txt" "os-release output unavailable"
    block "Desktop Session" "${SESSION_FILE}" "desktop session output unavailable"

    # Memory
    section "Memory"
    block "free -h" "${TMP_DIR}/free.txt" "free output unavailable"
    block "Memory Channel Topology (dmidecode)" "${TMP_DIR}/dmidecode_memory.txt" "dmidecode not found"

    # Storage
    section "Storage"
    block "lsblk" "${TMP_DIR}/lsblk.txt" "lsblk output unavailable"

    # CPU
    section "CPU"
    block "lscpu" "${TMP_DIR}/lscpu.txt" "lscpu output unavailable"
    block "Powercap" "${POWERCAP_FILE}" "powercap output unavailable"

    # GPU
    section "GPU"
    block "nvidia-smi query" "${TMP_DIR}/nvidia_query.txt" "nvidia-smi not found"
    if [[ -f "${TMP_DIR}/nvidia_banner.txt" ]]; then
        NVIDIA_SUMMARY_FILE="${TMP_DIR}/nvidia_summary.txt"
        head -n 20 "${TMP_DIR}/nvidia_banner.txt" >"${NVIDIA_SUMMARY_FILE}"
        block "nvidia-smi summary" "${NVIDIA_SUMMARY_FILE}" "nvidia-smi summary unavailable"
    fi

    # NPU
    section "NPU"
    block "Driver and device nodes" "${NPU_DRIVER_FILE}" "npu driver/device output unavailable"
    block "xrt-smi examine" "${TMP_DIR}/xrt.txt" "xrt-smi not found"
    block "NPU memory model evidence" "${NPU_EVIDENCE_FILE}" "npu memory evidence unavailable"

    MEMORY_CHANNEL_STATUS="unknown"
    if [[ -f "${TMP_DIR}/dmidecode_memory.txt" ]] && grep -q '^Memory Device$' "${TMP_DIR}/dmidecode_memory.txt"; then
        read -r POPULATED_MODULE_COUNT DISTINCT_CHANNEL_COUNT < <(awk '
            function flush() {
                if (in_dev && size != "" && size !~ /No Module Installed/) {
                    pop++
                    if (chan != "") seen[toupper(chan)] = 1
                }
                in_dev = 0; size = ""; chan = ""
            }
            /^Memory Device$/ { flush(); in_dev = 1; next }
            in_dev && /^[[:space:]]*Size:/ { v = $0; sub(/^[^:]*:[[:space:]]*/, "", v); size = v }
            in_dev && /^[[:space:]]*(Bank )?Locator:/ {
                if (match($0, /[Cc][Hh][Aa][Nn][Nn][Ee][Ll][[:space:]]*[A-Za-z0-9]+/))
                    chan = substr($0, RSTART, RLENGTH)
            }
            /^$/ { flush() }
            END { flush(); n = 0; for (c in seen) n++; print pop + 0, n + 0 }
        ' "${TMP_DIR}/dmidecode_memory.txt")

        if [[ "${POPULATED_MODULE_COUNT}" -ge 2 && "${DISTINCT_CHANNEL_COUNT}" -ge 2 ]]; then
            MEMORY_CHANNEL_STATUS="dual-channel (${POPULATED_MODULE_COUNT} modules across ${DISTINCT_CHANNEL_COUNT} channels)"
        elif [[ "${POPULATED_MODULE_COUNT}" -eq 1 ]]; then
            MEMORY_CHANNEL_STATUS="single-channel (1 module populated)"
        elif [[ "${POPULATED_MODULE_COUNT}" -ge 2 ]]; then
            MEMORY_CHANNEL_STATUS="${POPULATED_MODULE_COUNT} modules populated; channel labels inconclusive"
        fi
    fi

    NPU_SHARED_MEMORY_STATUS="unknown"
    if [[ -f "${TMP_DIR}/xrt.txt" ]] && grep -q 'Memory\s*:\s*[0-9]' "${TMP_DIR}/xrt.txt"; then
        if grep -Eqi 'HBM|VRAM|GDDR' "${TMP_DIR}/xrt.txt"; then
            NPU_SHARED_MEMORY_STATUS="no"
        else
            NPU_SHARED_MEMORY_STATUS="likely yes"
        fi
    fi

    section "Quick Validation Checklist"
    echo "- CPU info present: $(grep -q 'Model name' "${TMP_DIR}/lscpu.txt" && echo yes || echo no)"
    echo "- GPU query available: $( [[ -f "${TMP_DIR}/nvidia_query.txt" ]] && ! grep -q '^\[command failed\]' "${TMP_DIR}/nvidia_query.txt" && echo yes || echo no )"
    echo "- NPU runtime query available: $( [[ -f "${TMP_DIR}/xrt.txt" ]] && grep -q 'Device(s) Present' "${TMP_DIR}/xrt.txt" && echo yes || echo no )"
    echo "- energy_uj path present: $(grep -q 'energy_uj' "${TMP_DIR}/energy_paths.txt" && echo yes || echo no)"
    echo "- Installed RAM (MemTotal): $(awk '/^MemTotal:/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
    echo "- Memory channel configuration (dmidecode): ${MEMORY_CHANNEL_STATUS}"
    echo "- NPU shared-system-memory indication: ${NPU_SHARED_MEMORY_STATUS}"
} >"${OUT_FILE}"

echo "Validation report written to: ${OUT_FILE}"
