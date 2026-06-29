#!/usr/bin/env bash
set -uo pipefail

GPU_POWER_W="${1:-${GPU_POWER_W:-}}"
ACTION="${ACTION:-set}"

set_governor() {
    local gov="$1"
    if command -v cpupower >/dev/null 2>&1; then
        sudo cpupower frequency-set -g "$gov" >/dev/null 2>&1 && echo "CPU governor   -> ${gov}" && return
    fi
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "$gov" | sudo tee "$f" >/dev/null 2>&1
    done
    echo "CPU governor   -> ${gov} (sysfs)"
}

set_boost() {
    local on="$1"
    if [[ -e /sys/devices/system/cpu/cpufreq/boost ]]; then
        echo "$on" | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null 2>&1 \
            && echo "CPU boost      -> ${on}"
    fi
}

report() {
    echo "== current state =="
    echo "CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
    echo "CPU boost   : $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=persistence_mode,power.limit,power.default_limit,power.max_limit,clocks.max.sm \
            --format=csv,noheader 2>/dev/null | sed 's/^/GPU persist,limit,default,max,maxclk: /'
    fi
}

case "${ACTION}" in
    show)
        report
        exit 0
        ;;
    restore)
        set_governor schedutil || set_governor ondemand
        set_boost 1
        command -v nvidia-smi >/dev/null 2>&1 && sudo nvidia-smi -pm 0 >/dev/null 2>&1 && echo "GPU persistence -> off"
        echo
        report
        exit 0
        ;;
esac

set_governor performance
set_boost 0
if command -v nvidia-smi >/dev/null 2>&1; then
    sudo nvidia-smi -pm 1 >/dev/null 2>&1 && echo "GPU persistence -> on"
    if [[ -n "${GPU_POWER_W}" ]]; then
        sudo nvidia-smi -pl "${GPU_POWER_W}" >/dev/null 2>&1 && echo "GPU power cap  -> ${GPU_POWER_W} W"
    fi
fi
echo
report
