#!/usr/bin/env bash
set -uo pipefail

GPU_POWER_W="${1:-${GPU_POWER_W:-}}"
ACTION="${ACTION:-set}"

set_governor() {
    local gov="$1"
    if command -v cpupower >/dev/null 2>&1; then
        if sudo cpupower frequency-set -g "$gov" >/dev/null 2>&1; then
            echo "CPU governor   -> ${gov}"
            return 0
        fi
    fi
    local wrote=0
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -e "$f" ]] || continue
        echo "$gov" | sudo tee "$f" >/dev/null 2>&1 && wrote=1
    done
    if [[ "$wrote" -eq 1 ]]; then
        echo "CPU governor   -> ${gov} (sysfs)"
        return 0
    fi
    echo "CPU governor   -> FAILED to set ${gov}" >&2
    return 1
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
        if command -v nvidia-smi >/dev/null 2>&1; then
            default_w="$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
            if [[ "${default_w}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                sudo nvidia-smi -pl "${default_w}" >/dev/null 2>&1 && echo "GPU power cap  -> ${default_w} W (default)"
            fi
            sudo nvidia-smi -pm 0 >/dev/null 2>&1 && echo "GPU persistence -> off"
        fi
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
