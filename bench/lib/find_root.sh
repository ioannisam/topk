#!/usr/bin/env bash

find_root_dir() {
    local dir="$1"
    while [[ "${dir}" != "/" && ! -f "${dir}/CMakeLists.txt" ]]; do
        dir="$(dirname "${dir}")"
    done
    if [[ ! -f "${dir}/CMakeLists.txt" ]]; then
        echo "error: could not locate repo root (CMakeLists.txt)" >&2
        return 1
    fi
    echo "${dir}"
}
