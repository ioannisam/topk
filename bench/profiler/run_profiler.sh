#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/find_root.sh"
ROOT_DIR="$(find_root_dir "${SCRIPT_DIR}")" || exit 1

VENV_PY="${ROOT_DIR}/bench/profiler/.venv/bin/python"
PROFILER="${ROOT_DIR}/bench/profiler/profiler.py"
BOOTSTRAP="${ROOT_DIR}/bench/profiler/bootstrap_env.sh"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "Profiler venv not found. Bootstrapping..."
    "${BOOTSTRAP}"
fi

if ! "${VENV_PY}" -c "import matplotlib" >/dev/null 2>&1; then
    echo "matplotlib missing in profiler venv. Bootstrapping..."
    "${BOOTSTRAP}"
fi

exec "${VENV_PY}" "${PROFILER}" "$@"
