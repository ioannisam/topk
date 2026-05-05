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

VENV_PY="${ROOT_DIR}/test/prof/profiler/.venv/bin/python"
PROFILER="${ROOT_DIR}/test/prof/profiler/profiler.py"
BOOTSTRAP="${ROOT_DIR}/test/prof/profiler/bootstrap_env.sh"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "Profiler venv not found. Bootstrapping..."
    "${BOOTSTRAP}"
fi

if ! "${VENV_PY}" -c "import matplotlib" >/dev/null 2>&1; then
    echo "matplotlib missing in profiler venv. Bootstrapping..."
    "${BOOTSTRAP}"
fi

exec "${VENV_PY}" "${PROFILER}" "$@"
