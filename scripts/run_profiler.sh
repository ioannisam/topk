#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${ROOT_DIR}/test/prof/.venv/bin/python"
PROFILER="${ROOT_DIR}/test/prof/profiler.py"
BOOTSTRAP="${ROOT_DIR}/scripts/bootstrap_env.sh"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "Profiler venv not found. Bootstrapping..."
    "${BOOTSTRAP}"
fi

if ! "${VENV_PY}" -c "import matplotlib" >/dev/null 2>&1; then
    echo "matplotlib missing in profiler venv. Bootstrapping..."
    "${BOOTSTRAP}"
fi

exec "${VENV_PY}" "${PROFILER}" "$@"
