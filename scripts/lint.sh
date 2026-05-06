#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="check"

usage() {
    cat <<'EOF'
Usage: scripts/lint.sh [--fix]

Checks formatting and linting for the entire repository.
- C/C++: Uses clang-format
- Python: Uses ruff

Options:
  --fix    Apply formatting and auto-fixable changes in place
  -h       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)
            MODE="fix"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

# Check for required tools
if ! command -v clang-format >/dev/null 2>&1; then
    echo "Error: clang-format is not installed or not in PATH." >&2
    exit 127
fi

if ! command -v ruff >/dev/null 2>&1; then
    echo "Error: ruff is not installed or not in PATH." >&2
    exit 127
fi

# ==========================================
# C/C++ LINTING PREP
# ==========================================
collect_cpp_files() {
    if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        (
            cd "${ROOT_DIR}"
            git ls-files --cached --others --exclude-standard -- '*.cpp' '*.cu' '*.h' '*.hpp'
        )
    else
        (
            cd "${ROOT_DIR}"
            find . \
                -type d \( -name .git -o -name build -o -name build-fast -o -name CMakeFiles -o -name npu_env \) -prune -o \
                -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.hpp' \) -print | sed 's#^\./##'
        )
    fi
}

mapfile -t CANDIDATE_FILES < <(collect_cpp_files | sed '/^$/d' | sort -u)

CPP_FILES=()
for file in "${CANDIDATE_FILES[@]}"; do
    if [[ -f "${ROOT_DIR}/${file}" ]]; then
        CPP_FILES+=("${file}")
    fi
done

cd "${ROOT_DIR}"
GLOBAL_FAILED=0

# ==========================================
# FIX MODE
# ==========================================
if [[ "${MODE}" == "fix" ]]; then
    if [[ ${#CPP_FILES[@]} -gt 0 ]]; then
        echo "Formatting ${#CPP_FILES[@]} C/C++ file(s) with clang-format..."
        clang-format -i "${CPP_FILES[@]}"
    else
        echo "No C/C++ source files found to format."
    fi

    echo "Formatting and fixing Python files with ruff..."
    ruff format .
    ruff check --fix .
    
    echo "All auto-fixes applied."
    exit 0
fi

# ==========================================
# CHECK MODE
# ==========================================
echo "Running lint checks..."

# 1. C/C++ Checks
if [[ ${#CPP_FILES[@]} -gt 0 ]]; then
    FAILED_CPP=()
    for file in "${CPP_FILES[@]}"; do
        if ! clang-format --dry-run --Werror "${file}" >/dev/null 2>&1; then
            FAILED_CPP+=("${file}")
        fi
    done

    if [[ ${#FAILED_CPP[@]} -gt 0 ]]; then
        echo -e "\n[C/C++] Formatting issues found in ${#FAILED_CPP[@]} file(s):"
        for file in "${FAILED_CPP[@]}"; do
            echo "  - ${file}"
        done
        GLOBAL_FAILED=1
    else
        echo "[C/C++] clang-format checks passed."
    fi
fi

# 2. Python Checks
echo "[Python] Running ruff format check..."
if ! ruff format --check . ; then
    GLOBAL_FAILED=1
fi

echo "[Python] Running ruff lint check..."
if ! ruff check . ; then
    GLOBAL_FAILED=1
fi

# ==========================================
# FINAL REPORT
# ==========================================
if [[ ${GLOBAL_FAILED} -ne 0 ]]; then
    echo
    echo "Linting failed... Run 'scripts/lint.sh --fix' to auto-format and fix issues."
    exit 1
fi

echo "All lint checks passed!"
