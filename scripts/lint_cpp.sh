#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="check"

usage() {
    cat <<'EOF'
Usage: scripts/lint_cpp.sh [--fix]

Checks formatting with clang-format for all .cpp/.cu/.h/.hpp files in the repo.

Options:
  --fix    Apply formatting changes in place
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

if ! command -v clang-format >/dev/null 2>&1; then
    echo "Error: clang-format is not installed or not in PATH." >&2
    exit 127
fi

collect_files() {
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

mapfile -t CANDIDATE_FILES < <(collect_files | sed '/^$/d' | sort -u)

FILES=()
for file in "${CANDIDATE_FILES[@]}"; do
    if [[ -f "${ROOT_DIR}/${file}" ]]; then
        FILES+=("${file}")
    fi
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No C/C++ source files found."
    exit 0
fi

cd "${ROOT_DIR}"

if [[ "${MODE}" == "fix" ]]; then
    echo "Formatting ${#FILES[@]} file(s)..."
    clang-format -i "${FILES[@]}"
    echo "Done."
    exit 0
fi

echo "Linting ${#FILES[@]} file(s)..."
FAILED=()
for file in "${FILES[@]}"; do
    if ! clang-format --dry-run --Werror "${file}" >/dev/null 2>&1; then
        FAILED+=("${file}")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo
    echo "Formatting issues found in ${#FAILED[@]} file(s):"
    for file in "${FAILED[@]}"; do
        echo "  - ${file}"
    done
    echo
    echo "Run scripts/lint_cpp.sh --fix to auto-format."
    exit 1
fi

echo "All lint checks passed."
