#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
AUTOGO_PYTHON="${AUTOGO_PYTHON:-$(command -v python)}"

if [[ -z "${AUTOGO_PYTHON}" || ! -x "${AUTOGO_PYTHON}" ]]; then
    echo "Python executable not found. Activate the Conda environment first." >&2
    exit 1
fi
"${AUTOGO_PYTHON}" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); p.touch(); print(f"graceful stop requested: {p}")' "${EXP_DIR}/runtime/stop.requested"
