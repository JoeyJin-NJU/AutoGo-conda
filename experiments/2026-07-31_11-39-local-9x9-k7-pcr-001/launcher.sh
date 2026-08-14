#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
EXP_NAME="$(basename "${EXP_DIR}")"
AUTOGO_PYTHON="${AUTOGO_PYTHON:-$(command -v python)}"

if [[ -z "${AUTOGO_PYTHON}" || ! -x "${AUTOGO_PYTHON}" ]]; then
    echo "Python executable not found. Activate the Conda environment first." >&2
    exit 1
fi

export AUTOGO_PYTHON
export GAME_DATA_DIR="${REPO_ROOT}/local_data/game_data"
export AUTOGO_CHECKPOINT_DIR="${REPO_ROOT}/local_data/checkpoints/${EXP_NAME}"
export PYTHONUNBUFFERED=1

exec "${AUTOGO_PYTHON}" "${EXP_DIR}/controller.py" "$@"
