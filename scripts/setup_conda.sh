#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${AUTOGO_ENV_NAME:-autogo-conda}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
conda env create --name "${ENV_NAME}" --file environment.yml
ENV_PREFIX="$(conda run --name "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
"${ENV_PREFIX}/bin/python" -m pip install --editable '.[dev]'
AUTOGO_PYTHON="${ENV_PREFIX}/bin/python" bash scripts/build_cpp.sh
"${ENV_PREFIX}/bin/python" scripts/verify_install.py

echo "Setup complete. Activate with: conda activate ${ENV_NAME}"
