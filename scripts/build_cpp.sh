#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/src/alpha_go/cpp"
BUILD_DIR="${AUTOGO_CPP_BUILD_DIR:-${SOURCE_DIR}/build}"
PYTHON_BIN="${AUTOGO_PYTHON:-$(command -v python)}"
FETCH_DIR="${AUTOGO_CMAKE_FETCH_DIR:-${TMPDIR:-/tmp}/autogo-conda-cmake-fetchcontent}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found. Activate the Conda environment first." >&2
    exit 1
fi

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="${PYTHON_BIN}" \
    -DFETCHCONTENT_BASE_DIR="${FETCH_DIR}"
cmake --build "${BUILD_DIR}" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cmake --install "${BUILD_DIR}" --prefix "${SITE_PACKAGES}"

"${PYTHON_BIN}" -c 'import alpha_go_cpp; print("alpha_go_cpp import OK")'
echo "C++ extension installed into ${SITE_PACKAGES}"
