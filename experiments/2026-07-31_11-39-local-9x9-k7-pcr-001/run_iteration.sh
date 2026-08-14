#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${EXP_DIR}/launcher.sh" --once "$@"
