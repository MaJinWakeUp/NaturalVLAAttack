#!/usr/bin/env bash
set -euo pipefail
DURA_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DURA_PROJECT/scripts/runtime.sh"
cd "$DURA_PROJECT"
DURA_PYTHON="${DURA_PYTHON:-$DURA_SCRATCH/venv/bin/python}"
"$DURA_PYTHON" -u -m dura.natural_train --config "${1:-configs/libero_relaxed.yaml}"
