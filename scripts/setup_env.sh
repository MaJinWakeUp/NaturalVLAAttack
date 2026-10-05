#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--evaluation" ) ]]; then
  printf 'Usage: bash scripts/setup_env.sh [--evaluation]\n' >&2
  exit 2
fi
DURA_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DURA_PROJECT/scripts/runtime.sh"
DURA_ENV="${DURA_ENV:-$DURA_SCRATCH/venv}"
if [[ ! -x "$DURA_ENV/bin/python" ]]; then
  "${DURA_BASE_PYTHON:-python3.10}" -m venv "$DURA_ENV"
fi
"$DURA_ENV/bin/python" -m pip install --upgrade 'pip==25.3'
"$DURA_ENV/bin/python" -m pip install 'numpy==1.26.4'
"$DURA_ENV/bin/python" -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121
"$DURA_ENV/bin/python" -m pip install -r "$DURA_PROJECT/requirements-training.txt"
if [[ "${1:-}" == "--evaluation" ]]; then
  "$DURA_ENV/bin/python" -m pip install -r "$DURA_PROJECT/requirements-evaluation.txt"
fi
"$DURA_ENV/bin/python" -m pip check
# Run from the checkout; no editable-install metadata or model artifacts in the repo.
