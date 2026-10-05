#!/usr/bin/env bash
# Source before importing LIBERO/MuJoCo. Configuration and caches stay in scratch.
DURA_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DURA_PROJECT/scripts/runtime.sh"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$DURA_SCRATCH/libero_config"
export NUMBA_CACHE_DIR="$DURA_SCRATCH/cache/numba"
export MPLCONFIGDIR="$DURA_SCRATCH/cache/matplotlib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export DURA_LIBERO_ROOT="${DURA_LIBERO_ROOT:-$DURA_SCRATCH/external/LIBERO}"
export DURA_DATASETS_ROOT="${DURA_DATASETS_ROOT:-$DURA_SCRATCH/datasets}"
export DURA_PYTHON="${DURA_PYTHON:-$DURA_SCRATCH/venv/bin/python}"
mkdir -p "$LIBERO_CONFIG_PATH" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
"$DURA_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml
root = (Path(os.environ['DURA_LIBERO_ROOT']) / 'libero/libero').resolve()
if not (root / 'bddl_files').is_dir():
    raise FileNotFoundError(f'Set DURA_LIBERO_ROOT to your LIBERO checkout: {root}')
config = {'benchmark_root': str(root), 'bddl_files': str(root/'bddl_files'),
          'init_states': str(root/'init_files'), 'assets': str(root/'assets'),
          'datasets': os.environ['DURA_DATASETS_ROOT']}
path = Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml'
if path.exists() and yaml.safe_load(path.read_text()) != config:
    raise ValueError(f'Existing scratch LIBERO configuration differs: {path}; use a separate DURA_SCRATCH')
if not path.exists():
    path.write_text(yaml.safe_dump(config))
PY
