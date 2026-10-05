#!/usr/bin/env bash
# Source before importing HF, torch, or TensorFlow. All caches live in scratch.
export DURA_SCRATCH="${DURA_SCRATCH:-/scratch/jin7/dura}"
export HF_HOME="${HF_HOME:-/scratch/jin7/huggingface_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export XDG_CACHE_HOME="$DURA_SCRATCH/cache"
export TORCH_HOME="$DURA_SCRATCH/cache/torch"
export TMPDIR="$DURA_SCRATCH/tmp"
export PIP_CACHE_DIR="$DURA_SCRATCH/pip_cache"
export PYTHONPYCACHEPREFIX="$DURA_SCRATCH/pycache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_DISABLE_TELEMETRY=1
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" \
  "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$DURA_SCRATCH/logs" "$DURA_SCRATCH/runs"
