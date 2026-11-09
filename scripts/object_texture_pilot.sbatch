#!/usr/bin/env bash
#SBATCH --job-name=dura-object-pilot
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=a100:1
#SBATCH --constraint=gpu_a100_40gb
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@120
#SBATCH --export=NONE

set -euo pipefail
if [[ $# -ne 7 ]]; then
  printf 'Use scripts/submit_object_texture.py to freeze code, configure scratch logs, and submit.\n' >&2
  exit 2
fi
DURA_CODE_ROOT="$1"
DURA_PILOT_CONFIG="$2"
DURA_PILOT_STAGE="$3"
export DURA_LIBERO_ROOT="$4"
export DURA_DATASETS_ROOT="$5"
export DURA_SCRATCH="$6"
export HF_HOME="$7"
export DURA_PYTHON="$DURA_SCRATCH/venv/bin/python"
export PATH=/usr/bin:/bin
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
cd "$DURA_CODE_ROOT"
source scripts/eval_runtime.sh
printf 'Job %s, host %s, stage %s, CUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$DURA_PILOT_STAGE" "${CUDA_VISIBLE_DEVICES:-unset}"
# ALL here exports the scratch/cache settings created in this batch environment.
srun --export=ALL --ntasks=1 "$DURA_PYTHON" -u -m dura.object_texture_pilot \
  --config "$DURA_PILOT_CONFIG" "$DURA_PILOT_STAGE"
