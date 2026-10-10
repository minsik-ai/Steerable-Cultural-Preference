#!/usr/bin/env bash
#SBATCH --job-name=scpo-tulu-compare
#SBATCH --array=0-41%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
#SBATCH --no-requeue
set -euo pipefail
SOURCE="${1:?source snapshot}"
PY="${2:?python executable}"
OUT="${3:?comparison root}"
BASELINE="${4:?completed FP32 baseline}"
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -c 'import torch; assert torch.cuda.is_available()'
"$PY" "$SOURCE/scripts/tulu_method_comparison.py" run --source "$SOURCE" --root "$OUT" \
    --baseline "$BASELINE" --index "$SLURM_ARRAY_TASK_ID"
