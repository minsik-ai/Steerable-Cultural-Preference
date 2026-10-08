#!/usr/bin/env bash
#SBATCH --job-name=scpo-country-search
#SBATCH --array=0-4%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
#SBATCH --no-requeue
set -euo pipefail
SOURCE="${1:?immutable source snapshot}"
PY="${2:?absolute python executable}"
OUT="${3:?search output root}"
BASELINE="${4:?original completed output directory}"
EXTRA=()
if [[ $# -eq 5 ]]; then EXTRA=(--plan "$5"); fi
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PY" -c 'import torch; assert torch.cuda.is_available()'
"$PY" "$SOURCE/scripts/country_search.py" --repo "$SOURCE" --root "$OUT" \
    --baseline "$BASELINE" --index "$SLURM_ARRAY_TASK_ID" "${EXTRA[@]}"
