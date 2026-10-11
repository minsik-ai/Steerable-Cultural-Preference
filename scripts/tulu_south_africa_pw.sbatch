#!/usr/bin/env bash
#SBATCH --job-name=scpo-tulu-south-africa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
#SBATCH --no-requeue
set -euo pipefail
SOURCE="${1:?frozen source snapshot}"
PY="${2:?python executable}"
ARTIFACTS="${3:?repository artifact root}"
OUT="${4:?new search output directory}"
EXTRA=()
if [[ $# -eq 5 ]]; then EXTRA=(--plan "$5"); fi
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -c 'import torch; assert torch.cuda.is_available()'
"$PY" "$SOURCE/scripts/tulu_south_africa.py" --source "$SOURCE" --artifacts "$ARTIFACTS" --output "$OUT" "${EXTRA[@]}"
