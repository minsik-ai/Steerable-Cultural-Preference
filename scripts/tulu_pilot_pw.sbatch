#!/usr/bin/env bash
#SBATCH --job-name=scpo-tulu-pilot
#SBATCH --array=0-6%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
#SBATCH --no-requeue
set -euo pipefail
SOURCE="${1:?immutable source snapshot}"
PY="${2:?python executable}"
OUT="${3:?artifact root}"
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
COUNTRIES=(Chile 'South Africa' 'New Zealand' Australia Mexico Israel Canada)
COUNTRY="${COUNTRIES[$SLURM_ARRAY_TASK_ID]}"
FORMAT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["input_format"])' "$OUT/format-selection.json")
"$PY" -c 'import torch; assert torch.cuda.is_available()'
"$PY" "$SOURCE/scripts/tulu_replication.py" train --config "$SOURCE/configs/tulu_pilot.json" \
  --baseline "$OUT/baseline" --country "$COUNTRY" --format "$FORMAT" --output "$OUT/pilots/$SLURM_ARRAY_TASK_ID"
