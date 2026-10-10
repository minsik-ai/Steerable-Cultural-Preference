#!/usr/bin/env bash
#SBATCH --job-name=scpo-tulu-preflight
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
SPLIT="${4:?saved original survey split}"
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PY" -c 'import torch; assert torch.cuda.is_available()'
"$PY" "$SOURCE/scripts/tulu_replication.py" baseline --config "$SOURCE/configs/tulu_pilot.json" \
  --output "$OUT/baseline" --split "$SPLIT"
FORMAT=$("$PY" "$SOURCE/scripts/select_tulu_format.py" --report "$OUT/baseline/report.json" --output "$OUT/format-selection.json")
"$PY" "$SOURCE/scripts/tulu_replication.py" train --config "$SOURCE/configs/tulu_pilot.json" \
  --baseline "$OUT/baseline" --country Chile --format "$FORMAT" --max-steps 2 --output "$OUT/smoke-chile"
