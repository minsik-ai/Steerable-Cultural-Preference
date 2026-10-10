#!/usr/bin/env bash
#SBATCH --job-name=scpo-tulu-precision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
#SBATCH --no-requeue
set -euo pipefail
SOURCE="${1:?source snapshot}"
PY="${2:?python executable}"
OUT="${3:?diagnostics root}"
PREVIOUS="${4:?previous Tulu pilot root}"
SPLIT="${5:?original saved split}"
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -c 'import torch; assert torch.cuda.is_available()'
"$PY" "$SOURCE/scripts/tulu_precision_diagnostic.py" --config "$SOURCE/configs/tulu_pilot.json" \
  --baseline "$PREVIOUS/baseline" --pilot "$PREVIOUS/pilots/0" --output "$OUT/precision"
"$PY" "$SOURCE/scripts/tulu_replication.py" baseline --config "$SOURCE/configs/tulu_fp32.json" \
  --output "$OUT/fp32-baseline" --split "$SPLIT"
"$PY" "$SOURCE/scripts/select_tulu_format.py" --report "$OUT/fp32-baseline/report.json" --output "$OUT/format-selection.json"
