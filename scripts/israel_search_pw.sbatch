#!/usr/bin/env bash
#SBATCH --job-name=scpo-israel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
#SBATCH --no-requeue
set -euo pipefail
SOURCE="${1:?immutable source snapshot}"
PY="${2:?python executable}"
OUT="${3:?new search output root}"
REPO="${4:?artifact checkout root}"
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
"$PY" -c 'import torch; assert torch.cuda.is_available()'
"$PY" "$SOURCE/scripts/diagnose_israel_scoring.py" --repo "$REPO" \
    --settings "$SOURCE/configs/country_defaults.json" --output "$OUT/scoring-diagnostic.json"
"$PY" "$SOURCE/scripts/country_search.py" --repo "$SOURCE" --root "$OUT" \
    --baseline "$REPO" --country Israel --plan "$SOURCE/configs/israel_search.json"
