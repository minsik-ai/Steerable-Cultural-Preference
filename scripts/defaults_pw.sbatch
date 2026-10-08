#!/usr/bin/env bash
#SBATCH --job-name=scpo-defaults
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
set -euo pipefail
REPO="${1:?absolute checkout directory}"
OUT="${2:?new output directory}"
shift 2
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
cd "$REPO"
"$REPO/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available()'
"$REPO/.venv/bin/python" scripts/run_country_defaults.py --output "$OUT" "$@"
