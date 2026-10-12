#!/usr/bin/env bash
#SBATCH --job-name=scpo-tulu-chat-smoke
#SBATCH --nodes=1
#SBATCH --exclude=g0375,g0376,g0377,g0378,g0380,g0381
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
#SBATCH --no-requeue
set -euo pipefail
SOURCE="${1:?source snapshot}"
PY="${2:?python executable}"
OUT="${3:?artifact root}"
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_properties(0).total_memory > 100*1024**3, "Original Tulu protocol requires a large-memory GPU; do not change precision/checkpointing"'
"$PY" "$SOURCE/scripts/check_original_runtime.py" --lock "$SOURCE/requirements-pw-lock.txt" \
  --configs "$SOURCE/configs/tulu_chat_smoke.json" \
  --output "$OUT/smoke-runtime.json"
"$PY" "$SOURCE/scripts/tulu_replication.py" --config "$SOURCE/configs/tulu_chat_smoke.json" --country Chile --output "$OUT/smoke-tulu"
