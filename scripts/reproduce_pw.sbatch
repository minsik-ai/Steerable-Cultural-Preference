#!/usr/bin/env bash
#SBATCH --job-name=scpo-reproduce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=voltagepark
#SBATCH --qos=normal
#SBATCH --account=yejinchoi
set -euo pipefail
REPO="${1:?absolute checkout directory}"
OUT="${2:?new output directory}"
test "$#" -eq 2 || { echo "Original experiment accepts only REPO and OUT" >&2; exit 2; }
cd "$REPO"
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
PY="$REPO/.venv/bin/python"
test ! -e "$OUT" || { echo "Output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_PATCH=$(git diff HEAD --binary)
for source in preferences.py our_reward_trainer.py our_inverse_reward_trainer.py our_data_collator.py; do
    git show "94f8430054835aa919b226b4344b8f0e8424f29b:$source" | cmp - "$source"
done
status=0
"$PY" preferences.py "$OUT" || status=$?
"$PY" -c 'import importlib.metadata as m; print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in m.distributions())))' > "$OUT/environment.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$OUT/source-commit.txt"
printf '%s\n' "$SOURCE_PATCH" > "$OUT/source-changes.patch"
test "$status" -eq 0 || exit "$status"
for checkpoint in "$OUT"/trunc/*/ours_*epoch; do
    "$PY" scripts/validate_checkpoint.py "$checkpoint" --tokenizer OpenAssistant/reward-model-deberta-v3-base --output "$checkpoint/validation.json"
done
