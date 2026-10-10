#!/usr/bin/env bash
#SBATCH --job-name=scpo-tulu-training-checks
#SBATCH --array=0-2%1
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
OUT="${3:?diagnostics root}"
export HF_HOME="${HF_HOME:-/home/minsik/hf}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -c 'import torch; assert torch.cuda.is_available()'
"$PY" - "$SOURCE" "$OUT" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys,subprocess
from pathlib import Path
source,out=map(Path,sys.argv[1:3]);index=int(sys.argv[3])
plan=json.loads((source/'configs/tulu_training_checks.json').read_text())
case=plan['cases'][index]
config=json.loads((source/plan['baseline_config']).read_text())
config.update(learning_rate=case['learning_rate'],max_steps=case['max_steps'])
path=out/f'case-{index}.json'
with path.open('x') as stream: json.dump(config,stream,indent=2)
# Hold the original pilot format fixed to isolate precision/LR/steps.
fmt='chat-template'
subprocess.run([sys.executable,str(source/'scripts/tulu_replication.py'),'train','--config',str(path),
 '--output',str(out/'training'/case['name']),'--baseline',str(out/'fp32-baseline'),
 '--country',plan['country'],'--format',fmt],check=True)
PY
