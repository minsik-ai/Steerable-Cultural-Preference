#!/usr/bin/env bash
#SBATCH --job-name=scpo-tulu-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
set -euo pipefail
SOURCE="${1:?source snapshot}"
PY="${2:?python executable}"
OUT="${3:?comparison root}"
"$PY" "$SOURCE/scripts/tulu_method_comparison.py" summarize --source "$SOURCE" --root "$OUT"
