#!/usr/bin/env bash
#SBATCH --job-name=scpo-original-protocol-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=voltagepark
#SBATCH --qos=yejinchoi
#SBATCH --account=yejinchoi
set -euo pipefail
SOURCE="${1:?source snapshot}"
PY="${2:?python executable}"
OUT="${3:?artifact root}"
"$PY" "$SOURCE/scripts/summarize_original_protocol.py" --root "$OUT"
