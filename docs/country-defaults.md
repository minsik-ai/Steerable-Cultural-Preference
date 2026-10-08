# Recorded country defaults

Machine-readable source: `configs/country_defaults.json`. The default launcher reads this file; no per-country flags are needed. All entries currently use shuffle seed 0, batch size 8, and 1024 steps. Original model/split seeds remain 0 and Trainer seed remains its original implicit 42.

| Country | LR | Tau | Beta | Overall | Misranked | Status |
|---|---:|---:|---:|---:|---:|---|
| Australia | 1e-06 | 0.5 | 1 | 59.37% | 33.13% | provisional |
| Canada | 1e-06 | 0.7 | 1.1 | 66.18% | 27.74% | matched |
| Chile | 1e-06 | 0.5 | 1.1 | 60.61% | 37.62% | matched |
| Israel | 1e-06 | 0.5 | 1 | 65.14% | 38.10% | provisional |
| Mexico | 1e-05 | 0.7 | 1.1 | 64.75% | 54.24% | matched |
| New Zealand | 1e-06 | 0.5 | 1.1 | 62.40% | 44.52% | matched |
| South Africa | 1e-06 | 0.7 | 1.1 | 65.48% | 27.72% | matched |

These observed results come from retained, validated checkpoints, whose exact relative paths and both metrics are stored in the JSON. They are per-country fixed-test selections, not a common-configuration three-seed aggregate. Australia and Israel are best-so-far provisional defaults, not claims of meeting the paper.

## Reproduce

```bash
uv venv --python 3.11.16 .venv
uv pip sync --python .venv/bin/python requirements-pw-lock.txt
.venv/bin/python scripts/run_country_defaults.py --output artifacts/defaults-01
```

Use `--countries Chile Mexico` to run a subset. Use `--prepare-only` to inspect the generated original-code copies without training. The actual run checks the recorded Python/package versions and the current model/data Hub revisions; a mismatch stops rather than silently changing inputs. This is a drift guard, not a claim that historical inputs were pinned by the old code. On a Hub revision change, recover the recorded snapshot explicitly before proceeding. Recorded revision hashes are in the JSON.

On PW:

```bash
sbatch --output=artifacts/logs/defaults_%j.out scripts/defaults_pw.sbatch \
  "$PWD" "$PWD/artifacts/defaults-01"
```

Each country gets its generated source/diff, actual configuration, environment, checkpoint, both reports, and validation. The launcher does not depend on old checkpoint directories for training; those paths in JSON are provenance only. It uses the settings even if a rerun scores below the matching threshold, and reports actual scores. It never searches for alternate settings during a default run.

## Australia and Israel continuation

Frozen plan: `configs/remaining_search.json`. Each country starts from its best `tau=0.5, beta=1.0, LR=1e-6` run. Try shuffle seeds 1 and 2, beta 0.9/0.8/0.67, tau 0.4/0.6, and LR 5e-7, changing one factor at a time. There are eight new, nonduplicate candidates per country, at most two GPUs, with the same stopping thresholds (Australia 59.70%, Israel 67.30%). Do not rerun the other five countries. Keep all trial outcomes, including misses. New successful settings require explicit promotion to the versioned defaults file; Slurm workers never mutate the Git checkout.

The scheduled monitor remains paused as requested. Slurm handles all trial progression and saves results independently.

The generic first-search Slurm script defaults to five workers. For the remaining
plan, explicitly submit `--array=0-1%2` and pass the plan as its fifth argument;
this maps task 0 to Australia and task 1 to Israel. Example (source snapshot
`SOURCE`, runtime `PY`, new output `OUT`, checkout artifact root `REPO`):

```bash
sbatch --array=0-1%2 --qos=yejinchoi \
  --output="$REPO/artifacts/logs/remaining_%A_%a.out" \
  "$SOURCE/scripts/country_search_pw.sbatch" \
  "$SOURCE" "$PY" "$OUT" "$REPO" "$SOURCE/configs/remaining_search.json"
```
