# Australia ablations alongside the existing South Africa search

> Historical reconstructed-protocol record. Superseded for primary replication by [the original-source rollback](original-protocol-rollback.md). Existing results remain diagnostic evidence.

User requested a few Australia ablations as well. South Africa job 57313 already
runs its six-setting plan from its own frozen snapshot and is not resubmitted.
This new job covers only Australia; both searches keep LR fixed at 1e-6 and use
one GPU each. The original training sources, fixed split0, shuffle0, FP32,
chat-template and provisional LoRA configuration remain unchanged.

Australia references:
- Starting model: 66.4234%.
- Full-data fine-tuning (1024 steps, LR1e-6): 65.4501%.
- Existing filtering+weighting: 65.6934%; weighting-only ties the starting model.

| Candidate | Method | Steps | Tau | Beta |
|---|---|---:|---:|---:|
| filtered-128 | Filtering + weighting | 128 | 0.5 | 1 |
| filtered-256 | Filtering + weighting | 256 | 0.5 | 1 |
| filter-07-1024 | Filtering + weighting | 1024 | 0.7 | 1 |
| weighted-beta2 | Weighting-only | 1024 | inactive | 2 |

Stop at the first validated candidate strictly exceeding the starting model,
established full-data reference, and any additional equal-budget full-data
controls. A promising short run triggers its same-step full-data control before
acceptance. Four candidates plus at most two controls; every result is retained.
Results remain single-seed exploratory comparisons, including negative outcomes.

Config: `configs/tulu_australia.json`. The existing focused-search runner now
accepts an explicit `--plan` and permits only South Africa or Australia. Its
South Africa default remains backward compatible. The Slurm launcher takes the
plan path as an optional fifth argument. Source/output snapshot:
`artifacts/tulu-australia-05`, with per-trial progress in `search/status.json`.
Use `yejinchoi` QoS; no scheduled desktop monitor is started.

Validation: five targeted tests plus original-source fidelity passed locally;
static Python checks and Slurm syntax passed. The existing South Africa
acceptance-control test remains unchanged and passing.

The filter/weight ablations keep 1024 steps to isolate their effect against
existing 1024-step runs; only the first two candidates change the step budget.
