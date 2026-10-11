# Tulu South Africa focus, fixed learning rate 1e-6

The user asked to focus on South Africa and leave other countries alone. This
search runs only South Africa, in FP32 with the same chat template, LoRA capacity,
split0, shuffle0 and Trainer seed42. Learning rate is fixed to **1e-6**; 1e-5 is
excluded from all candidates and references.

Current South Africa reference scores (281 pairs):
- Starting global RM: 63.3452% (178 correct).
- Full-data fine-tuning, 1024 steps: 64.0569% (180 correct).
- Weighting-only, 1024 steps: 63.3452%.
- Filtering+weighting, 1024 steps: 60.8541%.

The filtered 1024-step run traversed its 1032-pair pool ~7.94 times; shorter
budgets test whether unnecessary repeated adaptation is hurting retention.

## Bounded sequence

| Candidate | Method | Steps | Tau | Beta |
|---|---|---:|---:|---:|
| filtered-128 | Filtering+weighting | 128 | 0.5 | 1 |
| filtered-256 | Filtering+weighting | 256 | 0.5 | 1 |
| filtered-512 | Filtering+weighting | 512 | 0.5 | 1 |
| filter-07-512 | Filtering+weighting | 512 | 0.7 | 1 |
| weighted-beta2 | Weighting-only | 1024 | inactive | 2 |
| weighted-beta4 | Weighting-only | 1024 | inactive | 4 |

Six candidates maximum, one GPU. If a shorter candidate beats the starting model
and existing full-data result, run full-data fine-tuning at its same step budget
before accepting it. Cache each control; at most three additional controls
(128/256/512). Stop only when a validated candidate strictly exceeds the starting
model and all full-data controls tested here. If only a starting-model improvement
is found, retain and report it honestly rather than label it a full-data win.
This remains single-seed exploratory selection on the fixed benchmark.

The runner requires complete final reports, successful reload validation, fixed
LR, matching model/data/software/LoRA settings and the same reference cohort.
Original experiment files, previous results and other countries are not changed.
Configuration: `configs/tulu_south_africa.json`. Outputs:
`artifacts/tulu-south-africa-04/search/status.json`, per-trial configs/logs,
full-data controls, checkpoints and reports. The source snapshot is alongside
that search directory. Slurm QoS is `yejinchoi`; the scheduled monitor stays paused.

## Caveat about the other countries

Five other filtered models exceed their starting global RM at LR1e-6:
Chile, New Zealand, Mexico, Israel and Canada. Australia does **not** yet satisfy
that condition: filtering+weighting is 65.6934% versus its starting model's
66.4234%; weighting-only ties the starting score. Australia is explicitly deferred
while focusing on South Africa, not marked as improved or retrained here.

Validation: three search/control tests passed locally and on PW; they verify
strict comparison against both references, reject premature acceptance when an
equal-step full-data control is stronger, enforce South-Africa-only scope and
fixed LR, and prevent output overwrite. The actual saved 64.0569% full-data
reference passed the result loader. Original-source fidelity, static checks,
Python compilation, and Slurm syntax pass.

Follow-up: the user also authorized a separate, four-setting Australia search.
The running South Africa job keeps its frozen source and original six candidates;
there is no duplicate South Africa submission. See `docs/tulu-australia.md`.
