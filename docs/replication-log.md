# PW replication log

## Scope and change plan

Starting revision: `94f8430054835aa919b226b4344b8f0e8424f29b`.
Cluster checkout: `/home/minsik/Steerable-Cultural-Preference`, beside InferLLM.
Branch: `codex/pw-replication`.

1. Replace the host-wide Ubuntu package freeze with the experiment runtime;
   retain original Torch/Transformers/TRL/Datasets versions. Record resolution failures.
2. Test the existing weighted loss and collator before modifying experiment behavior.
3. Add explicit bounded-run arguments, provenance, deterministic inference, and
   actionable validation errors. Preserve the default country set, weighting,
   train/test selection rules, and original input formatting for replication.
4. Run isolated installation/import checks, unit tests, and a real PRISM/DeBERTa
   GPU smoke experiment. Attempt the full default run if those pass.
5. Record missing paper artifacts and distinguish smoke checks from reproduction.

## Known scientific gaps before execution

- Original requirements include Ubuntu-only packages (`python-apt`,
  `command-not-found`, etc.) and omit PEFT, python-dotenv, and SentencePiece.
  `multiprocess==0.70.17` conflicts with the pinned `dill==0.3.8`; use 0.70.16.
- `max_steps=1024` overrides three epochs. The full default remains 1024 optimizer
  steps **per country**, to preserve the supplied script.
- Training uses a literal ` [SEP] ` concatenation; scoring uses tokenizer pairs.
  Retained for historical replication; equivalence must be measured before changing it.
- Training retains pairs with sigmoid(reference margin) < 0.7, while the edited
  test set uses strict reference misrankings (margin < 0). Thus edited baseline
  accuracy is zero by construction. These are different subsets by design of
  the supplied code, not independent evidence of improvement.
- Global model seed is zero but the original RewardConfig defaults to seed 42.
  An explicit trainer seed is needed for a documented, repeatable run; historical
  exact matching requires confirmation of the original run configuration.
- No Tulu adapter training configuration exists in preferences.py, although the
  Tulu uploader expects an adapter. Do not invent LoRA ranks/targets and label
  them a reproduction.
- No reference numerical result files, original checkpoints, exact dataset/base
  revisions, or complete paper ablation/evaluation runners are tracked. A fresh
  successful run alone cannot establish numerical reproduction of the paper.

## Execution evidence

To be updated from actual cluster runs. Raw logs and outputs are ignored under
`artifacts/`; compact validation results belong in this document.

### Validation on PW (2026-10-06)

- Clean `uv venv --python 3.11`: Python 3.11.16.
- Original requirements fail parsing `distro-info===0.23ubuntu1` before installation.
- First reduced installation exposed `ModuleNotFoundError: rich` importing TRL.
  Added original `rich==13.9.2` and `tyro==0.8.13` pins; `uv pip check` now
  reports all 69 installed packages compatible.
- Five `unittest` tests pass, including actual TRL training for one optimizer
  step with a tiny locally generated BERT and synthetic preference data,
  report generation, tokenizer/checkpoint export, and reload. This validates
  software plumbing only; its accuracy values are not experimental results.
- `preferences.py --help`, Python compilation, shell syntax, `git diff --check`,
  and Ruff critical-error checks (`E9,F63,F7,F82`) pass.
- Real PRISM preflight passes: 1,500 survey rows, 68,371 utterances, unique
  survey user IDs, disjoint user train/test split, identical prompts within
  interactions, and nonempty preference-pair sets for every target country.
- Dataset SHA: `18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6`.
- DeBERTa SHA: `7dab4156e63944834c3aae169c3fe83955672cff` (Hub accessible;
  the actual weights/inference are still awaiting GPU execution).

| Country | Train pairs | Test pairs |
|---|---:|---:|
| Chile | 2457 | 462 |
| South Africa | 2758 | 281 |
| New Zealand | 2655 | 359 |
| Australia | 2450 | 411 |
| Mexico | 2339 | 122 |
| Israel | 1943 | 284 |
| Canada | 1844 | 346 |

These are all unequal-human-score pairs, before reference-model filtering.

### Remaining execution constraint

All 64 PW GPUs were allocated at validation time. The workspace is limited to
2 GiB RAM, so full-model CPU training there is not a safe substitute. Initial
GPU job 53239 was cancelled while pending to replace its older provenance
wrapper. Replacement smoke job **53240** requests one GPU for at most 30 minutes,
Chile, 32 interactions per split, and two optimizer steps. It must complete
training and checkpoint reload validation before the dependent full run starts.
No real-model accuracy, numerical paper reproduction, or release-ready weights
are claimed from the completed checks.

Raw evidence: `artifacts/logs/original-requirements.log`, `install.log`,
`unit-baseline.log` (the initial missing-rich failure), `unit-integration.log`,
`hub-preflight.log`, `data-validation.log`, and `artifacts/data-validation.json`.
Slurm logs use `artifacts/logs/smoke_<jobid>.out` and `full_<jobid>.out`.

### Deliberate execution corrections and open questions

- Explicit eval/inference mode removes post-training dropout and gradient memory.
- Scoring is capped at 512 tokens instead of being unbounded. This can change
  reference rewards, filtered-pair membership, and accuracy on long examples.
  Paired longest-first truncation still differs from concatenated tail truncation;
  short example equality does not establish general format equivalence.
- Trainer/data seeds are explicit zero. This differs from the implicit original
  Trainer default 42; original experiment seed provenance is still missing.
- The sigmoid threshold avoids overflow while preserving the mathematical rule.
- Empty pair subsets stop with counts rather than divide by zero; existing output
  directories are protected from overwrites. Per-country reports survive later
  failures, but automatic checkpoint resume is not implemented.
- The weighted loss and reference collator remain unchanged and are regression
  tested for the current no-margin path. Optional margin broadcasting is not
  validated and is unused by this experiment.
- Full-data, filtering-only, weighting-only, inverse-weighting, multiple-seed and
  Tulu experiments need their original configurations/runners. The qualitative
  example script uses beta 1.0 versus the main experiment's 1.10. Country learning
  rate override dictionaries are not applied by the supplied training loop.

Full job **53241** is queued with `afterok:53240` and
`--kill-on-invalid-dep=yes`. It requests one GPU, 48 GiB, eight CPUs, and up to
12 hours, runs all seven countries at 1024 optimizer steps each, and pins the
above model/dataset SHAs. Outputs: `artifacts/smoke-01` and `artifacts/full-01`.
At handoff: smoke is pending for priority/resources; full is pending dependency.
