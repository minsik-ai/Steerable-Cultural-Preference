# PW replication log

## Initial setup and change plan (historical; execution changes reverted below)

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


## Original-execution restoration (2026-10-07)

User requires paper replication and requested removal of all execution changes.
Restored `preferences.py` exactly from `94f8430054835aa919b226b4344b8f0e8424f29b`.
Both reward trainers and the collator also match that revision byte-for-byte.
This restores implicit Trainer seed/data-seed behavior, unbounded paired scoring,
post-training model mode, gradient-enabled forward passes, original exponent
threshold, floating Hub references, and the original CLI and artifact behavior.
The synthetic smoke test now exercises the original function with test-only
model/data/budget substitution. Production execution has no such substitutions.

Kept the installable dependencies and external validation/Slurm tooling. The
runner checks source fidelity, rejects experiment overrides, and uses a fresh
output path. Reload validation is a separate post-run process with the base
tokenizer; it cannot alter the experiment's reward scores or JSON reports.

Prior modified-execution jobs 53240 and 53241 both completed successfully. The
full run's overall macro accuracy was 61.9299% versus global RM 58.5473%; its
reference-misranked subset macro accuracy was 19.6093%. All seven checkpoint
reload validations passed. These artifacts remain under `artifacts/full-01`
for comparison and are explicitly **not** original-execution reproduction.

Paper comparisons: tuned SCPO overall 63.42%, reference-misranked subset 28.69%
(Tables 1–2), three runs per setting (Appendix A). The supplied code globally
splits survey rows, whereas the paper says country-stratified; this restoration
does not invent a replacement split. Original seeds, environment, and selected
settings still need reconciliation before numerical reproduction can be claimed.
Source: https://arxiv.org/html/2606.18606v1

Restoration validation: four tests pass on PW, including byte-identical source
verification and the original seven-country training/reporting path on synthetic
fixtures (one step per country). Dependency compatibility, Python compilation,
critical Ruff checks, shell syntax, and whitespace checks pass. Log:
`artifacts/logs/original-source-tests.log`.

The original-code rerun uses `artifacts/original-01`, leaving previous outputs
intact. Submission/status are recorded in `artifacts/original-submission.txt` and
Slurm `artifacts/logs/original_<jobid>.out`. No experiment-configuration override
is passed. Numerical comparison must wait for that run to complete.

## Selective country search

After original run 53802, the user authorized the staged ablations with early
stopping and requested no extra work on countries already close to the paper.
See `docs/country-search.md` for exact predeclared thresholds, ordered settings,
source-fidelity constraints, and the eight passing validation tests. Canada and
South Africa retain their original-01 checkpoints. Only the remaining five
countries are searched, using an immutable source snapshot and external trial
copies. Selected scores are exploratory fixed-test matches.

## Recorded per-country defaults and remaining search (2026-10-08)

All five workers in array 54203 completed successfully, totaling 34 validated
trials. Chile, New Zealand and Mexico matched; Canada and South Africa retained
their prior models. Australia and Israel exhausted the first candidate set.

`configs/country_defaults.json` now records the selected per-country settings,
checkpoint provenance and both observed metrics. Australia's 59.3674% and
Israel's 65.1408% best settings remain provisional. The new default launcher
reads this configuration and renders trial sources without changing original
experiment execution. `requirements-pw-lock.txt` captures the 69-package PW
Python 3.11.16 environment; all selected trials used the same environment.
Runtime guards check that environment and the recorded model/data revisions.

The user authorized continuing only Australia and Israel. The second bounded
plan is `configs/remaining_search.json`: eight new candidates per country near
tau 0.5/beta 1.0/LR 1e-6, at most two GPUs, first validated match stops the country.
The account QoS is explicitly `yejinchoi`. The prior monitor stays paused.

Validation: 12 unit/integration tests pass on PW; the runtime guard confirms the
recorded environment and both Hub revisions. Prepare-only rendered all seven
recorded configurations. Original source fidelity, critical static checks, Python
compilation, and shell syntax pass. Search outputs will use
`artifacts/country-search-02`, with a frozen source snapshot and two array workers.


## Second search outcome (2026-10-09)

Array 54640 completed with exit code 0 for both workers: Australia 9m17s, Israel
34m51s. Australia matched after two trials, at shuffle=2, LR=1e-6, tau=0.5,
beta=1.0: overall 59.8540%, misranked 35.5422%. Israel exhausted eight trials;
its best was shuffle=0, LR=1e-6, tau=0.5, beta=0.8: overall 66.1972%, misranked
37.1429%. Saved weights and reload validation were verified, and Israel's package
environment matches the lock. Promoted these settings and their checkpoint/source
provenance to country_defaults.json. Six countries meet the declared threshold;
Israel is still provisional. The monitor remains paused and no extra jobs were
submitted by this status check.


## Israel targeted follow-up authorization

User requested further improvement after best overall 66.1972% (188/284 correct).
The best model recovers 39/105 reference mistakes and retains 149/179 reference
correct pairs. Four more correct pairs cross the 67.30% threshold. New plan:
`configs/israel_search.json`; diagnostic and four trials only, preserving the
original execution semantics. No other country is included.

## Accepted Israel default and Tulu initialization (2026-10-10)

User accepted Israel's 66.9014% result as sufficient. Promoted shuffle seed 2,
LR=1e-6, tau=0.5, beta=0.8 from israel-search-03/best-shuffle-2; split remains 0.
Status is `accepted`, not `matched`, because it is below the earlier threshold.
Other six settings remain unchanged.

Tulu recipe audit found no rank/target configuration in current/history code.
The model is ungated and pinned to revision 84e607acab8adab64fa8cdaa26a9391d28362802.
A separate baseline/LoRA pilot runner, explicit assumptions config, tests, and
Slurm dependency stages are documented in docs/tulu-replication.md. No model
publication, external messages, or scheduled monitor restart are involved.

Tulu pre-submission evidence: 20 tests passed in the locked PW environment;
pinned model snapshot downloaded; real PRISM pairing and tokenizer compatibility
passed. Raw evidence is artifacts/tulu-setup/{tests.log,download.log,data-preflight.log}.
Model pad token 128256 is valid within vocab 128264. No original experiment file
was edited. GPU jobs use a frozen source snapshot and explicit dependencies.

## Tulu weak-learning investigation

All initial Tulu jobs (55337 and array 55338) completed. Saved adapter audit:
LoRA B FP32 nonzero, score heads BF16 with only 61-66/4096 elements changed,
retained loss-weight means 0.475-0.544. This supports testing precision before
expanding model/hyperparameter search. Baseline discrepancy is 0.555 points mean
absolute country error with the chat template; no test rows were truncated.

Added explicit precision controls, full-set fresh-reload metrics, and first-observed
gradient/weight-change auditing to the separate Tulu runner. The next
bounded sequence is a saved-adapter precision comparison, all-country FP32
baseline, then three Chile-only training controls. All 23 tests pass on PW.
Original sources and previously produced artifacts/defaults remain unchanged.

## Matched full-data baseline comparison

User requested weighting-only and emphasized beating full-data fine-tuning.
Added explicit full_data/weighted_only/filtered_weighted modes to the separate
Tulu runner, leaving the original trainer/source untouched. Full-data uses unit
weights with a mathematically equivalent zero-reference loss input; true global
references remain intact for reporting. Loss and gradient equivalence are tested.

Configured 42 matched runs (seven countries, three methods, two learning rates),
1024 steps, FP32, at most two GPUs, gated on successful prior precision/Chile
controls. Full-data baselines get multiple data passes. Summary checks complete
cohorts and compares each method against both matched-rate and strongest tested
full-data results, without dropping losing cases. 27 tests passed on PW. Outputs
are artifacts/tulu-method-comparison-03; no scheduled monitor is enabled.

## South Africa-only Tulu search

The user restricted focus to South Africa, fixed LR at 1e-6, and accepted the
other country models conditional on beating their starting models. Verified
South Africa global=63.3452%, full-data=64.0569%, weighting-only=63.3452%,
filtered+weighted=60.8541%. Five other filtered models beat global; Australia
is an exception (65.6934% below global66.4234%, weighting-only ties), flagged to
user and deferred without launching extra country work.

New bounded plan tests three shorter filtered budgets, a gentler filter, and
two gentler weighting settings. It adds an equal-step full-data control only
when a candidate beats the established references. At most six candidates plus
three controls, one GPU, no 1e-5 candidates. It stops on a validated improvement
and retains every trial. See docs/tulu-south-africa.md.

## Add Australia ablations without duplicating South Africa

User asked for a few ablations on Australia too. Job57313 was running: South
Africa filtered-128 tied global at63.3452%, filtered-256 underway. It retains its
original frozen source. Added explicit Australia-only plan with four candidates
at LR1e-6, the same equal-budget full-data controls, and fresh-reload validation.
Other countries remain out of scope. Starting Australia66.4234% is higher than
its full-data reference65.4501%, so a mere full-data win is insufficient.
