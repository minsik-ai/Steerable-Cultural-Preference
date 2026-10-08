# Bounded country-specific configuration search

Authorized after the original-code run `53802` completed. Keep Canada (66.185%)
and South Africa (65.480%) from `artifacts/original-01`; no additional trials for
these countries. Search only Chile, New Zealand, Australia, Mexico, and Israel.

## Predeclared stopping rule

Stop each country at the first checkpoint-validated overall accuracy at least
its published tuned mean minus the reported +/- spread (paper Tables 15-16):

| Country | Paper mean | Spread | Minimum match |
|---|---:|---:|---:|
| Chile | 59.89 | 0.66 | 59.23 |
| New Zealand | 62.30 | 0.64 | 61.66 |
| Australia | 60.26 | 0.56 | 59.70 |
| Mexico | 63.39 | 1.25 | 62.14 |
| Israel | 67.84 | 0.54 | 67.30 |

Source: https://arxiv.org/html/2606.18606v1#A7

This is a practical closeness rule, not a significance test. Always retain the
reference-misranked subset accuracy as a diagnostic. Since the user explicitly
accepts matching configurations against the paper benchmark, selection uses
the fixed test results. Label selected scores as exploratory configuration
matches, not unbiased independent reproduction or proof of the historical recipe.
No new validation split is silently introduced into the original experiment.

## Ordered trials, independently per country

| Trial | Shuffle seed | LR | Tau | Beta |
|---|---:|---:|---:|---:|
| Country-isolation control | 0 | 1e-6 | 0.7 | 1.1 |
| Shuffle 1 | 1 | 1e-6 | 0.7 | 1.1 |
| Shuffle 2 | 2 | 1e-6 | 0.7 | 1.1 |
| LR 1e-5 | 0 | 1e-5 | 0.7 | 1.1 |
| LR 1e-4 | 0 | 1e-4 | 0.7 | 1.1 |
| Weighting change | 0 | 1e-6 | 0.7 | 1.0 |
| Filtering change | 0 | 1e-6 | 0.5 | 1.1 |
| Untuned pair | 0 | 1e-6 | 0.5 | 1.0 |

At most 40 trials, with no more than three GPUs simultaneously. Seeds 1 and 2
are proposed replication probes, not recovered historical seeds. Do not average
selected country scores and present that average as a common-configuration,
three-seed result. No extra confirmation runs for a country already meeting the
user's stopping criterion. An exhausted or failed search must be reported.

## Execution and evidence

Original experiment/trainer/collator files remain byte-identical to revision
`94f8430`. Derive trial copies by changing exactly the final country-list
assignment plus `RANDOM_DATA_SHUFFLE_SEED`, `EDITED_LR`, `REWARD_THRES`, and
`WEIGHTING_BETA`. Leave split seed, Trainer defaults, number of steps (1024),
batch size (8), filtering arithmetic, and original evaluation behavior intact.
The original test-time model mode is preserved, including any active dropout.
Checkpoint validation runs separately and does not verify the original reported
accuracy independently.

Run against a frozen `git archive` source snapshot. Every trial retains generated
source, its SHA/diff, config, package versions, stdout, both metrics, checkpoint,
and reload validation. Each country atomically updates `status.json` and stops
at its first validated match. A repeated submission refuses to overwrite an
existing country directory. Training failures stop that country and retain the
error; failed checkpoint validation never qualifies as a match.

Run `python -m unittest discover -s tests -v`. Tests cover the allowed AST changes,
thresholds, finite metrics, validation failures, early stopping, preserved trials,
keeping an existing close model, and synthetic isolated-country equivalence.

Validation on PW: all eight unit/integration tests passed. The synthetic
Chile-only control produced byte-equal tensor values and equal reported metrics
to Chile in the original seven-country loop. Dependency compatibility, critical
Ruff, Python compilation, and shell syntax passed. Actual-model isolation is
checked by the first candidate for each searched country.
