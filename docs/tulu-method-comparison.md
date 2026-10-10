# Tulu comparison against full-data fine-tuning

The user explicitly requires improvement over full-data fine-tuning, not just
improvement over the global RM. This experiment measures that comparison; it
does not assume or guarantee a positive result.

## Matched design

Seven countries x three methods x two learning rates = 42 runs, at most two GPUs.
Each run uses the same FP32 base, provisional rank8 q/v LoRA plus trainable score
head, chat-template encoding, fixed user split0, model seed0, Trainer seed42,
dataset shuffle0, batch8, and **1024 optimizer steps**. Learning rates are 1e-6
and 1e-5, tested for every method. This is a bounded two-rate, single-seed screen;
not the entire paper hyperparameter search or statistical proof of superiority.

| Method | Eligible training pairs | Loss weights |
|---|---|---|
| full_data | All unequal-rating country pairs | One |
| weighted_only | Same full pool, same order before common shuffling | SCPO |
| filtered_weighted | Reference preference probability <0.5 | SCPO |

SCPO uses beta=1.0. Full-data means all available training pairs, **not dense
full-parameter fine-tuning**: LoRA capacity is held identical to the other methods.
The shorter 128-step pilot offered only ~1024 pair presentations. The new budget
allows multiple passes over each full country pool; actual epochs and pool sizes
are recorded and the summarizer requires the full-data baseline to complete at
least one data pass. This is a fixed optimizer-step comparison, not an exact
GPU-time/token-budget or equal-epoch comparison.

The original weighted trainer stays unchanged. Full-data samples receive equal
zero reference inputs only inside the loss, making exp((0-0)/beta)=1 and yielding
ordinary unweighted pairwise log-sigmoid loss and gradients. Real reference
rewards remain intact in reports and define the same held-out misranked subset.

## Evaluation and reporting

All primary scores come from a freshly reloaded adapter in deterministic eval
mode. Require validation.json plus complete.json and final full-test reports;
a checkpoint-validation file alone is insufficient. The summarizer verifies
shared model/data revisions, LoRA settings, software, runner hash, seeds, pools,
precision, input format, and effective steps. It refuses incomplete-grid macros.

For every country and learning rate, report:
- Overall and reference-misranked accuracy for all three methods.
- SCPO minus full-data accuracy at the **same** learning rate.
- SCPO minus the **best tested** full-data baseline for that country.
- Country-macro comparisons at each learning rate and against the best tested
  full-data macro. No averaging only winning countries or dropping failures.

A positive single-seed difference is exploratory. A promising result should be
checked across additional matched training seeds before a robust improvement
claim. All runs and both learning-rate results are retained even if SCPO loses.

## Execution

Config: `configs/tulu_method_comparison.json`.
Runner: `scripts/tulu_method_comparison.py`.
GPU array: `scripts/tulu_comparison_pw.sbatch` (0-41, throttle2, yejinchoi QoS).
Summary: `scripts/tulu_comparison_summary_pw.sbatch` (CPU only, afterany dependency).

The array depends on the FP32 precision/baseline job 55667 and all three Chile
control tasks in array 55668 completing successfully. It uses frozen source and
new outputs under `artifacts/tulu-method-comparison-03`. Prior artifacts and all
DeBERTa defaults are unchanged. The scheduled monitor remains paused.

Validation: 27 tests passed on PW, including exact unit-weight loss/gradient
checks, identical full pools, filtering, complete-grid aggregation, comparison
against the strongest baseline, runtime mismatch rejection and source fidelity.
Static checks, Python compilation, and shell syntax also passed.
