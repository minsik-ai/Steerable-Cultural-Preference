# Tulu replication: baseline validation and reconstructed pilot

> Historical reconstructed-protocol record. Superseded for primary replication by [the original-source rollback](original-protocol-rollback.md). Existing results remain diagnostic evidence.

## Recovered versus missing recipe

The paper Appendix A specifies `allenai/Llama-3.1-Tulu-3-8B-RM`, batch 8,
learning rates 1e-6/1e-5/1e-4, 128/256/1024 optimizer steps, tau=0.5, beta=1.0,
LoRA alpha=64, dropout=0.1, and three shuffled runs per setting.

The repository contains only a commented Tulu model name, unused LoRA imports,
and `model_uploader_tulu.py`. The uploader loads an existing adapter and merges
it; it does not establish the training configuration. Git history and unreachable
blobs did not recover a LoRA rank or target modules. Historical input formatting,
precision, head training policy, sequence length and exact per-country settings
are also unresolved. The user was asked about original saved adapter configs;
none has been supplied.

Sources:
- https://arxiv.org/html/2606.18606v1#A1
- https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM

The model card specifies its embedded chat template. Tulu config lacks a pad
ID, while its tokenizer has an existing pad token; the runner sets the model's
pad ID to that tokenizer ID without changing embeddings.

## Initial configuration

`configs/tulu_pilot.json` explicitly records known settings and assumptions:

- Model revision `84e607acab8adab64fa8cdaa26a9391d28362802`.
- PRISM revision `18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6`; original saved
  global survey split (seed 0), verified against regenerated split user IDs.
- Pilot: LR 1e-6, 128 steps, batch 8, tau=0.5, beta=1.0, shuffle seed 0.
- Provisional LoRA: rank 8, q_proj/v_proj, alpha 64, dropout 0.1, score head saved.
- Reconstructed execution: BF16 on CUDA, SDPA, gradient checkpointing,
  max length 2048, dynamic padding, and deterministic evaluation. Truncated-row
  counts are recorded. These are not asserted to be the paper's original choices.

The original DeBERTa experiment, trainer, and collator remain unchanged. Tulu
reuses the actual weighted loss and collator in a separate runner. The training
filter uses the mathematically equivalent log-odds comparison, and the test
misranked subset remains strict reference chosen < rejected.

## Staged jobs

1. Score all seven countries' held-out sets in both legacy paired encoding and
   the official chat template. Save pairs, raw rewards, truncation counts,
   model/data revisions and per-country differences from paper global results.
2. If a format matches every global baseline within 0.05 points, use it for
   the pilot. Otherwise use the official chat template and explicitly record
   `baseline_matches_paper=false`; do not claim preprocessing reproduction.
3. Train Chile for two steps, export adapter/tokenizer, reload on a fresh base,
   compare scores (BF16 tolerance recorded), and check deterministic repeated eval.
4. Only after successful preflight and adapter validation, run seven separate
   128-step country pilots, at most two GPUs concurrently. Every pilot exports
   and validates its adapter and saves both overall and misranked accuracy.

No automatic parameter sweep or three-seed claim is made by this first pilot.
A nonmatching baseline or missing historical recipe remains an explicit gap even
if training succeeds. Both Slurm scripts use the account's `yejinchoi` QoS.
Outputs use `artifacts/tulu-pilot-01`; the snapshot is a frozen Git archive.
The scheduled monitor remains paused; Slurm dependencies run this sequence.

## Paper reference (untuned SCPO filtering + weighting)

| Country | Global RM | SCPO F+W |
|---|---:|---:|
| Chile | 63.64 | 64.07 |
| South Africa | 63.35 | 62.51 |
| New Zealand | 61.56 | 61.09 |
| Australia | 66.18 | 64.80 |
| Mexico | 51.64 | 53.01 |
| Israel | 62.68 | 61.74 |
| Canada | 68.79 | 62.33 |
| Macro | 62.55 | 61.36 |

These are comparison targets, not results from this pilot. Tuned and weighting-
only rows are separate configurations and must not be silently substituted.

## Local inspection and tests

```bash
.venv/bin/python scripts/tulu_replication.py --help
.venv/bin/python -m unittest discover -s tests -p 'test_tulu*.py' -v
```

The tests cover split/pair integrity, tied rewards, exact formatting choices and
truncation, format selection, and actual tiny-Llama weighted LoRA training plus
adapter/reward-head save/reload without downloading a model.

Validation on PW: all 20 repository tests passed, including tiny-Llama LoRA
training and fresh adapter reload. The full pinned model snapshot downloaded
successfully. Real PRISM preprocessing produced the same country pair counts as
the DeBERTa run with zero orphan users. Tokenizer pad ID 128256 is within model
vocabulary size 128264. GPU baseline/pilot results remain pending at submission.

## First pilot outcome and follow-up

All seven 128-step pilots and the preflight completed. Chat-template global
macro accuracy was 62.2754%, trained macro 62.5069%; legacy-pair global macro
was 52.4291%. No evaluated row was truncated. The baseline does not match the
paper exactly. Reference-misranked corrections were only 0-1.94% by country.
A saved-weight audit found mostly unchanged BF16 score heads despite nonzero
FP32 LoRA updates. See [precision diagnostics](tulu-diagnostics.md) for the
bounded next experiments and the distinction between old in-memory and new
fresh-adapter evaluation metrics.

The next method comparison explicitly includes **full-data fine-tuning** and
**weighting-only**, alongside filtering plus weighting, using identical LoRA
capacity and budgets. See [the matched comparison plan](tulu-method-comparison.md).
The paper's strongest average Tulu variant is weighting-only; a small gain over
the global RM alone is not the improvement criterion for this experiment.
