# Tulu learning-signal and precision diagnostics

> Historical reconstructed-protocol record. Superseded for primary replication by [the original-source rollback](original-protocol-rollback.md). Existing results remain diagnostic evidence.

## Evidence from the first seven pilots

All 128-step pilots completed and adapters were saved/reloaded, but reported
reference-misranked corrections were only 0-1.94%. Inspection of the actual saved
tensors found:

- LoRA B matrices are FP32 and have 1,310,720 nonzero entries per adapter. Learning
  is not entirely disabled; maximum absolute entries are around 6e-5.
- The score head is BF16. Only 61-66 of its 4096 elements changed from the base
  weights (about 1.5%), despite optimizer updates. Most small updates may round
  away. This is a concrete precision risk, not proof of the sole cause.
- Retained pair weights average 0.475-0.544, and only 0.3-1.8% are below 0.01.
  Thus near-zero loss weights do not explain the overall weak adaptation.
- The initial four-example reload tolerance allowed small BF16 score changes.
  Such changes can flip close pair rankings even if the numerical check passes.

Raw saved-tensor audit: `artifacts/tulu-diagnostics-02/saved-weight-audit.json`.
The original pilot outputs are retained unchanged.

## Controlled experiments

1. Load the same saved Chile adapter in BF16 and FP32. Score every held-out pair
   with the same format and batching. Compare base versus adapter, old report
   versus fresh reload, ties, rank flips, and both accuracies. This is inference
   diagnosis only; no checkpoint or default is mutated.
2. Recompute all seven global-model baselines in FP32, for both input formats.
   Compare against the paper before attributing differences to training.
3. After diagnostics succeed, run three Chile-only training controls:

| Case | Precision | LR | Steps |
|---|---|---:|---:|
| Same short run | FP32 | 1e-6 | 128 |
| Higher LR | FP32 | 1e-5 | 128 |
| Longer run | FP32 | 1e-6 | 1024 |

All keep tau=0.5, beta=1.0, LoRA rank8 q/v (still provisional), alpha64,
dropout0.1, batch8 and shuffle0. The chat-template format is held fixed to the first pilot for an isolated
comparison; baseline format diagnostics do not change these controls. Max one training-control GPU at a time; no all-country sweep.
FP32 controls disable BF16 training autocast. Head precision follows the base.

The runner now records first observed gradient-contribution norms and before/after weight
changes by trainable role. These hooks can observe only one contribution under gradient checkpointing,
not the complete accumulated backward gradient. Initial LoRA A gradients can be zero because LoRA B
starts at zero; that alone is not a failure. New reports retain in-memory results
but use full held-out evaluation of the freshly reloaded adapter as primary
metrics. They explicitly report ranking changes at reload. This prevents weak
four-row tolerance from being mistaken for full prediction equivalence.

Precision consistency between the baseline and training configuration is
required. Original DeBERTa sources/defaults and previous Tulu snapshots remain
unchanged. All Tulu controls are still reconstructed experiments, not a verified
historical recipe. The scheduled monitor remains paused.

## Validation

23 tests passed on PW, including actual tiny-Llama FP32 weighted training,
nonzero head/B gradients and weight changes, exact reload predictions, data
pairing, format handling, and original-source fidelity. Static Python checks,
compilation and shell syntax also passed. Actual full-model outcomes remain
pending. Source/config snapshots and outputs are under
`artifacts/tulu-diagnostics-02`.
