# Tülu chat inputs with shared evaluation

The user explicitly authorized retaining Tülu chat formatting while matching
the rest of DeBERTa's evaluation. Profile `tulu-chat-v1` implements this boundary
using the same source-derived experiment as `original-source-v1`.

## Evaluation contract

- Same survey split (seed 0), original interaction grouping, preference pairs,
  human-tie exclusion, accuracy calculation and reference-misranked definition.
- Same original interaction order and batches of up to eight responses.
- No evaluation truncation or added length cap.
- Starting model uses its original loaded mode; final scoring uses the trained
  in-memory model without changing its mode or resetting RNG. Consequently
  adapter dropout remains active after training, as in the original procedure.
- No deterministic-eval override, gradient-disabling context, or checkpoint
  reload before the authoritative score. A later reload check is separate.
- The sole input exception is Tülu's existing tokenizer chat template, applied
  with no generation prompt and no duplicate special tokens. Reference and
  trained models use the same formatting.

Each model's reference-misranked subset is defined by its own starting rewards;
the two subsets need not contain the same pairs. Training affects the final RNG
state; matching the procedure does not imply identical dropout masks across models.

## Training choices

Retain the selected per-country method, steps, tau/beta, shuffle seed and LoRA
settings. The user subsequently authorized both LR 1e-6 and 1e-5. Restore chat training inputs, dynamic padding,
and a 2048-token training cap. Keep the source-derived FP32 model loading,
fresh training-model construction, original filtering/loss and Trainer defaults.
Gradient checkpointing is not enabled: the original runs fit on H200, and the
longer training inputs are checked in the actual runs.
Do not silently change evaluation to solve a memory failure.

DeBERTa files, defaults and existing results are unchanged. Both historical raw
Tülu and deterministic reconstructed runs are retained. The new profile has its
own manifests and outputs; it is not proof of the missing historical Tülu recipe
or an independent three-seed paper replication. Previous selected checkpoints
cannot recover the original end-of-training RNG state by reloading: retraining
is required for the primary evaluation contract.

## Runs

The launch array uses indices 0–6 for selected country configurations and 7–13
for full-data training at 1024 steps, at the chosen LR and the same LoRA capacity.
Each array runs at most two workers. The user removed the smoke-test gate;
submit the country array directly, passing `1e-6` or `1e-5` as its fourth
argument and using a separate artifact root for each rate. Outputs use a new immutable source snapshot; old artifacts remain intact.
The full-data control uses the existing unit-weight loss variant and all eligible
training pairs. It is trained with LoRA, not all 8B parameters.
