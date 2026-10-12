# Restore original execution before comparing DeBERTa and Tulu

The user explicitly requested rolling back non-hyperparameter execution changes.
The primary Tulu path now derives its program from the **unchanged** original
`preferences.py` (revision94f8430), just as the DeBERTa country launcher does.
Earlier Tulu results are retained as diagnostic history and must not be combined
with original DeBERTa scores as a comparable-backbone result.

## Audit and rollback

| Prior Tulu reconstruction | Restored shared source behavior |
|---|---|
| Chat-template or consistent pair training/scoring | Original tokenizer pairs for scoring; literal `prompt + " [SEP] " + answer` for training |
| 2048-token cap on both paths, dynamic training padding | Original training max512 with max-length padding; untruncated scoring |
| Flat country-row scoring batches | Original batches within each interaction, in original order |
| Forced eval and no-grad scoring | No explicit mode reset or no-grad; original post-training mode/RNG sequence |
| Explicit BF16/FP32, autocast and SDPA controls | Original model/Trainer defaults (recorded environment loads FP32, no mixed-precision override) |
| Explicit low-memory loading, labels and cache overrides | Original `from_pretrained` calls |
| Enabled checkpointing/input gradients | Original Trainer defaults; no explicit overrides |
| Log-odds filtering, Python-float/JSON references | Original tensor exponential expression and in-memory reference tensors |
| Cached baseline stage/reusing reference model for adaptation | Original loading/reference/training sequence with a fresh training model |
| Changed eval dataset, logging/eval settings | Original Trainer construction and full test evaluation dataset |
| Gradient hooks and update instrumentation | Removed from primary child execution |
| Fresh adapter reload replaces headline metrics | Original in-memory reports remain authoritative |
| Normalized IDs/extra data reconstruction | Original dataset grouping, pairing, shuffling and IDs |

The model itself necessarily differs. Tulu adds its model identifier, sets the
model pad ID from the tokenizer's **existing** pad token, and wraps the fresh
training model with the declared PEFT sequence-classification adapter. No new
pad token/embedding resize is used. LoRA rank/targets remain explicit provisional
hyperparameters; the original Tulu training code is absent. Paper-specified
alpha/dropout do not establish all other historical settings.

**This restores shared-code comparability, not proof of the missing Tulu paper
recipe.** In particular, restoring raw original inputs removes the official chat
template; a lower Tulu baseline is possible and must be reported, not corrected
silently. Likewise, same protocol does not imply equal trainable parameter count.

## Scope and RNG

Each family/country runs in its own fresh subprocess, matching the existing
country-isolated DeBERTa search. Model seed0 occurs once in that subprocess;
Trainer's original seed42/data_seedNone defaults remain. This does not reproduce
the evolving RNG history of an original seven-country sequential invocation.
Both families use the same isolated-run convention. No probe, reload, or reseed
is inserted before the original primary evaluation.

Previously authorized full-data and weighting-only methods remain explicit
method ablations: swap the training pair pool, and use equal zero references
for the unweighted loss only. They do not change scoring or preprocessing.

## Entry points and safeguards

- `scripts/original_protocol.py`: shared source-derived launcher.
- `scripts/tulu_replication.py`: Tulu default wrapper for that launcher.
- `configs/tulu_original_protocol.json`: previous selected Tulu hyperparameters,
  now awaiting reruns under this restored protocol.
- `scripts/tulu_reconstructed.py`: archived historical implementation. Direct CLI
  use requires explicit `--allow-reconstructed`; old baseline/train invocations
  through the default entry point fail with an explanatory message.

```bash
.venv/bin/python scripts/tulu_replication.py \
  --config configs/tulu_original_protocol.json --country Chile \
  --output artifacts/tulu-original-run
```

`--prepare-only` renders source and its diff for inspection. Any stale execution
knobs such as input_format, torch_dtype, max_length or gradient_checkpointing are
rejected. Per-country numeric settings and declared method variants are retained;
no new hyperparameter optimization is performed during this rollback.

The original loaders are unmodified and still resolve main. Separate preflight
processes check the recorded Python/package versions and current model/data
SHAs against the saved references before each GPU run. This check does not
silently pin/modify calls inside the original program. Environment, generated
source/hash/diff and original reports are saved with protocol `original-source-v1`.

## Verification and paired reruns

Tests check whole-program AST identity after removing only allowlisted changes,
original train formatter/Trainer construction, scoring batch/mode/RNG traces,
and real tiny Llama+LoRA execution (grad-enabled scoring, padded512 training,
post-training training mode, adapter save). Original DeBERTa source fidelity stays
byte-identical. The primary data/algorithm checks do not rely on a reload probe.

A two-step Chile smoke run for both families gates 14 paired reruns, seven per
family, using selected hyperparameters and the same protocol. Large-memory H200
nodes g0374/g0379 are used because restoring default precision and no checkpointing
may require more memory. A failure is reported; the runner must not silently
switch precision, truncate, or enable checkpointing to make it fit.

Outputs: `artifacts/original-protocol-06`. The final summary requires all14 primary
reports, matching source protocol, runtime packages, revision guards and ordered
train/test users. It rejects archived reconstructed results and incomplete-grid
averages. Previous cross-backbone comparison numbers are superseded pending this
new comparison. No scheduled monitor is enabled.

Validation outcome: all 42 tests passed on PW, including the real tiny-Llama
source-derived training path. The independent runtime guard verified Python
3.11.16, the package lock, both model revisions, and the PRISM revision. Static
checks, Python compilation and all new Slurm shell syntax checks passed.
Full-size GPU smoke/comparison results are pending submission/completion.
