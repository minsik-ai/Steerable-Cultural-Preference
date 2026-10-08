# Steerable Cultural Preference Optimization of Reward Models (SCPO)

Paper: [Steerable Cultural Preference Optimization of Reward Models](https://arxiv.org/abs/2606.18606)

Accepted to **Pluralistic Alignment @ ICML 2026**.

SCPO adapts a *global* reward model into *culturally steerable* reward models. Starting
from an off-the-shelf RM trained on aggregate human preferences, it re-trains the model
per country so that it better reflects the preferences of users residing in that country —
without losing the universal preferences the global model already captures.

The method has two core mechanisms, both driven by the original (global) reward model's
scores:

- **Filtering** — keep only the preference pairs where the global RM *disagrees* with the
  local (minority) preference. These are the culturally distinctive signals worth learning from.
- **Weighting** — down-weight preference pairs in proportion to how strongly the global RM
  disagrees, so the loss focuses on high quality samples with subtle cultural differences.

## Data

All experiments run on the [PRISM Alignment dataset](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment)
(`utterances` and `survey` splits), loaded automatically from the Hugging Face Hub at
runtime. Users are bucketed by `survey.location.reside_country`, and only countries present
in both PRISM and GlobalOpinionsQA are used. The default run targets:

> Chile, South Africa, New Zealand, Australia, Mexico, Israel, Canada

The survey set is split 85/15 train/test (`RANDOM_DATA_SPLIT_SEED = 0`) and the split is
saved to disk for reproducibility.

## Method overview

For each country, `preferences.py`:

1. **Builds preference pairs.** Within each user/interaction turn, utterances are paired and
   ranked by the human `score` field. The higher-scored response becomes `chosen`, the lower
   `rejected`.
2. **Scores every utterance with the global RM** (`OpenAssistant/reward-model-deberta-v3-base`)
   to get reference rewards `r⁺`, `r⁻`.
3. **Filters** to the *edited* set: a pair is retained when the global RM disagrees with the
   human preference beyond a threshold —
   `exp(r⁺) / (exp(r⁺) + exp(r⁻)) < τ` (`REWARD_THRES = τ = 0.7`).
4. **Trains a country-specific RM** on the edited pairs with a weighted Bradley–Terry loss.
   Each pair is weighted by `w = min(exp((r⁺ − r⁻) / β), 1)` (`WEIGHTING_BETA = β = 1.10`),
   implemented in `OurRewardTrainer.compute_loss`.
5. **Evaluates** pairwise accuracy (does the RM rank `chosen` above `rejected`?) on both the
   full test set and the filtered/edited test set, before and after training.

Results are written as JSON to `<out_path>/experiment/seed/<seed>/`:

- `full_report.json` — accuracy on the full test set, per country + average
- `partial_report.json` — accuracy on the filtered (edited) test set
- `edited_ratios.json` — fraction of pairs retained by the filter, per country

## Repository layout

| File | Purpose |
| --- | --- |
| `preferences.py` | **Entry point.** End-to-end pipeline: data loading → pairing → filtering → per-country training → evaluation. |
| `our_reward_trainer.py` | `OurRewardTrainer`, a `trl.RewardTrainer` subclass with the global-RM-weighted loss. |
| `our_data_collator.py` | Collator that pads batches and carries the reference rewards (`ref_rewards_chosen/rejected`) through to the loss. |
| `our_inverse_reward_trainer.py` | Variant trainer (inverse weighting), used for ablations. |
| `extract_table_examples.py` | Reproduces the qualitative filtering/weighting examples (paper Tables 13 & 14). |
| `reward_model_test.py` | Minimal sanity check that scores a single (question, answer) pair. |
| `model_uploader_openassistant.py` / `model_uploader_tulu.py` | Push trained checkpoints to the Hugging Face Hub. |
| `requirements.txt` | Pinned environment. |

## Setup

The recorded PW defaults require Python 3.11.16, Linux, and a CUDA GPU (the pipeline trains DeBERTa-v3 & Tülu 3 reward models). Key
pinned dependencies: `torch==2.4.1`, `transformers==4.45.2`, `trl==0.11.4`,
`datasets==3.0.1`, plus `peft`.

```bash
pip install -r requirements-pw-lock.txt
```

## Usage

```bash
python scripts/run_country_defaults.py --output <out_path>
```

`<out_path>` must be a new output directory. The original unmodified experiment
remains available as `python preferences.py <out_path>` for historical comparison.

Selected per-country defaults and their observed results are recorded in
[configs/country_defaults.json](configs/country_defaults.json). See
[reproduction instructions](docs/country-defaults.md) for the exact environment,
PW launcher, and provisional Australia/Israel settings. Outputs from this default
launcher are grouped as `<out_path>/<country>/default/run/`.

### Original-script hyperparameters (top of `preferences.py`)

| Name | Meaning |
| --- | --- |
| `MODEL_NAME` | Global reward model to steer. |
| `REWARD_THRES` (τ) | Filtering threshold; lower keeps fewer, more distinctive pairs. |
| `WEIGHTING_BETA` (β) | Loss-weighting temperature; lower sharpens the down-weighting. |
| `EDITED_EPOCHS` / `EDITED_LR` | Epochs and LR for country-specific training. |
| `MAX_STEPS` | Step cap per country. |
| `BATCH_SIZE` | Per-device batch size. |
| `RANDOM_DATA_SPLIT_SEED` / `RANDOM_DATA_SHUFFLE_SEED` / `RANDOM_MODEL_SEED` | Reproducibility seeds. |

Use `configs/country_defaults.json` for active per-country overrides. The original
script and its historical override dictionaries are retained for comparison.

## Citation

```
@misc{scpo,
      title={Steerable Cultural Preference Optimization of Reward Models}, 
      author={Minsik Oh and Advit Deepak and Sophie Wu and Douwe Kiela and Ekaterina Shutova},
      year={2026},
      eprint={2606.18606},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.18606}, 
}
```

## PW replication

Use **Python 3.11** for the pinned dependency environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_data.py --output artifacts/data-validation.json
```

`preferences.py`, both trainers, and the collator are byte-for-byte identical to
original revision `94f8430054835aa919b226b4344b8f0e8424f29b`. The fidelity test
checks this explicitly. The integration test uses synthetic data and a tiny
local model; it does not constitute paper reproduction.

The PW checkout is `/home/minsik/Steerable-Cultural-Preference`, beside InferLLM.
Run the original seven-country experiment with a new output directory:

```bash
mkdir -p artifacts/logs
sbatch --output=artifacts/logs/original_%j.out scripts/reproduce_pw.sbatch \
  "$PWD" "$PWD/artifacts/original-01"
```

The runner rejects additional experiment overrides. It invokes the original
`python preferences.py <out_path>` entry point with its original seeds, scoring,
filtering, model mode, and 1024 optimizer steps per country. It records source
revision/diff and installed package versions outside the experiment code.
Checkpoint reload validation runs afterward in a separate process, using the
base tokenizer because the original script does not export tokenizer files.
That validator's evaluation mode does not affect training or reported accuracy.

Previous `artifacts/full-01` results came from the modified execution at commit
`676d16f` and must not be described as an original-code reproduction. The
original source uses floating Hub references and the repaired dependency file
is not an exact historical environment lock. Matching the original code is a
necessary checkpoint, not proof of matching the paper's three-run results.
See [the replication log](docs/replication-log.md) for evidence and remaining gaps.

### Country-specific search

The optional [bounded search](docs/country-search.md) keeps the original source
intact and stops independently when each remaining country meets its declared
paper-score threshold. `scripts/country_search_pw.sbatch` runs five country
workers, at most three at once; Canada and South Africa retain their existing
models. Trial configurations, both evaluation metrics, source diffs, and every
attempt are preserved. These are exploratory configuration matches.
