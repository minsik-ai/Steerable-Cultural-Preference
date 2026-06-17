# Steerable Cultural Preference Optimization of Reward Models (SCPO)

Paper: Steerable Cultural Preference Optimization of Reward Models

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

Requires Python 3.8+ and a CUDA GPU (the pipeline trains DeBERTa-v3 & Tülu 3 reward models). Key
pinned dependencies: `torch==2.4.1`, `transformers==4.45.2`, `trl==0.11.4`,
`datasets==3.0.1`, plus `peft`.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python preferences.py <out_path>
```

`<out_path>` is the root directory for all artifacts — the saved data split, per-country
model checkpoints (`<out_path>/trunc/<country>/...`), and the JSON reports above.

### Key hyperparameters (top of `preferences.py`)

| Name | Meaning |
| --- | --- |
| `MODEL_NAME` | Global reward model to steer. |
| `REWARD_THRES` (τ) | Filtering threshold; lower keeps fewer, more distinctive pairs. |
| `WEIGHTING_BETA` (β) | Loss-weighting temperature; lower sharpens the down-weighting. |
| `EDITED_EPOCHS` / `EDITED_LR` | Epochs and LR for country-specific training. |
| `MAX_STEPS` | Step cap per country. |
| `BATCH_SIZE` | Per-device batch size. |
| `RANDOM_DATA_SPLIT_SEED` / `RANDOM_DATA_SHUFFLE_SEED` / `RANDOM_MODEL_SEED` | Reproducibility seeds. |

Per-country overrides are available via `REWARD_THRES_OVERRIDE`, `LR_OVERRIDE`, and
`EDITED_LR_OVERRIDE`.

