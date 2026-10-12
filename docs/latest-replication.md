# Latest replication results

Updated 2026-10-12. Results and reproduction details for the released country
reward models. The original experiment files remain unchanged.

## Released country reward models

[Hugging Face collection](https://huggingface.co/collections/MulticulturalAlignment/scpo-country-reward-models-6acd6ef257ba2b9f33b8db95) ·
[Release records and exact published revisions](../release/deberta/README.md)

All seven selected DeBERTa configurations reproduced their previously recorded
overall scores exactly in the source-derived rerun (array 58957, indices 0–6).
Each public MIT package includes the tokenizer, exact country training code,
settings, environment lockfile, citation, and provenance. Loading and inference
checks passed for all packages; uploaded weight hashes matched local checkpoints.

| Country | Starting model | SCPO | Gain (percentage points) |
|---|---:|---:|---:|
| Mexico | 51.64% | 64.75% | +13.11 |
| Chile | 54.55% | 60.61% | +6.06 |
| New Zealand | 56.55% | 62.40% | +5.85 |
| Canada | 60.40% | 66.18% | +5.78 |
| Israel | 63.03% | 66.90% | +3.87 |
| South Africa | 64.06% | 65.48% | +1.42 |
| Australia | 59.61% | 59.85% | +0.24 |

Reproduce the country settings with `scripts/run_country_defaults.py` and
`configs/country_defaults.json`, or use `reproduction/preferences.py` from a
model repository. Configuration selection used fixed test data; these are
checkpoint results, not an independent estimate of the paper's three-run means.
The table records original in-memory scoring; standard reloaded `eval()` inference
can differ. See the model cards for the complete evaluation context.

## Reproduction entry points

- Country defaults: `scripts/run_country_defaults.py`.
- Configuration: `configs/country_defaults.json`.
- Original source execution: `scripts/original_protocol.py`.
- Pinned environment: `requirements-pw-lock.txt` (Python 3.11.16).
