# DeBERTa release

[Public release collection](https://huggingface.co/collections/MulticulturalAlignment/scpo-country-reward-models-6acd6ef257ba2b9f33b8db95).

Organization: [MulticulturalAlignment](https://huggingface.co/MulticulturalAlignment).
License: MIT (updated at the user’s request for permissive distribution).

Seven country-specific full-model checkpoints from the verified original-protocol
rerun are packaged with a pinned base tokenizer, exact reproduction sources,
country settings, environment lockfile, load validation and weight checksums.
No optimizer states, authentication files, or PRISM records are published.

The accuracy below is the original in-memory experiment score, not a new
benchmark of reloaded eval-mode inference. Country configurations were selected
on fixed test data; this is not an independent three-seed reproduction.

| Country | Model | Original overall accuracy | Gain over starting model |
|---|---|---:|---:|
| Mexico | [scpo-deberta-v3-base-mexico](https://huggingface.co/MulticulturalAlignment/scpo-deberta-v3-base-mexico) | 64.7541% | +13.11 pp |
| Chile | [scpo-deberta-v3-base-chile](https://huggingface.co/MulticulturalAlignment/scpo-deberta-v3-base-chile) | 60.6061% | +6.06 pp |
| New Zealand | [scpo-deberta-v3-base-new-zealand](https://huggingface.co/MulticulturalAlignment/scpo-deberta-v3-base-new-zealand) | 62.3955% | +5.85 pp |
| Canada | [scpo-deberta-v3-base-canada](https://huggingface.co/MulticulturalAlignment/scpo-deberta-v3-base-canada) | 66.1850% | +5.78 pp |
| Israel | [scpo-deberta-v3-base-israel](https://huggingface.co/MulticulturalAlignment/scpo-deberta-v3-base-israel) | 66.9014% | +3.87 pp |
| South Africa | [scpo-deberta-v3-base-south-africa](https://huggingface.co/MulticulturalAlignment/scpo-deberta-v3-base-south-africa) | 65.4804% | +1.42 pp |
| Australia | [scpo-deberta-v3-base-australia](https://huggingface.co/MulticulturalAlignment/scpo-deberta-v3-base-australia) | 59.8540% | +0.24 pp |

## Release procedure

`prepare_deberta_release.py` verifies settings/source/metric correspondence and
loads every packaged checkpoint with its tokenizer. `publish_deberta_release.py`
checks local weight hashes, refuses to overwrite pre-existing untracked Hub
repositories, uploads privately, verifies remote weight checksums and file sizes,
then makes all seven public. Its `publication.json` records commit IDs and status.

PW package root: `artifacts/deberta-release-01`.
Verification job: 60618 (completed, exit 0). Publication job: 60619 (completed, exit 0). All seven public model APIs and the
seven-item collection were verified without authentication. `publication.json`
records the published revisions.
Tülu is excluded from this release. No message is sent to Niels.
