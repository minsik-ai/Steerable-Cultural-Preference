---
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: OpenAssistant/reward-model-deberta-v3-base
base_model_relation: finetune
datasets:
- HannahRoseKirk/prism-alignment
language:
- en
tags:
- reward-model
- scpo
- cultural-alignment
- israel
- arxiv:2606.18606
---
# SCPO Reward Model — Israel

A scalar reward model fine-tuned on PRISM preference pairs from participants
residing in Israel. Higher scores indicate a preferred response for the
same prompt. Scores are not probabilities and are not calibrated across models.

[Paper](https://arxiv.org/abs/2606.18606) ·
[Code](https://github.com/minsik-ai/Steerable-Cultural-Preference)

## Recorded experiment results

| Measurement | Accuracy |
|---|---:|
| Starting global model, overall | 63.0282% |
| SCPO model, overall | 66.9014% |

These are original in-memory post-training experiment scores, with no model-mode
or random-state reset before scoring. A fresh `eval()` inference session disables
dropout and may produce different accuracy. `validation.json` checks loading and
finite, repeatable inference on generic probes; it is not a benchmark result.

Configurations were tuned on a fixed test split; these checkpoint results differ
from the paper’s three-run averages. Reproduction details are in `provenance.json`.

## Training and reproduction

- Full-model fine-tuning from `OpenAssistant/reward-model-deberta-v3-base`.
- Base revision: `7dab4156e63944834c3aae169c3fe83955672cff`.
- PRISM revision: `18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6`.
- Learning rate: `1e-06`; batch size: 8; optimizer steps: 1024.
- Filtering tau: 0.5; weighting beta: 0.8.
- Data shuffle seed: 2; survey split/model seed: 0;
  Trainer seed: 42; Trainer data seed: unset.
- Survey users split 85/15; human-rating ties excluded from preference pairs.
- Original training inputs use literal `prompt + " [SEP] " + response`, capped
  and padded to 512 tokens. Original scoring uses tokenizer prompt/response pairs
  in interaction batches of at most eight with no explicit truncation.

See `release-settings.json`, `provenance.json`, and `requirements-pw-lock.txt`.
The exact country script and unchanged trainer/collator are included under
`reproduction/`. With the recorded environment installed, run:

```bash
python reproduction/preferences.py ./reproduction-output
```

Original loaders use upstream `main`; verify the recorded model/data revisions
before rerunning. No private dataset records are bundled.

## Inference

The following uses standard evaluation-mode inference.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "MulticulturalAlignment/scpo-reward-model-israel"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
inputs = tokenizer("How can I disagree respectfully?",
                   "Listen carefully and explain your reasons calmly.",
                   return_tensors="pt")
with torch.inference_mode():
    reward = model(**inputs).logits.item()
print(reward)
```

## Intended use and limitations

Research on preference modeling and pluralistic alignment. Country labels reflect
participants' residence, not a representative account of a country's culture or
every resident's preferences. The English-dominant sample and small country test
sets limit generalization. This reward model does not generate text and is not
validated for consequential decisions about people.

## Attribution and release license

These fine-tuned weights are released under the [MIT License](https://opensource.org/license/mit).
The [OpenAssistant base model](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-base) is
marked MIT. The [PRISM dataset card](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment)
lists human text as CC-BY-4.0 and model responses as CC-BY-NC-4.0, subject to
original provider terms. This package contains no PRISM records.

## Citation

```bibtex
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
