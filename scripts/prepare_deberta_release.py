"""Package and load-check the seven reproduced DeBERTa checkpoints; never upload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


CITATION = '@misc{scpo,\n      title={Steerable Cultural Preference Optimization of Reward Models}, \n      author={Minsik Oh and Advit Deepak and Sophie Wu and Douwe Kiela and Ekaterina Shutova},\n      year={2026},\n      eprint={2606.18606},\n      archivePrefix={arXiv},\n      primaryClass={cs.CL},\n      url={https://arxiv.org/abs/2606.18606}, \n}'

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def card(country, settings, metrics, defaults):
    slug = country.lower().replace(' ', '-')
    return f'''---
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: {defaults['model']}
base_model_relation: finetune
datasets:
- {defaults['dataset']}
language:
- en
tags:
- reward-model
- scpo
- cultural-alignment
- {slug}
- arxiv:2606.18606
---
# SCPO Reward Model — {country}

A scalar reward model fine-tuned on PRISM preference pairs from participants
residing in {country}. Higher scores indicate a preferred response for the
same prompt. Scores are not probabilities and are not calibrated across models.

[Paper](https://arxiv.org/abs/2606.18606) ·
[Code](https://github.com/minsik-ai/Steerable-Cultural-Preference)

## Recorded experiment results

| Measurement | Accuracy |
|---|---:|
| Starting global model, overall | {metrics['global_overall']:.4f}% |
| SCPO model, overall | {metrics['overall']:.4f}% |

These are original in-memory post-training experiment scores, with no model-mode
or random-state reset before scoring. A fresh `eval()` inference session disables
dropout and may produce different accuracy. `validation.json` checks loading and
finite, repeatable inference on generic probes; it is not a benchmark result.

Configurations were tuned on a fixed test split; these checkpoint results differ
from the paper’s three-run averages. Reproduction details are in `provenance.json`.

## Training and reproduction

- Full-model fine-tuning from `{defaults['model']}`.
- Base revision: `{defaults['model_revision']}`.
- PRISM revision: `{defaults['dataset_revision']}`.
- Learning rate: `{settings['learning_rate']}`; batch size: 8; optimizer steps: 1024.
- Filtering tau: {settings['tau']}; weighting beta: {settings['beta']}.
- Data shuffle seed: {settings['shuffle_seed']}; survey split/model seed: 0;
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

model_id = "MulticulturalAlignment/scpo-reward-model-{slug}"
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
The [OpenAssistant base model](https://huggingface.co/{defaults['model']}) is
marked MIT. The [PRISM dataset card](https://huggingface.co/datasets/{defaults['dataset']})
lists human text as CC-BY-4.0 and model responses as CC-BY-NC-4.0, subject to
original provider terms. This package contains no PRISM records.

## Citation

```bibtex
{chr(10).join(line.rstrip() for line in CITATION.splitlines())}
```
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    defaults = json.loads((args.repo / 'configs/country_defaults.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(defaults['model'], revision=defaults['model_revision'])
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for country, settings in defaults['countries'].items():
        slug = country.lower().replace(' ', '-')
        run = args.runs / slug
        manifest = json.loads((run / 'manifest.json').read_text())
        if manifest['status'] != 'completed' or manifest['family'] != 'deberta' or manifest['protocol'] != 'original-source-v1':
            raise ValueError(f'Unexpected run: {country}')
        for key in ('learning_rate', 'tau', 'beta', 'shuffle_seed'):
            if manifest['config'][key] != settings[key]:
                raise ValueError(f'Settings mismatch: {country}/{key}')
        for key, value in settings['observed'].items():
            if abs(manifest['primary_metrics'][key] - value) > 1e-9:
                raise ValueError(f'Reproduction mismatch: {country}/{key}')
        for name, expected in manifest['original_sha256'].items():
            if digest(args.repo / name) != expected:
                raise ValueError(f'Original source changed: {name}')
        checkpoint = Path(manifest['checkpoint'])
        package = args.output / f'scpo-reward-model-{slug}'
        package.mkdir()
        for name in ('config.json', 'model.safetensors'):
            shutil.copyfile(checkpoint / name, package / name)
        tokenizer.save_pretrained(package)
        shutil.copyfile(args.repo / 'requirements-pw-lock.txt', package / 'requirements-pw-lock.txt')
        write_json(package / 'release-settings.json', dict(country=country, **manifest['config'],
                   split_seed=0, model_seed=0, trainer_seed=42, trainer_data_seed=None))
        write_json(package / 'provenance.json', {
            'country': country, 'protocol': manifest['protocol'],
            'source_revision': manifest['source_revision'],
            'original_sha256': manifest['original_sha256'],
            'generated_sha256': manifest['generated_sha256'],
            'base_model_revision': defaults['model_revision'],
            'dataset_revision': defaults['dataset_revision'],
            'experiment_metrics': manifest['primary_metrics'],
            'weight_sha256': digest(checkpoint / 'model.safetensors'),
            'selection': defaults['interpretation'],
        })
        (package / 'README.md').write_text(card(country, settings, manifest['primary_metrics'], defaults))
        subprocess.run([sys.executable, str(args.repo / 'scripts/validate_checkpoint.py'),
                        str(package), '--output', str(package / 'validation.json')], check=True)
        validation = json.loads((package / 'validation.json').read_text())
        validation['checkpoint'] = package.name
        write_json(package / 'validation.json', validation)
        hashes = {p.name: digest(p) for p in package.iterdir() if p.is_file()}
        write_json(package / 'checksums.json', hashes)
        summary[country] = dict(package=package.name, metrics=manifest['primary_metrics'],
                                validated=True, weight_sha256=hashes['model.safetensors'])
        write_json(args.output / 'release-summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
