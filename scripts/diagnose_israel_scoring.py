"""Separate scoring-variance diagnostic; never changes checkpoints or defaults."""
import argparse
import collections
import itertools
import json
from pathlib import Path
import statistics

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


def accuracy(scores, pairs):
    if not pairs:
        raise ValueError('Empty diagnostic subset')
    return 100 * sum(scores[a] > scores[b] for a, b in pairs) / len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='artifact checkout root')
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostic results')
    settings = json.loads(args.settings.read_text())
    record = settings['countries']['Israel']
    run = args.repo / record['run']
    test = load_from_disk(str(run / 'dataset/PRISM/seed/0'))['test']
    users = {r['user_id'] for r in test if r['location']['reside_country'] == 'Israel'}
    utterances = load_dataset(settings['dataset'], 'utterances', split='train', revision=settings['dataset_revision'])
    groups = collections.defaultdict(list)
    for row in utterances:
        if row['user_id'] in users:
            groups[f"{row['user_id']}_{row['interaction_id']}"].append(row)
    pairs = []
    for rows in groups.values():
        for a, b in itertools.combinations(rows, 2):
            if a['score'] != b['score']:
                chosen, rejected = (a, b) if a['score'] > b['score'] else (b, a)
                pairs.append((chosen['utterance_id'], rejected['utterance_id']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(settings['model'], revision=settings['model_revision'])

    def score(model):
        result = {}
        # Preserve original interaction ordering, batching, tokenization and length.
        # No-grad is diagnostic-only: no optimizer or backward pass is involved.
        with torch.no_grad():
            for rows in groups.values():
                for start in range(0, len(rows), 8):
                    batch = rows[start:start + 8]
                    inputs = tokenizer([r['user_prompt'] for r in batch],
                                       [r['model_response'] for r in batch],
                                       padding=True, return_tensors='pt').to(device)
                    values = model(**inputs).logits.flatten().cpu()
                    if not torch.isfinite(values).all():
                        raise ValueError('Nonfinite diagnostic rewards')
                    result.update({r['utterance_id']: values[i].item() for i, r in enumerate(batch)})
        return result

    model = AutoModelForSequenceClassification.from_pretrained(
        settings['model'], revision=settings['model_revision']).to(device).eval()
    reference = score(model)
    misranked = [(a, b) for a, b in pairs if reference[a] < reference[b]]
    agreed = [(a, b) for a, b in pairs if reference[a] > reference[b]]
    if (len(users), len(pairs), len(misranked), len(agreed)) != (9, 284, 105, 179):
        raise ValueError('Diagnostic cohort differs from the recorded Israel experiment')
    if abs(accuracy(reference, pairs) - record['observed']['global_overall']) > 1e-8:
        raise ValueError('Reference scoring does not match the recorded experiment')
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.repo / record['checkpoint'], local_files_only=True).to(device)
    results = []
    for mode, seeds in [('train_dropout', range(5)), ('eval', [0])]:
        model.train(mode == 'train_dropout')
        for seed in seeds:
            set_seed(seed)
            values = score(model)
            results.append(dict(mode=mode, scoring_seed=seed, overall=accuracy(values, pairs),
                                misranked=accuracy(values, misranked), agreed=accuracy(values, agreed)))
    values = [r['overall'] for r in results if r['mode'] == 'train_dropout']
    report = dict(checkpoint=record['checkpoint'], interpretation='Diagnostic only; does not replace original results',
                  users=len(users), pairs=len(pairs), misranked_pairs=len(misranked), agreed_pairs=len(agreed),
                  recorded=record['observed'], runs=results,
                  dropout_summary=dict(mean=statistics.mean(values), stdev=statistics.stdev(values),
                                       minimum=min(values), maximum=max(values)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
