"""Check PRISM split/pair prerequisites without loading a reward model."""
import argparse
import collections
import json
from itertools import combinations
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

COUNTRIES = ['Chile', 'South Africa', 'New Zealand', 'Australia', 'Mexico', 'Israel', 'Canada']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--revision')
    args = parser.parse_args()
    revision = HfApi().dataset_info('HannahRoseKirk/prism-alignment', revision=args.revision).sha
    survey = load_dataset('HannahRoseKirk/prism-alignment', 'survey', split='train', revision=revision)
    utterances = load_dataset('HannahRoseKirk/prism-alignment', 'utterances', split='train', revision=revision)
    assert len(set(survey['user_id'])) == len(survey), 'duplicate survey users'
    split = survey.train_test_split(test_size=0.15, seed=0)
    assert set(split['train']['user_id']).isdisjoint(split['test']['user_id'])
    users = {row['user_id']: (name, row['location']['reside_country'])
             for name, rows in split.items() for row in rows}
    groups = collections.defaultdict(list)
    for row in utterances:
        if row['user_id'] not in users:
            continue
        name, country = users[row['user_id']]
        if country in COUNTRIES:
            assert isinstance(row['user_prompt'], str) and isinstance(row['model_response'], str)
            assert isinstance(row['score'], (int, float))
            groups[(name, country, row['user_id'], row['interaction_id'])].append(row)
    counts = {name: {country: dict(interactions=0, utterances=0, pairs=0)
                     for country in COUNTRIES} for name in ('train', 'test')}
    for (name, country, user, interaction), rows in groups.items():
        assert len({row['user_prompt'] for row in rows}) == 1, 'mixed prompts within interaction'
        values = counts[name][country]
        values['interactions'] += 1
        values['utterances'] += len(rows)
        values['pairs'] += sum(a['score'] != b['score'] for a, b in combinations(rows, 2))
    assert all(values['pairs'] > 0 for countries in counts.values() for values in countries.values())
    result = dict(dataset_revision=revision, survey_rows=len(survey), utterance_rows=len(utterances),
                  split_seed=0, user_disjoint=True, counts=counts)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
