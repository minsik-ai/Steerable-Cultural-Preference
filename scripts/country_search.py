"""Bounded per-country configuration search; never edits the original experiment.

Selection uses the fixed published test benchmark, so matching scores are
exploratory configuration matches, not independent proof of reproduction.
"""
import argparse
import difflib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ORIGINAL_HASHES = {
    'preferences.py': 'df4c9e3991b8238c5a50a3d672d67ca3d6350f06a28a99bc6959e8edd817cd8e',
    'our_reward_trainer.py': '35aea8a2f4b30f51e5427b61418698d69528b868fd7b35abad7132508d092ab3',
    'our_inverse_reward_trainer.py': 'b41fc006ae62dfdd5553217cdb231c35d45c80dd5437f3f7c70ce006f59835a1',
    'our_data_collator.py': '67c1b081e7886ed8a5747abe7612ff255f0043d8c3d59ef64bb774db12518c79',
}
# Published tuned OpenAssistant overall means and reported +/- spreads, Tables 15-16.
TARGETS = {'Chile': (59.89, 0.66), 'South Africa': (64.17, 0.41),
           'New Zealand': (62.30, 0.64), 'Australia': (60.26, 0.56),
           'Mexico': (63.39, 1.25), 'Israel': (67.84, 0.54), 'Canada': (66.09, 0.16)}
COUNTRIES = ['Chile', 'New Zealand', 'Australia', 'Mexico', 'Israel']
BASE = dict(shuffle_seed=0, learning_rate=1e-6, tau=0.7, beta=1.1)
CANDIDATES = [
    dict(name='isolation-control', **BASE),
    dict(BASE, name='shuffle-1', shuffle_seed=1),
    dict(BASE, name='shuffle-2', shuffle_seed=2),
    dict(BASE, name='lr-1e-5', learning_rate=1e-5),
    dict(BASE, name='lr-1e-4', learning_rate=1e-4),
    dict(BASE, name='tau-07-beta-10', beta=1.0),
    dict(BASE, name='tau-05-beta-11', tau=0.5),
    dict(BASE, name='tau-05-beta-10', tau=0.5, beta=1.0),
]
COUNTRY_LINE = '    countries = ["Chile", "South Africa", "New Zealand", "Australia", "Mexico", "Israel", "Canada"]'


def verify_sources(repo):
    for name, expected in ORIGINAL_HASHES.items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original source changed: {name}')


def trial_source(original, country, config):
    if country not in TARGETS:
        raise ValueError(f'Unknown country: {country}')
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_HASHES['preferences.py']:
        raise ValueError('Cannot derive trial from modified source')
    replacements = {
        COUNTRY_LINE: f'    countries = [{country!r}]',
        'RANDOM_DATA_SHUFFLE_SEED = 0': f'RANDOM_DATA_SHUFFLE_SEED = {config["shuffle_seed"]}',
        'EDITED_LR = 1e-6': f'EDITED_LR = {config["learning_rate"]!r}',
        'REWARD_THRES = 0.7': f'REWARD_THRES = {config["tau"]!r}',
        'WEIGHTING_BETA = 1.10': f'WEIGHTING_BETA = {config["beta"]!r}',
    }
    lines = original.splitlines()
    for before, after in replacements.items():
        if lines.count(before) != 1:
            raise ValueError(f'Expected exactly one assignment: {before}')
        lines[lines.index(before)] = after
    return '\n'.join(lines) + '\n'


def read_metrics(output, country):
    result = {}
    for metric, filename in [('overall', 'full_report.json'), ('misranked', 'partial_report.json')]:
        report = json.loads((output / 'experiment/seed/0' / filename).read_text())
        value = report[country]['ours_partial']
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f'Invalid {metric} metric: {value!r}')
        result[metric] = value
        result[f'global_{metric}'] = report[country]['start']
    return result


def qualifies(country, metrics, validated):
    value = metrics.get('overall')
    return (validated and isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 100
            and value >= TARGETS[country][0] - TARGETS[country][1])


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def search(repo, root, baseline, country, *, candidates=None, require_match=True):
    verify_sources(repo)
    country_dir = root / country.lower().replace(' ', '-')
    country_dir.mkdir(parents=True, exist_ok=False)
    state_path = country_dir / 'status.json'
    state = dict(country=country, status='running', trials=[], paper_mean=TARGETS[country][0],
                 threshold=TARGETS[country][0] - TARGETS[country][1],
                 selection=('first validated overall >= paper mean minus reported spread'
                            if require_match else 'run recorded country default once'),
                 interpretation='exploratory fixed-test configuration match; not independent replication',
                 baseline=read_metrics(baseline, country) if baseline is not None else None,
                 slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(state_path, state)
    try:
        if baseline is not None and require_match:
            baseline_checkpoint = baseline / 'trunc' / country / 'ours_3epoch'
            validation = json.loads((baseline_checkpoint / 'validation.json').read_text())
            if qualifies(country, state['baseline'], validation.get('deterministic') is True):
                state.update(status='matched_existing', selected_checkpoint=str(baseline_checkpoint))
                write_json(state_path, state)
                return
        original = (repo / 'preferences.py').read_text()
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(repo) + os.pathsep + environment.get('PYTHONPATH', '')
        environment['TOKENIZERS_PARALLELISM'] = 'true'
        for config in CANDIDATES if candidates is None else candidates:
            verify_sources(repo)
            trial = country_dir / config['name']
            trial.mkdir()
            source = trial_source(original, country, config)
            script = trial / 'preferences_trial.py'
            script.write_text(source)
            (trial / 'source.patch').write_text(''.join(difflib.unified_diff(
                original.splitlines(keepends=True), source.splitlines(keepends=True),
                fromfile='original/preferences.py', tofile='trial/preferences.py')))
            write_json(trial / 'config.json', dict(config, country=country,
                       source_sha256=hashlib.sha256(source.encode()).hexdigest()))
            (trial / 'environment.txt').write_text('\n'.join(sorted(
                '{}=={}'.format(d.metadata['Name'], d.version) for d in importlib.metadata.distributions())) + '\n')
            output = trial / 'run'
            entry = dict(config=config, directory=str(trial), status='running')
            state['trials'].append(entry)
            write_json(state_path, state)
            with (trial / 'train.log').open('w') as log:
                process = subprocess.run([sys.executable, str(script), str(output)], cwd=repo,
                                         env=environment, stdout=log, stderr=subprocess.STDOUT)
            entry['exit_code'] = process.returncode
            if process.returncode:
                entry['status'] = 'failed_training'
                raise RuntimeError(f'Training failed: {trial}/train.log')
            entry['metrics'] = read_metrics(output, country)
            checkpoint = output / 'trunc' / country / 'ours_3epoch'
            with (trial / 'validation.log').open('w') as log:
                check = subprocess.run([sys.executable, str(repo / 'scripts/validate_checkpoint.py'),
                    str(checkpoint), '--tokenizer', 'OpenAssistant/reward-model-deberta-v3-base',
                    '--output', str(checkpoint / 'validation.json')], cwd=repo, env=environment,
                    stdout=log, stderr=subprocess.STDOUT)
            entry['validated'] = check.returncode == 0
            entry['status'] = 'completed' if entry['validated'] else 'failed_validation'
            write_json(state_path, state)
            if not require_match and not entry['validated']:
                raise RuntimeError(f'Checkpoint validation failed: {trial}/validation.log')
            if entry['validated'] and (not require_match or qualifies(country, entry['metrics'], True)):
                state.update(status='matched' if require_match else 'completed', selected_config=config, selected_metrics=entry['metrics'],
                             selected_checkpoint=str(checkpoint))
                write_json(state_path, state)
                return
        state['status'] = 'exhausted'
        write_json(state_path, state)
    except Exception as error:
        state.update(status='blocked', error=str(error))
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='immutable source snapshot')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--country', choices=list(TARGETS))
    parser.add_argument('--index', type=int)
    parser.add_argument('--plan', type=Path, help='explicit bounded continuation plan')
    args = parser.parse_args()
    if (args.country is None) == (args.index is None):
        parser.error('provide exactly one of --country or --index')
    plan = json.loads(args.plan.read_text()) if args.plan else None
    countries = plan['countries'] if plan else COUNTRIES
    if args.index is not None and not 0 <= args.index < len(countries):
        parser.error('country index is outside the plan')
    country = args.country or countries[args.index]
    if country not in countries:
        parser.error('country is outside the plan')
    baseline = args.baseline.resolve()
    if plan:
        baseline = baseline / plan['baselines'][country]
    search(args.repo.resolve(), args.root.resolve(), baseline, country,
           candidates=plan['candidates'] if plan else None)
