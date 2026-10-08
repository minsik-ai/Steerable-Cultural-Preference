"""Reproduce the recorded per-country defaults without editing original sources."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.country_search import BASE, TARGETS, search, trial_source, verify_sources, write_json

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = REPO / 'configs/country_defaults.json'


def load_settings(path):
    settings = json.loads(path.read_text())
    if set(settings['countries']) != set(TARGETS):
        raise ValueError('Settings must include exactly the seven experiment countries')
    fixed = dict(model='OpenAssistant/reward-model-deberta-v3-base',
                 dataset='HannahRoseKirk/prism-alignment',
                 experiment_source_revision='94f8430054835aa919b226b4344b8f0e8424f29b',
                 batch_size=8, max_steps=1024, split_seed=0, model_seed=0,
                 trainer_seed=42, trainer_data_seed=None)
    for key, expected in fixed.items():
        if settings.get(key) != expected:
            raise ValueError(f'Original execution requires {key}={expected!r}')
    for country, config in settings['countries'].items():
        if not isinstance(config['shuffle_seed'], int) or isinstance(config['shuffle_seed'], bool) or config['shuffle_seed'] < 0:
            raise ValueError(f'Invalid shuffle seed for {country}')
        for key in ('learning_rate', 'tau', 'beta'):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid {key} for {country}')
        if config['tau'] > 1:
            raise ValueError(f'Invalid tau for {country}')
    return settings


def check_runtime(settings):
    if '.'.join(map(str, sys.version_info[:3])) != settings['python']:
        raise ValueError(f"Use Python {settings['python']} for the recorded environment")
    for line in (REPO / 'requirements-pw-lock.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Environment mismatch: install {name}=={expected}')
    from huggingface_hub import HfApi
    api = HfApi()
    # Original code resolves main itself. Refuse drift instead of modifying its loaders.
    for kind in ('model', 'dataset'):
        actual = getattr(api, f'{kind}_info')(settings[kind]).sha
        if actual != settings[f'{kind}_revision']:
            raise ValueError(f'{kind} main has changed: {actual}; recover the recorded revision before reproducing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--countries', nargs='+', choices=list(TARGETS))
    parser.add_argument('--prepare-only', action='store_true', help='render and inspect trial sources without training')
    args = parser.parse_args()
    settings = load_settings(args.settings)
    verify_sources(REPO)
    if not args.prepare_only:
        check_runtime(settings)
    countries = list(dict.fromkeys(args.countries or TARGETS))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'settings.json', settings)
    manifest = dict(status='prepared' if args.prepare_only else 'running', countries=countries, completed=[])
    write_json(root / 'manifest.json', manifest)
    try:
        for country in countries:
            entry = settings['countries'][country]
            config = dict(name='default', **{key: entry[key] for key in BASE})
            if args.prepare_only:
                folder = root / country.lower().replace(' ', '-')
                folder.mkdir()
                (folder / 'preferences_trial.py').write_text(trial_source(
                    (REPO / 'preferences.py').read_text(), country, config))
                write_json(folder / 'config.json', config)
            else:
                search(REPO, root, None, country, candidates=[config], require_match=False)
            manifest['completed'].append(country)
            write_json(root / 'manifest.json', manifest)
        manifest['status'] = 'prepared' if args.prepare_only else 'complete'
        write_json(root / 'manifest.json', manifest)
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        write_json(root / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
