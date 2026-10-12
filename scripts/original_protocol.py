"""Run both backbones through the supplied experiment, with recorded adaptations.

Each country runs in a fresh process. This matches the isolated DeBERTa search,
not the RNG history of the historical seven-country sequential invocation.
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

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.country_search import BASE, ORIGINAL_HASHES, TARGETS, trial_source, verify_sources, write_json

MODELS = {'deberta': 'OpenAssistant/reward-model-deberta-v3-base',
          'tulu': 'allenai/Llama-3.1-Tulu-3-8B-RM'}
METHODS = ('filtered_weighted', 'weighted_only', 'full_data')
HYPERPARAMETERS = ('shuffle_seed', 'learning_rate', 'tau', 'beta', 'batch_size', 'max_steps')


def resolve_config(document, country):
    """Overlay recorded per-country settings without interpreting old runner knobs."""
    countries = document.get('countries', list(TARGETS))
    if country not in countries or country not in TARGETS:
        raise ValueError(f'Country not present in config: {country}')
    config = {key: value for key, value in document.items() if key != 'countries'}
    if isinstance(countries, dict):
        config.update(countries[country])
    return config


def normalized_config(family, config):
    if family not in MODELS:
        raise ValueError(f'Unknown family: {family}')
    # Reject stale reconstruction controls instead of silently ignoring them.
    forbidden = ('precision', 'input_format', 'gradient_checkpointing', 'score_batch_size',
                 'max_length', 'torch_dtype', 'attn_implementation', 'cache_dir',
                 'scoring_seed', 'eval_mode', 'evaluation_mode', 'deterministic_eval')
    for key in forbidden:
        if key in config:
            raise ValueError(f'{key} is not an original-protocol hyperparameter; remove the old runner control')
    for key, value in {'split_seed': 0, 'model_seed': 0, 'trainer_seed': 42,
                       'trainer_data_seed': None}.items():
        if config.get(key, value) != value:
            raise ValueError(f'Original {key} must remain {value!r}')
    result = dict(BASE, batch_size=8, max_steps=1024, method='filtered_weighted')
    for key in (*HYPERPARAMETERS, 'method'):
        if key in config:
            result[key] = config[key]
    for key in ('shuffle_seed', 'batch_size', 'max_steps'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'shuffle_seed' else 1):
            raise ValueError(f'Invalid {key}: {value!r}')
    for key in ('learning_rate', 'tau', 'beta'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid {key}: {value!r}')
    if result['method'] not in METHODS:
        raise ValueError('Unknown ablation method')
    result['model_name'] = config.get('model_name', config.get('model', MODELS[family]))
    if result['model_name'] != MODELS[family]:
        raise ValueError('Model name must match the selected family')
    if family == 'tulu':
        lora = config.get('lora')
        if not isinstance(lora, dict):
            raise ValueError('Tulu requires an explicit LoRA configuration; historical recipe is unknown')
        allowed = {'r', 'target_modules', 'lora_alpha', 'lora_dropout', 'modules_to_save', 'bias'}
        if set(lora) - allowed:
            raise ValueError(f'Unsupported LoRA options: {sorted(set(lora) - allowed)}')
        rank = lora.get('r')
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError('LoRA rank r must be an explicit positive integer')
        targets = lora.get('target_modules')
        if not isinstance(targets, list) or not targets or not all(isinstance(x, str) and x for x in targets):
            raise ValueError('LoRA target_modules must be an explicit nonempty string list')
        for key in ('lora_alpha', 'lora_dropout'):
            value = lora.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'LoRA {key} must be explicit and finite')
        if lora['lora_alpha'] <= 0 or not 0 <= lora['lora_dropout'] < 1:
            raise ValueError('Invalid LoRA alpha/dropout')
        if 'modules_to_save' in lora and (not isinstance(lora['modules_to_save'], list)
                or not all(isinstance(x, str) and x for x in lora['modules_to_save'])):
            raise ValueError('modules_to_save must be a string list')
        if lora.get('bias', 'none') not in ('none', 'all', 'lora_only'):
            raise ValueError('Invalid LoRA bias')
        result['lora'] = dict(lora)
    return result


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'Expected exactly one source anchor: {before!r}')
    return source.replace(before, after, 1)


def derive_source(original, family, country, config):
    """Make only explicit hyperparameter, family-adapter, and method substitutions."""
    config = normalized_config(family, config)
    source = trial_source(original, country, config)
    if config['batch_size'] != 8:
        source = replace_once(source, 'BATCH_SIZE = 8\n', f'BATCH_SIZE = {config["batch_size"]}\n')
    if config['max_steps'] != 1024:
        source = replace_once(source, 'max_steps=MAX_STEPS[count],', f'max_steps={config["max_steps"]},')
    if family == 'tulu':
        source = replace_once(source, 'from peft import LoraConfig, TaskType\n',
                              'from peft import LoraConfig, TaskType, get_peft_model\n')
        source = replace_once(source, f'MODEL_NAME = "{MODELS["deberta"]}"\n',
                              f'MODEL_NAME = {config["model_name"]!r}\n')
        source = replace_once(source, '    model = model.to(device)\n',
                              '    model.config.pad_token_id = tokenizer.pad_token_id\n'
                              '    model = model.to(device)\n')
        anchor = '        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n'
        source = replace_once(source, anchor, anchor
                              + '        model.config.pad_token_id = tokenizer.pad_token_id\n'
                              + f'        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, **{config["lora"]!r}))\n')
    if config['method'] in ('weighted_only', 'full_data'):
        source = replace_once(source,
                              'Dataset.from_list(utt_pref_edited_train[count]).map(format_func_train)',
                              'Dataset.from_list(utt_pref_train[count]).map(format_func_train)')
    if config['method'] == 'full_data':
        # Equal training references make the unchanged loss weight exactly one.
        # Raw reference scores, filtering diagnostics and test references stay intact.
        for side in ('chosen', 'rejected'):
            anchor = f'            ref_rewards_{side} = utt_train_rewards[examples["{side}_id"]]\n'
            source = replace_once(source, anchor, anchor
                                  + f'            ref_rewards_{side} = torch.zeros_like(ref_rewards_{side})\n')
    compile(source, '<original-protocol>', 'exec')
    return source


def primary_metrics(output, country):
    result = {}
    for name, filename in [('overall', 'full_report.json'), ('misranked', 'partial_report.json')]:
        report = json.loads((output / 'experiment/seed/0' / filename).read_text())
        if set(report) != {country, 'avg'}:
            raise ValueError(f'Unexpected country coverage in {filename}')
        for group in (country, 'avg'):
            for field in ('start', 'ours_partial'):
                value = report[group][field]
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f'Invalid {filename} {group}/{field}: {value!r}')
        result[name] = report[country]['ours_partial']
        result[f'global_{name}'] = report[country]['start']
    return result


def run_country(repo, output, family, country, config, prepare_only=False):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    verify_sources(repo)
    settings = normalized_config(family, config)
    original = (repo / 'preferences.py').read_text()
    source = derive_source(original, family, country, settings)
    directory = output / country.lower().replace(' ', '-')
    directory.mkdir(parents=True, exist_ok=False)
    script = directory / 'preferences_trial.py'
    script.write_text(source)
    (directory / 'source.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile='original/preferences.py', tofile='trial/preferences.py')))
    (directory / 'environment.txt').write_text('\n'.join(sorted(
        f'{dist.metadata["Name"]}=={dist.version}' for dist in importlib.metadata.distributions())) + '\n')
    adaptations = ['country-isolated subprocess', 'explicit hyperparameter substitutions']
    if family == 'tulu':
        adaptations.extend(['Tulu model ID', 'model pad ID from unchanged tokenizer',
                            'explicit PEFT SEQ_CLS adapter on fresh training model'])
    if settings['method'] != 'filtered_weighted':
        adaptations.append(f'authorized method ablation: {settings["method"]}')
    manifest = dict(protocol='original-source-v1', family=family, country=country, config=settings, status='prepared',
                    primary_evaluation='original in-memory post-training scoring; no mode or RNG reset',
                    process_semantics='one fresh process per country; not original seven-country sequential RNG history',
                    source_revision='94f8430054835aa919b226b4344b8f0e8424f29b',
                    original_sha256=ORIGINAL_HASHES,
                    generated_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    adaptations=adaptations,
                    historical_lora_recipe_verified=False if family == 'tulu' else None,
                    revisions='Original from_pretrained/load_dataset calls are unpinned; config revision metadata is not applied',
                    selection='Previously selected settings do not establish independent reproduction',
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    path = directory / 'manifest.json'
    write_json(path, manifest)
    if prepare_only:
        return manifest
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(repo) + os.pathsep + environment.get('PYTHONPATH', '')
    environment['TOKENIZERS_PARALLELISM'] = 'true'
    manifest['status'] = 'running'
    write_json(path, manifest)
    try:
        verify_sources(repo)
        with (directory / 'train.log').open('w') as log:
            process = subprocess.run([sys.executable, str(script), str(directory / 'run')],
                                     cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT)
        manifest['exit_code'] = process.returncode
        if process.returncode:
            raise RuntimeError(f'Original-protocol child failed: {directory / "train.log"}')
        manifest['primary_metrics'] = primary_metrics(directory / 'run', country)
        manifest['checkpoint'] = str(directory / 'run/trunc' / country / 'ours_3epoch')
        manifest['checkpoint_reload_validated'] = False
        manifest['status'] = 'completed'
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        write_json(path, manifest)
    return manifest


def main(default_family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if default_family is None:
        parser.add_argument('--family', required=True, choices=list(MODELS))
    else:
        parser.set_defaults(family=default_family)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--country', action='append', choices=list(TARGETS))
    group.add_argument('--countries', nargs='+', choices=list(TARGETS))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    document = json.loads(args.config.read_text())
    countries = args.country or args.countries or list(document.get('countries', TARGETS))
    if len(countries) != len(set(countries)):
        parser.error('Duplicate countries are not allowed')
    # Validate the complete request before starting the first country.
    configs = {country: normalized_config(args.family, resolve_config(document, country)) for country in countries}
    for country, config in configs.items():
        run_country(args.repo, args.output, args.family, country, config, args.prepare_only)


if __name__ == '__main__':
    main()
