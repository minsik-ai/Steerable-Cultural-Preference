"""Execute and summarize matched Tulu methods against full-data fine-tuning."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

METHODS = ('full_data', 'weighted_only', 'filtered_weighted')


def cases(plan):
    if tuple(plan['methods']) != METHODS or len(set(plan['countries'])) != len(plan['countries']):
        raise ValueError('Need unique countries and all three comparison methods')
    result = []
    for country in plan['countries']:
        for lr in plan['learning_rates']:
            for method in METHODS:
                result.append(dict(name=f"{country.lower().replace(' ', '-')}-lr{lr:g}-{method}",
                                   country=country, method=method, learning_rate=lr,
                                   max_steps=plan['max_steps'], shuffle_seed=plan['shuffle_seed']))
    if len({x['name'] for x in result}) != len(result):
        raise ValueError('Duplicate comparison cases')
    return result


def run_case(source, root, baseline, plan, index):
    items = cases(plan)
    if not 0 <= index < len(items):
        raise ValueError('Case index outside plan')
    case = items[index]
    config = json.loads((source / plan['base_config']).read_text())
    config.update({key: case[key] for key in ('method', 'learning_rate', 'max_steps', 'shuffle_seed')})
    config['torch_dtype'] = plan['precision']
    folder = root / 'configs'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{case['name']}.json"
    with path.open('x') as stream:
        json.dump(config, stream, indent=2)
    subprocess.run([sys.executable, str(source / 'scripts/tulu_replication.py'), 'train',
                    '--config', str(path), '--output', str(root / 'runs' / case['name']),
                    '--baseline', str(baseline), '--country', case['country'],
                    '--format', plan['input_format']], check=True)


def read_case(root, case, plan):
    folder = root / 'runs' / case['name']
    for name in ('complete.json', 'validation.json', 'report.json', 'manifest.json'):
        if not (folder / name).exists():
            return None
    completed = json.loads((folder / 'complete.json').read_text())
    validation = json.loads((folder / 'validation.json').read_text())
    report = json.loads((folder / 'report.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    config = manifest['config']
    if completed.get('complete') is not True or validation.get('passed') is not True:
        raise ValueError(f'Unvalidated case: {case["name"]}')
    for key in ('method', 'learning_rate', 'max_steps', 'shuffle_seed'):
        if config.get(key, 'filtered_weighted' if key == 'method' else None) != case[key]:
            raise ValueError(f'Case configuration mismatch: {key}')
    if (config['torch_dtype'] != plan['precision'] or report['country'] != case['country']
            or report.get('method') != case['method']
            or report['input_format'] != plan['input_format']
            or report['actual_global_step'] != case['max_steps']
            or report['evaluation_mode'] != 'fresh adapter reload, eval, no autocast'):
        raise ValueError('Case execution mismatch')
    if case['method'] in ('full_data', 'weighted_only') and report['retained_train_pairs'] != report['train_pairs']:
        raise ValueError('Full-data method did not use the complete training pool')
    if report['loss_weighting'] != ('unit' if case['method'] == 'full_data' else 'SCPO clipped exponential'):
        raise ValueError('Wrong loss weighting for method')
    if case['method'] == 'full_data' and report['actual_epochs'] < 1:
        raise ValueError('Full-data baseline has not completed one data pass')
    for stat in (report['metrics'], report['metrics']['reference_misranked']):
        value = stat['accuracy']
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Invalid accuracy')
    return dict(case=case, report=report, manifest=manifest)


def summarize(root, plan):
    loaded, missing = [], []
    for case in cases(plan):
        result = read_case(root, case, plan)
        if result is None:
            missing.append(case['name'])
        else:
            loaded.append(result)
    summary = dict(complete=not missing, expected_cases=len(cases(plan)), completed_cases=len(loaded),
                   missing=missing, interpretation=plan['interpretation'])
    if missing:
        return summary  # Never compute a macro over an incomplete country/method grid.
    first = loaded[0]['manifest']
    for item in loaded:
        config, reference = item['manifest']['config'], first['config']
        for key in ('model_name', 'model_revision', 'dataset_name', 'dataset_revision',
                    'lora', 'batch_size', 'max_length', 'gradient_checkpointing', 'tau', 'beta', 'score_batch_size'):
            if config[key] != reference[key]:
                raise ValueError(f'Unmatched shared setting: {key}')
        for key in ('packages', 'runner_sha256', 'python', 'model_seed', 'trainer_seed', 'survey_split_seed'):
            if item['manifest'][key] != first[key]:
                raise ValueError(f'Unmatched runtime: {key}')
    lookup = {(x['case']['country'], x['case']['learning_rate'], x['case']['method']): x['report'] for x in loaded}
    budgets, countries = [], {}
    for country in plan['countries']:
        rows = [x['report'] for x in loaded if x['case']['country'] == country]
        signatures = {(r['metrics']['pairs'], r['metrics']['reference_misranked']['pairs'],
                       r['baseline_metrics']['correct'], r['train_pairs']) for r in rows}
        if len(signatures) != 1:
            raise ValueError(f'Inconsistent country cohort/reference: {country}')
        best_full = max(lookup[country, lr, 'full_data']['metrics']['accuracy'] for lr in plan['learning_rates'])
        countries[country] = dict(best_tested_full_data=best_full, settings=[])
        for lr in plan['learning_rates']:
            reports = {m: lookup[country, lr, m] for m in METHODS}
            full = reports['full_data']['metrics']['accuracy']
            for method, report in reports.items():
                accuracy = report['metrics']['accuracy']
                countries[country]['settings'].append(dict(learning_rate=lr, method=method,
                    accuracy=accuracy, misranked_accuracy=report['metrics']['reference_misranked']['accuracy'],
                    delta_vs_matched_full_data=accuracy-full, delta_vs_best_tested_full_data=accuracy-best_full,
                    train_pool_pairs=report['retained_train_pairs'], actual_epochs=report['actual_epochs']))
    for lr in plan['learning_rates']:
        averages = {method: sum(lookup[c, lr, method]['metrics']['accuracy'] for c in plan['countries']) /
                    len(plan['countries']) for method in METHODS}
        budgets.append(dict(learning_rate=lr, macro_accuracy=averages,
                            delta_vs_full_data={m: averages[m]-averages['full_data'] for m in METHODS}))
    best_full_macro = max(x['macro_accuracy']['full_data'] for x in budgets)
    for budget in budgets:
        budget['delta_vs_best_tested_full_data_macro'] = {m: value - best_full_macro
            for m, value in budget['macro_accuracy'].items()}
    summary.update(countries=countries, matched_budgets=budgets,
                   best_full_data_macro=best_full_macro,
                   claim='Descriptive single-seed results; a positive delta is not a significance test')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run', 'summarize'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    plan = json.loads((args.source / 'configs/tulu_method_comparison.json').read_text())
    if args.stage == 'run':
        if args.baseline is None or args.index is None:
            parser.error('run requires baseline and index')
        run_case(args.source.resolve(), args.root.resolve(), args.baseline.resolve(), plan, args.index)
    else:
        result = summarize(args.root, plan)
        (args.root / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, indent=2))
