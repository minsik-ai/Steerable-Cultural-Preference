"""Bounded, explicitly reconstructed Tulu replication; never modifies historical code."""
import argparse
import collections
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COUNTRIES = ['Chile', 'South Africa', 'New Zealand', 'Australia', 'Mexico', 'Israel', 'Canada']
PAPER_GLOBAL = dict(zip(COUNTRIES, [63.64, 63.35, 61.56, 66.18, 51.64, 62.68, 68.79]))
FORMATS = ('legacy-pair', 'chat-template')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_config(config):
    for key in ('model_name', 'model_revision', 'dataset_name', 'dataset_revision', 'countries'):
        if not config.get(key):
            raise ValueError(f'Missing config field: {key}')
    for key in ('model_revision', 'dataset_revision'):
        revision = config[key]
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError(f'{key} must be a pinned 40-character commit SHA')
    if config['countries'] != COUNTRIES:
        raise ValueError('Baseline must include the seven countries in canonical order')
    if config.get('max_length', 2048) <= 0:
        raise ValueError('max_length must be positive')
    return config


def build_data(survey_split, utterances, countries=COUNTRIES):
    """Keep distinct users/interactions separate; omit human-rating ties."""
    users = {}
    result = {part: {country: {'rows': [], 'pairs': []} for country in countries}
              for part in ('train', 'test')}
    for part in ('train', 'test'):
        for row in survey_split[part]:
            uid = row['user_id']
            if uid in users:
                raise ValueError('Duplicate survey user or train/test leakage')
            users[uid] = (part, row['location']['reside_country'])
    groups = collections.defaultdict(list)
    seen = set()
    missing_survey_rows = 0
    for row in utterances:
        if row['user_id'] not in users:
            missing_survey_rows += 1
            continue
        part, country = users[row['user_id']]
        if country not in countries:
            continue
        uid = str(row['utterance_id'])
        if uid in seen:
            raise ValueError('Duplicate utterance ID')
        seen.add(uid)
        if not isinstance(row['user_prompt'], str) or not isinstance(row['model_response'], str):
            raise ValueError('Non-text prompt or response')
        if not isinstance(row['score'], (float, int)) or not math.isfinite(row['score']):
            raise ValueError('Non-finite human rating')
        clean = {key: row[key] for key in ('user_id', 'interaction_id', 'user_prompt', 'model_response', 'score')}
        clean['utterance_id'] = uid
        result[part][country]['rows'].append(clean)
        groups[part, country, row['user_id'], row['interaction_id']].append(clean)
    for (part, country, _, _), rows in groups.items():
        if len({r['user_prompt'] for r in rows}) != 1:
            raise ValueError('Mixed prompts in user interaction')
        for a, b in itertools.combinations(rows, 2):
            if a['score'] == b['score']:
                continue
            chosen, rejected = (a, b) if a['score'] > b['score'] else (b, a)
            result[part][country]['pairs'].append({'chosen_id': chosen['utterance_id'],
                                                  'rejected_id': rejected['utterance_id']})
    result['_diagnostics'] = {'utterances_without_survey_user': missing_survey_rows}
    return result


def encode_rows(tokenizer, rows, input_format, max_length):
    if input_format not in FORMATS:
        raise ValueError('Unknown input format')
    prompts = [r['user_prompt'] for r in rows]
    answers = [r['model_response'] for r in rows]
    kwargs = dict(padding=False, truncation=False)
    if input_format == 'chat-template':
        texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p},
                                               {'role': 'assistant', 'content': a}],
                                              tokenize=False, add_generation_prompt=False)
                 for p, a in zip(prompts, answers)]
        unbounded = tokenizer(texts, add_special_tokens=False, **kwargs)
        tokens = tokenizer(texts, add_special_tokens=False, padding=False,
                           truncation=True, max_length=max_length)
    else:
        unbounded = tokenizer(prompts, answers, **kwargs)
        tokens = tokenizer(prompts, answers, padding=False, truncation=True, max_length=max_length)
    lengths = [len(ids) for ids in unbounded['input_ids']]
    features = [{key: tokens[key][i] for key in ('input_ids', 'attention_mask')}
                for i in range(len(rows))]
    return features, {'rows': len(rows), 'truncated_rows': sum(n > max_length for n in lengths),
                      'maximum_untruncated_length': max(lengths, default=0), 'max_length': max_length}


def score(model, tokenizer, rows, input_format, max_length, batch_size):
    import torch
    features, truncation = encode_rows(tokenizer, rows, input_format, max_length)
    rewards = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = tokenizer.pad(features[start:start + batch_size], padding=True, return_tensors='pt')
            batch = {k: v.to(model.device) for k, v in batch.items()}
            values = model(**batch).logits.float().reshape(-1).cpu().tolist()
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Non-finite reward')
            rewards.update({r['utterance_id']: v for r, v in zip(rows[start:start + batch_size], values)})
    return rewards, truncation


def metrics(pairs, rewards, reference=None):
    correct = sum(rewards[p['chosen_id']] > rewards[p['rejected_id']] for p in pairs)
    result = {'pairs': len(pairs), 'correct': correct,
              'accuracy': 100 * correct / len(pairs) if pairs else None}
    if reference is not None:
        wrong = [p for p in pairs if reference[p['chosen_id']] < reference[p['rejected_id']]]
        result['reference_misranked'] = metrics(wrong, rewards)
    return result


def load_model(config, tokenizer):
    import torch
    from transformers import AutoModelForSequenceClassification
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'], revision=config['model_revision'], torch_dtype=dtype,
        attn_implementation='sdpa', num_labels=1, low_cpu_mem_usage=True)
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer needs an existing pad token; no embedding resize is permitted')
    model.config.pad_token_id = tokenizer.pad_token_id
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


def manifest(config, stage):
    import importlib.metadata
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'datasets', 'trl', 'peft', 'accelerate')}
    return {'stage': stage, 'config': config, 'python': platform.python_version(), 'packages': packages,
            'model_seed': 0, 'trainer_seed': 42, 'survey_split_seed': 0,
            'execution': 'reconstructed Tulu recipe; deterministic eval, explicit formatting and length cap',
            'scoring_batches': 'country rows batched sequentially, rather than historical per-interaction batches; eval mode',
            'historical_unknowns': ['LoRA rank and target modules', 'Tulu preprocessing', 'historical library versions'],
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def baseline(config, output, split_path=None):
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer, set_seed
    set_seed(0)
    survey = load_dataset(config['dataset_name'], 'survey', split='train', revision=config['dataset_revision'])
    expected = survey.train_test_split(test_size=0.15, seed=0)
    split = load_from_disk(str(split_path)) if split_path else expected
    for part in ('train', 'test'):
        if split[part]['user_id'] != expected[part]['user_id']:
            raise ValueError('Saved split disagrees with pinned dataset and seed 0')
    utterances = load_dataset(config['dataset_name'], 'utterances', split='train', revision=config['dataset_revision'])
    data = build_data(split, utterances)
    if any(not data[part][country]['pairs'] for part in ('train', 'test') for country in COUNTRIES):
        raise ValueError('Missing country train/test pairs')
    split.save_to_disk(str(output / 'survey-split'))
    write_json(output / 'data.json', data)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'])
    model = load_model(config, tokenizer)
    report = {'formats': {}, 'paper_global_macro': 62.55}
    for fmt in FORMATS:
        countries = {}
        for country in COUNTRIES:
            item = data['test'][country]
            rewards, truncation = score(model, tokenizer, item['rows'], fmt, config.get('max_length', 2048),
                                        config.get('score_batch_size', 4))
            stats = metrics(item['pairs'], rewards)
            stats.update(rewards=rewards, truncation=truncation, paper_global=PAPER_GLOBAL[country],
                         difference_pp=stats['accuracy'] - PAPER_GLOBAL[country])
            countries[country] = stats
            print(fmt, country, stats['accuracy'], flush=True)
            report['formats'][fmt] = {'countries': countries}
            write_json(output / 'report.json', report)
        report['formats'][fmt]['macro_accuracy'] = sum(x['accuracy'] for x in countries.values()) / len(COUNTRIES)
        report['formats'][fmt]['mean_absolute_difference_pp'] = sum(abs(x['difference_pp']) for x in countries.values()) / len(COUNTRIES)
        report['formats'][fmt]['maximum_absolute_error_pp'] = max(abs(x['difference_pp']) for x in countries.values())
    write_json(output / 'report.json', report)


def train(config, output, baseline_path, country, input_format, max_steps=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from transformers import AutoTokenizer, set_seed
    from trl import RewardConfig
    from our_reward_trainer import OurRewardTrainer
    from our_data_collator import OurRewardDataCollatorWithPadding
    baseline_manifest = json.loads((baseline_path / 'manifest.json').read_text())
    for key in ('model_name', 'model_revision', 'dataset_name', 'dataset_revision'):
        if config[key] != baseline_manifest['config'][key]:
            raise ValueError(f'Baseline mismatch: {key}')
    if config.get('max_length', 2048) != baseline_manifest['config'].get('max_length', 2048):
        raise ValueError('Baseline maximum length mismatch')
    data = json.loads((baseline_path / 'data.json').read_text())
    base_report = json.loads((baseline_path / 'report.json').read_text())
    reference_test = base_report['formats'][input_format]['countries'][country]['rewards']
    training, testing = data['train'][country], data['test'][country]
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'])
    model = load_model(config, tokenizer)
    max_length = config.get('max_length', 2048)
    reference, truncation = score(model, tokenizer, training['rows'], input_format, max_length,
                                  config.get('score_batch_size', 4))
    tau, beta = config.get('tau', .5), config.get('beta', 1.)
    if not 0 < tau < 1 or beta <= 0:
        raise ValueError('Require 0 < tau < 1 and beta > 0')
    # sigmoid(delta) < tau, written as log-odds to avoid overflow.
    retained = [p for p in training['pairs'] if reference[p['chosen_id']] - reference[p['rejected_id']]
                < math.log(tau / (1 - tau))]
    if not retained:
        raise ValueError('No training pairs survived filtering')
    features, _ = encode_rows(tokenizer, training['rows'], input_format, max_length)
    encoded = {r['utterance_id']: f for r, f in zip(training['rows'], features)}
    samples = []
    for pair in retained:
        sample = {}
        for side in ('chosen', 'rejected'):
            uid = pair[f'{side}_id']
            sample.update({f'{key}_{side}': value for key, value in encoded[uid].items()})
            sample[f'ref_rewards_{side}'] = [reference[uid]]
        samples.append(sample)
    lora = config['lora']
    for key in ('r', 'target_modules', 'lora_alpha', 'lora_dropout', 'modules_to_save'):
        if key not in lora:
            raise ValueError(f'Explicit reconstructed LoRA field required: {key}')
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, **lora))
    steps = max_steps if max_steps is not None else config.get('max_steps', 128)
    if steps <= 0:
        raise ValueError('max_steps must be positive')
    model.config.use_cache = False
    gradient_checkpointing = config.get('gradient_checkpointing', True)
    if gradient_checkpointing:
        model.enable_input_require_grads()
    args = RewardConfig(output_dir=str(output / 'trainer'), per_device_train_batch_size=config.get('batch_size', 8),
                        max_steps=steps, learning_rate=config.get('learning_rate', 1e-6),
                        gradient_checkpointing=gradient_checkpointing, bf16=torch.cuda.is_available(),
                        eval_strategy='no', save_strategy='no', logging_steps=1, report_to='none',
                        max_length=max_length, remove_unused_columns=False, seed=42)
    trainer = OurRewardTrainer(model=model, args=args, beta=beta,
                               data_collator=OurRewardDataCollatorWithPadding(tokenizer),
                               train_dataset=Dataset.from_list(samples).shuffle(seed=config.get('shuffle_seed', 0)))
    trainer.train()
    adapter = output / 'adapter'
    model.save_pretrained(str(adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    rewards, test_truncation = score(model, tokenizer, testing['rows'], input_format, max_length,
                                   config.get('score_batch_size', 4))
    report = {'country': country, 'input_format': input_format, 'evaluation_mode': 'eval; dropout disabled',
              'max_steps': steps, 'actual_global_step': trainer.state.global_step,
              'train_pairs': len(training['pairs']), 'retained_train_pairs': len(retained),
              'train_truncation': truncation, 'test_truncation': test_truncation,
              'metrics': metrics(testing['pairs'], rewards, reference_test),
              'baseline_metrics': metrics(testing['pairs'], reference_test, reference_test),
              'rewards': rewards, 'train_reference_rewards': reference}
    write_json(output / 'report.json', report)
    probe = testing['rows'][:min(4, len(testing['rows']))]
    expected, _ = score(model, tokenizer, probe, input_format, max_length, 1)
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(load_model(config, tokenizer), str(adapter))
    actual, _ = score(model, tokenizer, probe, input_format, max_length, 1)
    repeated, _ = score(model, tokenizer, probe, input_format, max_length, 1)
    before = torch.tensor(list(expected.values()))
    after = torch.tensor(list(actual.values()))
    torch.testing.assert_close(before, after, rtol=.01, atol=.05)
    torch.testing.assert_close(after, torch.tensor(list(repeated.values())), rtol=0, atol=1e-6)
    write_json(output / 'validation.json', {'passed': True, 'probe_count': len(probe),
                'reload_max_abs_error': (before - after).abs().max().item(),
                'deterministic_eval': True, 'reload_rtol': .01, 'reload_atol': .05})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('baseline', 'train'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--country', choices=COUNTRIES)
    parser.add_argument('--format', dest='input_format', choices=FORMATS)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text()))
    if args.stage == 'train' and not all((args.baseline, args.country, args.input_format)):
        parser.error('train requires --baseline, --country, and --format')
    args.output.mkdir(parents=True, exist_ok=False)
    record = manifest(config, args.stage)
    record['arguments'] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output / 'manifest.json', record)
    if args.stage == 'baseline':
        baseline(config, args.output, args.split)
    else:
        train(config, args.output, args.baseline, args.country, args.input_format, args.max_steps)
    write_json(args.output / 'complete.json', {'stage': args.stage, 'complete': True})


if __name__ == '__main__':
    main()
