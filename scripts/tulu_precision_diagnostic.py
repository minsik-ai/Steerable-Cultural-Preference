"""Read-only precision audit of an existing reconstructed Tulu pilot."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tulu_reconstructed import load_model, metrics, score, write_json


def summarize(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {'count': 0}
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite diagnostic value')
    def quantile(q):
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {'count': len(values), 'min': values[0], 'p10': quantile(.1),
            'median': quantile(.5), 'p90': quantile(.9), 'max': values[-1],
            'mean': sum(values) / len(values)}


def compare_rewards(pairs, before, after):
    """Compare strict correctness and all three ranking outcomes, including ties."""
    if set(before) != set(after):
        raise ValueError('Reward IDs differ; comparison would change evaluation coverage')
    result = {'pair_count': len(pairs), 'changed_pair_ranks': 0,
              'correctness_flips': 0, 'fixed': 0, 'regressed': 0,
              'before_ties': 0, 'after_ties': 0}
    for pair in pairs:
        chosen, rejected = pair['chosen_id'], pair['rejected_id']
        a, b = before[chosen] - before[rejected], after[chosen] - after[rejected]
        rank_a, rank_b = (a > 0) - (a < 0), (b > 0) - (b < 0)
        result['changed_pair_ranks'] += rank_a != rank_b
        result['correctness_flips'] += (a > 0) != (b > 0)
        result['fixed'] += a <= 0 < b
        result['regressed'] += b <= 0 < a
        result['before_ties'] += a == 0
        result['after_ties'] += b == 0
    differences = [after[key] - before[key] for key in before]
    result['reward_difference'] = summarize(differences)
    result['absolute_reward_difference'] = summarize(abs(v) for v in differences)
    return result


def reference_weights(pairs, reference, tau, beta):
    cutoff = math.log(tau / (1 - tau))
    gaps = [reference[p['chosen_id']] - reference[p['rejected_id']] for p in pairs]
    retained = [gap for gap in gaps if gap < cutoff]
    weights = [math.exp(min(gap / beta, 0)) for gap in retained]
    total = sum(weights)
    square_sum = sum(w * w for w in weights)
    return {'all_reference_gaps': summarize(gaps), 'retained_reference_gaps': summarize(retained),
            'retained_weights': summarize(weights),
            'effective_sample_size': total * total / square_sum if square_sum else 0.,
            'weights_below_1e-3': sum(w < 1e-3 for w in weights),
            'tau': tau, 'beta': beta}


def audit_adapter(config, adapter):
    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    kwargs = {'repo_id': config['model_name'], 'revision': config['model_revision']}
    try:
        index_path = hf_hub_download(filename='model.safetensors.index.json', **kwargs)
    except EntryNotFoundError:
        shard = hf_hub_download(filename='model.safetensors', **kwargs)
    else:
        index = json.loads(Path(index_path).read_text())
        shard_name = index['weight_map'].get('score.weight')
        if shard_name is None:
            raise ValueError('Pinned model index has no score.weight')
        shard = hf_hub_download(filename=shard_name, **kwargs)
    with safe_open(shard, framework='pt', device='cpu') as source:
        base = source.get_tensor('score.weight')
    entries = {}
    heads = []
    with safe_open(str(adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as source:
        for key in source.keys():
            tensor = source.get_tensor(key)
            entry = {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                     'elements': tensor.numel(), 'nonzero': torch.count_nonzero(tensor).item(),
                     'max_abs': tensor.float().abs().max().item(),
                     'l2_norm': torch.linalg.vector_norm(tensor.float()).item()}
            if 'score' in key and key.endswith('weight'):
                if tensor.shape != base.shape:
                    raise ValueError('Adapter score head has incompatible shape')
                delta = tensor.float() - base.float()
                quantized_base_delta = tensor.float() - base.to(tensor.dtype).float()
                entry.update(base_dtype=str(base.dtype), changed_from_base=torch.count_nonzero(delta).item(),
                             difference=summarize(delta.flatten().tolist()),
                             changed_from_base_cast_to_adapter_dtype=torch.count_nonzero(quantized_base_delta).item())
                heads.append(key)
            entries[key] = entry
    if not heads or not any('lora_B' in key for key in entries):
        raise ValueError('Saved adapter lacks reward head or LoRA B parameters')
    return {'pinned_base_shard': shard, 'score_head_keys': heads, 'parameters': entries,
            'lora_B_total_nonzero': sum(v['nonzero'] for k, v in entries.items() if 'lora_B' in k),
            'lora_B_total_elements': sum(v['elements'] for k, v in entries.items() if 'lora_B' in k)}


def run(config, baseline, pilot, output):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed
    data = json.loads((baseline / 'data.json').read_text())
    old = json.loads((pilot / 'report.json').read_text())
    baseline_report = json.loads((baseline / 'report.json').read_text())
    country, fmt = old['country'], old['input_format']
    for folder in (baseline, pilot):
        saved_config = json.loads((folder / 'manifest.json').read_text())['config']
        for key in ('model_name', 'model_revision', 'dataset_name', 'dataset_revision'):
            if config[key] != saved_config[key]:
                raise ValueError(f'{folder}: config mismatch for {key}')
        if config.get('max_length', 2048) != saved_config.get('max_length', 2048):
            raise ValueError('Length cap differs from saved experiment')
    pairs, rows = data['test'][country]['pairs'], data['test'][country]['rows']
    original_reference = baseline_report['formats'][fmt]['countries'][country]['rewards']
    report = {'country': country, 'input_format': fmt, 'config': config,
              'baseline_path': str(baseline.resolve()), 'pilot_path': str(pilot.resolve()),
              'original_precision_assumption': 'Original runner loads BF16 on CUDA; Trainer BF16 autocast, fresh scoring no explicit autocast. Historical paper precision unknown.',
              'diagnostic_policy': 'No tuning or selection; fixed heldout pairs, deterministic eval, original score batch size',
              'old_pilot_metrics': old['metrics'], 'old_baseline_metrics': old['baseline_metrics'],
              'adapter_audit': audit_adapter(config, pilot / 'adapter'),
              'reference_weights': reference_weights(data['train'][country]['pairs'], old['train_reference_rewards'],
                                                     config.get('tau', .5), config.get('beta', 1.)),
              'precisions': {}}
    write_json(output / 'report.json', report)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'])
    for precision in ('bfloat16', 'float32'):
        set_seed(0)
        current_config = dict(config, torch_dtype=precision)
        model = load_model(current_config, tokenizer)
        actual_dtype = str(next(model.parameters()).dtype)
        if actual_dtype != 'torch.' + precision:
            raise ValueError(f'load_model ignored precision {precision}: got {actual_dtype}')
        baseline_rewards, truncation = score(model, tokenizer, rows, fmt, config.get('max_length', 2048),
                                             config.get('score_batch_size', 8))
        model = PeftModel.from_pretrained(model, str(pilot / 'adapter'))
        adapter_rewards, _ = score(model, tokenizer, rows, fmt, config.get('max_length', 2048),
                                   config.get('score_batch_size', 8))
        report['precisions'][precision] = {
            'base_dtype': actual_dtype, 'truncation': truncation,
            'loaded_adapter_parameter_dtypes': {name: str(param.dtype) for name, param in model.named_parameters()
                                               if 'lora_' in name or 'modules_to_save' in name},
            'base_metrics': metrics(pairs, baseline_rewards),
            'adapter_metrics_original_reference': metrics(pairs, adapter_rewards, original_reference),
            'adapter_metrics_same_precision_reference': metrics(pairs, adapter_rewards, baseline_rewards),
            'base_to_adapter': compare_rewards(pairs, baseline_rewards, adapter_rewards),
            'old_report_to_reloaded_adapter': compare_rewards(pairs, old['rewards'], adapter_rewards),
            'original_baseline_to_current_base': compare_rewards(pairs, original_reference, baseline_rewards),
            'base_rewards': baseline_rewards, 'adapter_rewards': adapter_rewards}
        write_json(output / 'report.json', report)
        print(precision, report['precisions'][precision]['adapter_metrics_original_reference'], flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report['cross_precision_adapter'] = compare_rewards(pairs, report['precisions']['bfloat16']['adapter_rewards'],
                                                        report['precisions']['float32']['adapter_rewards'])
    report['cross_precision_base'] = compare_rewards(pairs, report['precisions']['bfloat16']['base_rewards'],
                                                     report['precisions']['float32']['base_rewards'])
    write_json(output / 'report.json', report)
    write_json(output / 'complete.json', {'complete': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    run(config, args.baseline, args.pilot, args.output)


if __name__ == '__main__':
    main()
