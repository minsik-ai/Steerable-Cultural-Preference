"""Bounded South Africa search at fixed LR, with equal-step full-data checks."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_result(folder, country, method, steps):
    for name in ('complete.json', 'validation.json', 'report.json', 'manifest.json'):
        if not (folder / name).exists():
            raise ValueError(f'Incomplete result: {folder}/{name}')
    if not json.loads((folder/'complete.json').read_text()).get('complete'):
        raise ValueError('Run not complete')
    if not json.loads((folder/'validation.json').read_text()).get('passed'):
        raise ValueError('Adapter did not validate')
    report = json.loads((folder/'report.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    config = manifest['config']
    if (report['country'] != country or report['method'] != method or report['actual_global_step'] != steps
            or config['learning_rate'] != 1e-6 or config.get('shuffle_seed', 0) != 0
            or config.get('torch_dtype') != 'float32' or report['input_format'] != 'chat-template'
            or report['evaluation_mode'] != 'fresh adapter reload, eval, no autocast'):
        raise ValueError('Result does not match fixed South Africa protocol')
    for metric in (report['metrics'], report['metrics']['reference_misranked']):
        value = metric['accuracy']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Invalid metric')
    return {'directory': str(folder), 'metrics': report['metrics'], 'baseline_metrics': report['baseline_metrics'],
            'config': config, 'manifest': manifest}


def comparable(result, reference):
    for key in ('model_name', 'model_revision', 'dataset_name', 'dataset_revision', 'lora',
                'batch_size', 'max_length', 'score_batch_size', 'gradient_checkpointing'):
        if result['config'][key] != reference['config'][key]:
            raise ValueError(f'Comparison config drift: {key}')
    for key in ('python', 'packages', 'runner_sha256', 'model_seed', 'trainer_seed', 'survey_split_seed'):
        if result['manifest'][key] != reference['manifest'][key]:
            raise ValueError(f'Comparison runtime drift: {key}')
    if result['baseline_metrics'] != reference['baseline_metrics']:
        raise ValueError('Starting scores or reference subset drifted')


def beats(score, starting, full_data):
    return math.isfinite(score) and score > max(starting, full_data) + 1e-9


def search(source, artifacts, output, plan):
    if plan['country'] != 'South Africa' or plan['learning_rate'] != 1e-6 or plan['shuffle_seed'] != 0:
        raise ValueError('This search is South Africa only with LR1e-6/shuffle0')
    country = plan['country']
    original_full = load_result(artifacts/plan['full_data'], country, 'full_data', 1024)
    starting = original_full['baseline_metrics']['accuracy']
    initial_full = original_full['metrics']['accuracy']
    controls = {1024: original_full}
    output.mkdir(parents=True, exist_ok=False)
    state_path = output/'status.json'
    state = dict(status='running', country=country, learning_rate=1e-6, starting_accuracy=starting,
                 initial_full_data_accuracy=initial_full, best_full_data_accuracy=initial_full,
                 trials=[], full_data_controls={'1024': {'directory': original_full['directory'],
                    'metrics': original_full['metrics']}}, interpretation='single-seed exploratory search')
    write_json(output/'plan.json', plan)
    write_json(state_path, state)
    base = json.loads((source/plan['base_config']).read_text())

    def execute(name, config):
        path=output/f'{name}.json'
        with path.open('x') as stream:
            json.dump(config, stream, indent=2)
        folder=output/'runs'/name
        with (output/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,str(source/'scripts/tulu_replication.py'),'train',
                '--config',str(path),'--output',str(folder),'--baseline',str(artifacts/plan['baseline']),
                '--country',country,'--format',plan['input_format']],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=load_result(folder,country,config['method'],config['max_steps'])
        comparable(result,original_full)
        return result

    try:
        for candidate in plan['candidates']:
            config=dict(base, **{k:v for k,v in candidate.items() if k!='name'})
            config.update(learning_rate=1e-6,shuffle_seed=0)
            entry=dict(candidate=candidate,status='running')
            state['trials'].append(entry)
            write_json(state_path,state)
            result=execute(candidate['name'],config)
            entry.update(status='completed',directory=result['directory'],metrics=result['metrics'])
            value=result['metrics']['accuracy']
            entry['delta_vs_starting']=value-starting
            entry['delta_vs_initial_full_data']=value-initial_full
            write_json(state_path,state)
            if beats(value,starting,state['best_full_data_accuracy']):
                steps=candidate['max_steps']
                if steps not in controls:
                    control=dict(base,method='full_data',learning_rate=1e-6,max_steps=steps,shuffle_seed=0)
                    control_result=execute(f'full-data-{steps}',control)
                    controls[steps]=control_result
                    state['full_data_controls'][str(steps)]={'directory':control_result['directory'],
                                                            'metrics':control_result['metrics']}
                    state['best_full_data_accuracy']=max(r['metrics']['accuracy'] for r in controls.values())
                entry['equal_step_full_data_accuracy']=controls[steps]['metrics']['accuracy']
                write_json(state_path,state)
                if beats(value,starting,state['best_full_data_accuracy']):
                    state.update(status='matched',selected=candidate,selected_metrics=result['metrics'],
                                 selected_checkpoint=str(Path(result['directory'])/'adapter'))
                    write_json(state_path,state)
                    return
        state['status']='exhausted'
        best=max(state['trials'],key=lambda t:t['metrics']['accuracy'])
        state['best_trial']=best
        state['best_beats_starting']=best['metrics']['accuracy']>starting
        write_json(state_path,state)
    except Exception as error:
        state.update(status='blocked',error=str(error))
        write_json(state_path,state)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((args.source/'configs/tulu_south_africa.json').read_text())
    search(args.source.resolve(),args.artifacts.resolve(),args.output.resolve(),plan)
