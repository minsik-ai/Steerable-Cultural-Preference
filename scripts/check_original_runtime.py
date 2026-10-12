"""Preflight outside the experiment process; never patches original model/data loaders."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform


def check(configs, lock):
    if platform.python_version() != '3.11.16':
        raise ValueError('Use the recorded Python 3.11.16 environment')
    for line in lock.read_text().splitlines():
        if line and not line.startswith('#'):
            name, version = line.split('==')
            if importlib.metadata.version(name) != version:
                raise ValueError(f'Package drift: expected {line}')
    from huggingface_hub import HfApi
    api=HfApi()
    records=[]
    for path in configs:
        config=json.loads(path.read_text())
        model=config.get('model_name',config.get('model'))
        dataset=config.get('dataset_name',config.get('dataset','HannahRoseKirk/prism-alignment'))
        model_sha=api.model_info(model).sha
        dataset_sha=api.dataset_info(dataset).sha
        if model_sha != config['model_revision'] or dataset_sha != config['dataset_revision']:
            raise ValueError('Hub main changed; refuse to alter the original loaders or use mismatched inputs')
        records.append(dict(config=str(path),model=model,model_revision=model_sha,dataset=dataset,dataset_revision=dataset_sha))
    return dict(python=platform.python_version(),original_loaders_unmodified=True,verified=records)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',type=Path,nargs='+',required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=check(args.configs,args.lock)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
