"""Compare only completed runs sharing the restored source-backed protocol."""
import argparse
import hashlib
import json
from pathlib import Path

COUNTRIES=['Chile','South Africa','New Zealand','Australia','Mexico','Israel','Canada']


def split_ids(run):
    import pyarrow.ipc as ipc
    result={}
    for part in ('train','test'):
        files=sorted((run/'dataset/PRISM/seed/0'/part).glob('*.arrow'))
        if not files:
            raise ValueError('Missing saved survey split')
        ids=[]
        for path in files:
            with ipc.open_stream(str(path)) as reader:
                ids.extend(reader.read_all().column('user_id').to_pylist())
        result[part]=ids
    if set(result['train']) & set(result['test']):
        raise ValueError('Train/test users overlap')
    return result


def summarize(root):
    found={};missing=[]
    for family in ('deberta','tulu'):
        for country in COUNTRIES:
            folder=root/family/country.lower().replace(' ','-')
            path=folder/'manifest.json'
            if not path.exists():
                missing.append(f'{family}/{country}')
                continue
            manifest=json.loads(path.read_text())
            if manifest['status']!='completed':
                missing.append(f'{family}/{country}: {manifest["status"]}')
                continue
            if manifest.get('protocol')!='original-source-v1':
                raise ValueError('Refusing to compare reconstructed/deterministic protocol results')
            found[family,country]=(folder,manifest)
    if missing:
        return dict(complete=False,missing=missing,completed=len(found))
    _,reference=next(iter(found.values()))
    reference_ids=None;reference_environment=None;dataset_revision=None;model_revisions={}
    for (family,country),(folder,manifest) in found.items():
        for key in ('protocol','original_sha256','primary_evaluation','process_semantics'):
            if manifest[key]!=reference[key]:
                raise ValueError(f'Protocol mismatch for {family}/{country}: {key}')
        if hashlib.sha256((folder/'preferences_trial.py').read_bytes()).hexdigest()!=manifest['generated_sha256']:
            raise ValueError('Generated source changed since execution')
        index=COUNTRIES.index(country)+(7 if family=='tulu' else 0)
        runtime=json.loads((root/f'runtime-{index}.json').read_text())['verified'][0]
        if runtime['model']!=manifest['config']['model_name']:
            raise ValueError('Runtime model identity mismatch')
        if dataset_revision is None:
            dataset_revision=runtime['dataset_revision']
        elif dataset_revision!=runtime['dataset_revision']:
            raise ValueError('Runtime dataset revision mismatch')
        if family in model_revisions and model_revisions[family]!=runtime['model_revision']:
            raise ValueError('Model revision differs within one family')
        model_revisions[family]=runtime['model_revision']
        environment=(folder/'environment.txt').read_text()
        ids=split_ids(folder/'run')
        if reference_ids is None:
            reference_ids,reference_environment=ids,environment
        elif ids!=reference_ids or environment!=reference_environment:
            raise ValueError('Split or runtime differs between compared runs')
    rows={}
    for country in COUNTRIES:
        d=found['deberta',country][1];t=found['tulu',country][1]
        rows[country]=dict(deberta=d['primary_metrics'],tulu=t['primary_metrics'],
            tulu_minus_deberta_pp=t['primary_metrics']['overall']-d['primary_metrics']['overall'],
            deberta_config=d['config'],tulu_config=t['config'])
    return dict(complete=True,protocol=reference['protocol'],countries=rows,
                macro_accuracy={family:sum(found[family,c][1]['primary_metrics']['overall'] for c in COUNTRIES)/7
                                for family in ('deberta','tulu')},
                limitations=['Selected hyperparameters and model parameterizations differ',
                             'Single-country processes; not historical seven-country RNG history',
                             'Original Tulu paper recipe remains unverified; raw original inputs are used'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.root)
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
