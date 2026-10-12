"""Publish the explicitly authorized MulticulturalAlignment DeBERTa release."""
import argparse
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from dotenv import load_dotenv

from prepare_deberta_release import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--release-files', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    load_dotenv('.env')
    api = HfApi()
    summary = json.loads((args.packages / 'release-summary.json').read_text())
    if len(summary) != 7 or not all(item['validated'] for item in summary.values()):
        raise ValueError('All seven packages must pass load validation before publishing')
    journal_path = args.packages / 'publication.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    # Check all destinations and all package hashes before uploading anything.
    for country, item in summary.items():
        package = args.packages / item['package']
        repo_id = 'MulticulturalAlignment/' + package.name
        if digest(package / 'model.safetensors') != item['weight_sha256']:
            raise ValueError(f'Weights changed after validation: {country}')
        validation = json.loads((package / 'validation.json').read_text())
        if not validation['deterministic']:
            raise ValueError(f'Invalid export: {country}')
        try:
            api.model_info(repo_id)
        except RepositoryNotFoundError:
            pass
        else:
            if repo_id not in journal:
                raise ValueError(f'Refusing to overwrite an existing untracked model: {repo_id}')
    for country, item in summary.items():
        package = args.packages / item['package']
        repo_id = 'MulticulturalAlignment/' + package.name
        if journal.get(repo_id, {}).get('public'):
            continue
        card = args.release_files / 'model-cards' / (country.lower().replace(' ', '-') + '.md')
        text = card.read_text()
        if 'YOUR_NAMESPACE' in text or 'license: cc-by-nc-4.0' not in text or repo_id not in text:
            raise ValueError(f'Unfinalized model card: {country}')
        shutil.copyfile(card, package / 'README.md')
        shutil.copyfile(args.release_files / 'LICENSE', package / 'LICENSE')
        run = args.runs / country.lower().replace(' ', '-')
        manifest = json.loads((run / 'manifest.json').read_text())
        source = run / 'preferences_trial.py'
        if digest(source) != manifest['generated_sha256']:
            raise ValueError(f'Generated source changed: {country}')
        reproduction = package / 'reproduction'
        reproduction.mkdir(exist_ok=True)
        shutil.copyfile(source, reproduction / 'preferences.py')
        repo = Path(__file__).resolve().parents[1]
        for name in ('our_reward_trainer.py', 'our_data_collator.py', 'our_inverse_reward_trainer.py'):
            if digest(repo / name) != manifest['original_sha256'][name]:
                raise ValueError(f'Original source changed: {name}')
            shutil.copyfile(repo / name, reproduction / name)
        hashes = {p.relative_to(package).as_posix(): digest(p) for p in package.rglob('*')
                  if p.is_file() and p.name != 'checksums.json'}
        write_json(package / 'checksums.json', hashes)
        if repo_id not in journal:
            api.create_repo(repo_id, repo_type='model', private=True, exist_ok=False)
            journal[repo_id] = {'created': True, 'public': False}
            write_json(journal_path, journal)
        print(f'Uploading {repo_id}', flush=True)
        commit = api.upload_folder(repo_id=repo_id, folder_path=str(package), repo_type='model',
                                   allow_patterns=list(hashes) + ['checksums.json'],
                                   commit_message='Release reproduced country-specific SCPO DeBERTa checkpoint')
        info = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
        remote = {f.rfilename: f for f in info.siblings}
        weights = remote['model.safetensors']
        if weights.lfs is None or weights.lfs.sha256 != item['weight_sha256']:
            raise ValueError(f'Uploaded weights do not match validated checkpoint: {repo_id}')
        for name in list(hashes) + ['checksums.json']:
            if name not in remote or remote[name].size != (package / name).stat().st_size:
                raise ValueError(f'Uploaded file missing or wrong size: {repo_id}/{name}')
        journal[repo_id].update(commit=commit.oid, upload_verified=True)
        write_json(journal_path, journal)
    # Only expose repositories after all seven uploads have been verified.
    for repo_id, record in journal.items():
        if not record.get('upload_verified'):
            raise ValueError(f'Upload not verified: {repo_id}')
        if not record.get('public'):
            api.update_repo_visibility(repo_id, private=False, repo_type='model')
            if api.model_info(repo_id).private:
                raise ValueError(f'Repository still private: {repo_id}')
            record['public'] = True
            write_json(journal_path, journal)
        print(f'Published https://huggingface.co/{repo_id}', flush=True)


if __name__ == '__main__':
    main()
