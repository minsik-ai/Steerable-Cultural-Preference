import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.country_search import BASE, search, trial_source
from scripts.run_country_defaults import DEFAULT_SETTINGS, REPO, load_settings


class CountryDefaultsTests(unittest.TestCase):
    def test_defaults_render_all_recorded_settings(self):
        settings = load_settings(DEFAULT_SETTINGS)
        for country, entry in settings['countries'].items():
            source = trial_source((REPO / 'preferences.py').read_text(), country, entry)
            self.assertIn(f'EDITED_LR = {entry["learning_rate"]!r}', source)
            self.assertIn(f'REWARD_THRES = {entry["tau"]!r}', source)
            self.assertIn(f'WEIGHTING_BETA = {entry["beta"]!r}', source)
            self.assertIn(f'RANDOM_DATA_SHUFFLE_SEED = {entry["shuffle_seed"]}', source)
        self.assertEqual(settings['countries']['Mexico']['learning_rate'], 1e-5)
        for country in ('Australia', 'Israel'):
            self.assertEqual(settings['countries'][country]['status'], 'provisional')

    def test_refuse_invalid_settings_and_execution_drift(self):
        original = load_settings(DEFAULT_SETTINGS)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            for key, value in [('trainer_seed', 0), ('max_steps', 512),
                               ('model', 'different/model'), ('dataset', 'different/data'),
                               ('experiment_source_revision', 'different-source')]:
                data = copy.deepcopy(original)
                data[key] = value
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_settings(path)
            data = copy.deepcopy(original)
            data['countries']['Chile']['tau'] = 1.1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_settings(path)

    def test_default_runs_without_baseline_or_paper_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'

            def fake_run(command, **kwargs):
                if command[1].endswith('preferences_trial.py'):
                    report = Path(command[2]) / 'experiment/seed/0'
                    report.mkdir(parents=True)
                    for name in ('full_report.json', 'partial_report.json'):
                        (report / name).write_text(json.dumps({'Chile': {'start': 0., 'ours_partial': 55.}}))
                return SimpleNamespace(returncode=0)

            with patch('scripts.country_search.subprocess.run', side_effect=fake_run) as run, \
                 patch('scripts.country_search.importlib.metadata.distributions', return_value=[]):
                search(REPO, root, None, 'Chile', candidates=[dict(BASE, name='default')], require_match=False)
            result = json.loads((root / 'chile/status.json').read_text())
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['selected_metrics']['overall'], 55.)
            self.assertEqual(run.call_count, 2)

    def test_continuation_only_targets_remaining_countries_without_duplicates(self):
        from scripts.country_search import CANDIDATES
        plan = json.loads((REPO / 'configs/remaining_search.json').read_text())
        self.assertEqual(plan['countries'], ['Australia', 'Israel'])
        signature = lambda config: tuple(config[key] for key in BASE)
        old = {signature(c) for c in CANDIDATES}
        new = [signature(c) for c in plan['candidates']]
        self.assertEqual(len(new), len(set(new)))
        self.assertTrue(old.isdisjoint(new))


if __name__ == '__main__':
    unittest.main()
