import ast
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.country_search import (BASE, CANDIDATES, COUNTRIES, ORIGINAL_HASHES,
                                   qualifies, read_metrics, search, trial_source, verify_sources)

REPO = Path(__file__).resolve().parents[1]


def reports(root, country, overall=50.0, misranked=20.0):
    folder = root / 'experiment/seed/0'
    folder.mkdir(parents=True, exist_ok=True)
    for filename, value in [('full_report.json', overall), ('partial_report.json', misranked)]:
        (folder / filename).write_text(json.dumps({country: dict(start=0.0, ours_partial=value)}))


class CountrySearchTests(unittest.TestCase):
    def test_only_approved_assignments_differ(self):
        original = (REPO / 'preferences.py').read_text()
        allowed = {'RANDOM_DATA_SHUFFLE_SEED', 'EDITED_LR', 'REWARD_THRES', 'WEIGHTING_BETA'}

        def canonical(source):
            tree = ast.parse(source)
            assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                           and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)]
            last_country = max(n.lineno for n in assignments if n.targets[0].id == 'countries')
            for node in assignments:
                if node.targets[0].id in allowed or node.lineno == last_country:
                    node.value = ast.Constant(value='approved')
            return ast.dump(tree)

        for country in COUNTRIES:
            for config in CANDIDATES:
                generated = trial_source(original, country, config)
                self.assertEqual(canonical(original), canonical(generated))
        with self.assertRaises(ValueError):
            trial_source(original + '# unapproved drift\n', 'Chile', BASE)
        verify_sources(REPO)

    def test_validated_threshold_and_nonfinite_values(self):
        self.assertTrue(qualifies('Chile', {'overall': 59.89 - .66}, True))
        self.assertFalse(qualifies('Chile', {'overall': 59.22}, True))
        self.assertFalse(qualifies('Chile', {'overall': 99.}, False))
        for value in (math.nan, math.inf, -1., 101., None, True):
            self.assertFalse(qualifies('Chile', {'overall': value}, True))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reports(root, 'Chile', misranked=math.nan)
            with self.assertRaises(ValueError):
                read_metrics(root, 'Chile')

    def test_stop_first_validated_match_and_preserve_all_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo, baseline, output = root / 'source', root / 'baseline', root / 'search'
            repo.mkdir()
            for name in ORIGINAL_HASHES:
                shutil.copyfile(REPO / name, repo / name)
            reports(baseline, 'Chile')
            validation = baseline / 'trunc/Chile/ours_3epoch/validation.json'
            validation.parent.mkdir(parents=True)
            validation.write_text('{"deterministic": true}')
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                if command[1].endswith('preferences_trial.py'):
                    path = Path(command[2])
                    # The control already meets the score, but fails validation.
                    reports(path, 'Chile', overall=60.)
                    return SimpleNamespace(returncode=0)
                return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)

            with patch('scripts.country_search.subprocess.run', side_effect=fake_run), \
                 patch('scripts.country_search.importlib.metadata.distributions', return_value=[]):
                search(repo, output, baseline, 'Chile')
            state = json.loads((output / 'chile/status.json').read_text())
            self.assertEqual(state['status'], 'matched')
            self.assertEqual(state['selected_config']['name'], 'shuffle-1')
            self.assertEqual(len(state['trials']), 2)
            self.assertEqual(state['trials'][0]['status'], 'failed_validation')
            self.assertEqual(state['trials'][1]['metrics']['misranked'], 20.)
            self.assertEqual(len(calls), 4)
            # A repeated submission cannot overwrite the successful search.
            with self.assertRaises(FileExistsError):
                search(repo, output, baseline, 'Chile')

    def test_keep_existing_close_country_without_launching_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = root / 'baseline'
            reports(baseline, 'Canada', overall=66.18)
            validation = baseline / 'trunc/Canada/ours_3epoch/validation.json'
            validation.parent.mkdir(parents=True)
            validation.write_text('{"deterministic": true}')
            with patch('scripts.country_search.subprocess.run') as run:
                search(REPO, root / 'search', baseline, 'Canada')
                run.assert_not_called()
            state = json.loads((root / 'search/canada/status.json').read_text())
            self.assertEqual(state['status'], 'matched_existing')


if __name__ == '__main__':
    unittest.main()
