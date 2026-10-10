import json
from pathlib import Path
import tempfile
import unittest
from scripts.tulu_method_comparison import METHODS, cases, summarize


class ComparisonTests(unittest.TestCase):
    def test_complete_grid_compares_matched_and_best_full_data(self):
        plan = dict(countries=['Chile', 'Canada'], methods=list(METHODS), learning_rates=[1e-6, 1e-5],
                    max_steps=1024, shuffle_seed=0, precision='float32', input_format='chat-template',
                    interpretation='test fixture')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for case in cases(plan):
                path = root / 'runs' / case['name']
                path.mkdir(parents=True)
                low = case['learning_rate'] == 1e-6
                score = {'full_data': 60. if low else 70., 'weighted_only': 64. if low else 71.,
                         'filtered_weighted': 65. if low else 68.}[case['method']]
                config = dict(case, torch_dtype='float32', model_name='fixture', model_revision='r',
                              dataset_name='fixture', dataset_revision='d', lora={'r': 8},
                              batch_size=8, score_batch_size=8, max_length=2048, gradient_checkpointing=True, tau=.5, beta=1.)
                manifest = dict(config=config, packages={'torch': 'fixture'}, runner_sha256='same',
                                python='3.11', model_seed=0, trainer_seed=42, survey_split_seed=0)
                report = dict(country=case['country'], method=case['method'], input_format='chat-template',
                              actual_global_step=1024, evaluation_mode='fresh adapter reload, eval, no autocast',
                              metrics=dict(accuracy=score, pairs=100, reference_misranked=dict(accuracy=20., pairs=40)),
                              baseline_metrics=dict(correct=60), train_pairs=2000, retained_train_pairs=2000,
                              actual_epochs=4.096, loss_weighting='unit' if case['method'] == 'full_data' else 'SCPO clipped exponential')
                for name, data in [('complete.json', {'complete': True}), ('validation.json', {'passed': True}),
                                   ('manifest.json', manifest), ('report.json', report)]:
                    (path / name).write_text(json.dumps(data))
            result = summarize(root, plan)
            self.assertTrue(result['complete'])
            self.assertEqual(result['best_full_data_macro'], 70.)
            low_weighted = next(x for x in result['countries']['Chile']['settings']
                                if x['learning_rate'] == 1e-6 and x['method'] == 'weighted_only')
            self.assertEqual(low_weighted['delta_vs_matched_full_data'], 4.)
            self.assertEqual(low_weighted['delta_vs_best_tested_full_data'], -6.)
            first = root / 'runs' / cases(plan)[0]['name'] / 'complete.json'
            first.unlink()
            partial = summarize(root, plan)
            self.assertFalse(partial['complete'])
            self.assertNotIn('matched_budgets', partial)

    def test_shared_runtime_mismatch_is_rejected_in_case_read(self):
        from scripts.tulu_method_comparison import read_case
        plan = dict(countries=['Chile'], methods=list(METHODS), learning_rates=[1e-6], max_steps=1024,
                    shuffle_seed=0, precision='float32', input_format='chat-template')
        case = cases(plan)[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runs' / case['name']
            path.mkdir(parents=True)
            for name, data in [('complete.json', {'complete': True}), ('validation.json', {'passed': True}),
                               ('manifest.json', {'config': dict(case, learning_rate=1e-4)}), ('report.json', {})]:
                (path / name).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                read_case(Path(folder), case, plan)
