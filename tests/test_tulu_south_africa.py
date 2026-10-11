import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.tulu_south_africa import beats, search


class SouthAfricaSearchTests(unittest.TestCase):
    def test_must_beat_both_references_strictly(self):
        self.assertFalse(beats(64., 63., 64.))
        self.assertFalse(beats(63.5, 63., 64.))
        self.assertTrue(beats(64.1, 63., 64.))
        self.assertFalse(beats(float('nan'), 63., 64.))

    def test_equal_budget_control_prevents_premature_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'source';source.mkdir()
            (source/'base.json').write_text(json.dumps({'learning_rate':1e-6,'shuffle_seed':0,'tau':.5,'beta':1.}))
            plan=dict(country='South Africa',learning_rate=1e-6,shuffle_seed=0,base_config='base.json',
                      baseline='baseline',full_data='established',input_format='chat-template',
                      candidates=[dict(name='short',method='filtered_weighted',max_steps=128),
                                  dict(name='medium',method='filtered_weighted',max_steps=256),
                                  dict(name='never-run',method='weighted_only',max_steps=1024)])
            values={'established':64.,'short':65.,'full-data-128':66.,'medium':67.,'full-data-256':66.5}
            def fake_result(path,*args):
                return {'directory':str(path),'metrics':{'accuracy':values[path.name]},
                        'baseline_metrics':{'accuracy':63.},'config':{},'manifest':{}}
            with patch('scripts.tulu_south_africa.load_result',side_effect=fake_result), \
                 patch('scripts.tulu_south_africa.comparable'), \
                 patch('scripts.tulu_south_africa.subprocess.run') as run:
                search(source,root,root/'output',plan)
            state=json.loads((root/'output/status.json').read_text())
            self.assertEqual(state['status'],'matched')
            self.assertEqual(state['selected']['name'],'medium')
            self.assertEqual(state['best_full_data_accuracy'],66.5)
            self.assertEqual(len(state['trials']),2)
            self.assertEqual(run.call_count,4)
            for call in run.call_args_list:
                command=call.args[0]
                self.assertEqual(command[command.index('--country')+1],'South Africa')
                config=json.loads(Path(command[command.index('--config')+1]).read_text())
                self.assertEqual(config['learning_rate'],1e-6)
            with self.assertRaises(FileExistsError):
                with patch('scripts.tulu_south_africa.load_result',side_effect=fake_result):
                    search(source,root,root/'output',plan)

    def test_plan_freezes_other_countries_and_lr(self):
        root=Path(__file__).resolve().parents[1]
        plan=json.loads((root/'configs/tulu_south_africa.json').read_text())
        self.assertEqual(plan['country'],'South Africa')
        self.assertEqual(plan['learning_rate'],1e-6)
        self.assertEqual(len(plan['candidates']),6)
        self.assertEqual(len(set(plan['frozen_other_countries'])),6)
        self.assertNotIn('South Africa',plan['frozen_other_countries'])


class AustraliaPlanTests(unittest.TestCase):
    def test_australia_has_four_new_settings_at_fixed_lr(self):
        root=Path(__file__).resolve().parents[1]
        plan=json.loads((root/'configs/tulu_australia.json').read_text())
        self.assertEqual(plan['country'],'Australia')
        self.assertEqual(plan['learning_rate'],1e-6)
        self.assertEqual([c['name'] for c in plan['candidates']],
                         ['filtered-128','filtered-256','filter-07-1024','weighted-beta2'])
        self.assertNotIn('Australia',plan['frozen_other_countries'])
        self.assertIn('South Africa',plan['frozen_other_countries'])
        self.assertIn('australia-lr1e-06-full_data',plan['full_data'])

    def test_unrequested_country_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError):
                search(root,root,root/'output',dict(country='Canada',learning_rate=1e-6,shuffle_seed=0))
