"""Exercise the real trainer/report/export path offline with synthetic data.

This is a software integration test, not a PRISM/model reproduction.
"""
import json
import tempfile
import unittest
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch
from datasets import Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


class MainSmokeTests(unittest.TestCase):
    def test_training_reporting_and_export(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'tiny-fixture'
            base.mkdir()
            (base / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nquestion\ngood\nbad\n')
            tokenizer = BertTokenizer(str(base / 'vocab.txt'))
            tokenizer.save_pretrained(base)
            model = BertForSequenceClassification(BertConfig(
                vocab_size=8, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, num_labels=1,
                max_position_embeddings=512))
            model.eval()
            inputs = tokenizer(['question', 'question'], ['good', 'bad'], return_tensors='pt')
            with torch.inference_mode():
                scores = model(**inputs).logits.flatten()
            self.assertNotEqual(scores[0].item(), scores[1].item())
            # Oppose the reference model so edited train/test subsets are nonempty.
            preferred = int(scores.argmin())
            model.save_pretrained(base)
            surveys, utterances = [], []
            countries = ['Chile', 'South Africa', 'New Zealand', 'Australia', 'Mexico', 'Israel', 'Canada']
            for country in countries:
                for user_index in range(20):
                    user = f'{country}-{user_index}'
                    surveys.append(dict(user_id=user, location=dict(
                        reside_country=country, birth_country=country)))
                    for answer_index, answer in enumerate(('good', 'bad')):
                        utterances.append(dict(user_id=user, interaction_id=user,
                                               utterance_id=f'{user}-{answer_index}',
                                               user_prompt='question', model_response=answer,
                                               score=1 if answer_index == preferred else 0))
            data = {'survey': Dataset.from_list(surveys), 'utterances': Dataset.from_list(utterances)}
            output = root / 'run'
            # Load the original CLI unchanged. Only this synthetic test substitutes
            # its local fixture model/data and one-step training budget.
            with patch.object(sys, 'argv', ['preferences.py', str(output)]):
                namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'preferences.py'))
            main = namespace['main']
            main.__globals__['MODEL_NAME'] = str(base)
            main.__globals__['MAX_STEPS'] = {country: 1 for country in countries}
            with patch('datasets.load_dataset', side_effect=lambda name, config, **kw: data[config]):
                main(str(output))
            report = json.loads((output / 'experiment/seed/0/full_report.json').read_text())
            self.assertEqual(set(report), set(countries) | {'avg'})
            self.assertTrue(all('ours_partial' in report[country] for country in countries))
            partial = json.loads((output / 'experiment/seed/0/partial_report.json').read_text())
            self.assertEqual(partial['Chile']['start'], 0)
            self.assertIn('ours_partial', partial['Chile'])
            checkpoint = output / 'trunc/Chile/ours_3epoch'
            reloaded = BertForSequenceClassification.from_pretrained(checkpoint).eval()
            with torch.inference_mode():
                self.assertTrue(torch.isfinite(reloaded(**inputs).logits).all())


if __name__ == '__main__':
    unittest.main()
