"""Exercise the complete audited Tulu train/export/reload path with a tiny model."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TuluTrainingRunTests(unittest.TestCase):
    def test_fp32_training_audit_and_fresh_reload_metrics(self):
        import torch
        from transformers import BertTokenizer, LlamaConfig, LlamaForSequenceClassification
        from scripts.tulu_replication import train, score
        torch.set_num_threads(2)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'base'
            base.mkdir()
            (base / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nquestion\none\ntwo\n')
            tokenizer = BertTokenizer(str(base / 'vocab.txt'))
            tokenizer.save_pretrained(base)
            model = LlamaForSequenceClassification(LlamaConfig(vocab_size=8, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2, num_labels=1, pad_token_id=0))
            model.save_pretrained(base)
            rows = [dict(utterance_id=str(i), user_prompt='question', model_response=text)
                    for i, text in enumerate(('one', 'two'))]
            reference, _ = score(model, tokenizer, rows, 'legacy-pair', 32, 2)
            order = sorted(reference, key=reference.get)
            self.assertLess(reference[order[0]], reference[order[1]])
            pairs = [dict(chosen_id=order[0], rejected_id=order[1])]
            cfg = dict(model_name=str(base), model_revision='0'*40, dataset_name='fixture',
                       dataset_revision='0'*40, torch_dtype='float32', max_length=32,
                       batch_size=1, score_batch_size=2, learning_rate=1e-4, max_steps=1,
                       tau=.5, beta=1., gradient_checkpointing=True,
                       lora=dict(r=2, lora_alpha=4, lora_dropout=.1,
                                 target_modules=['q_proj', 'v_proj'], modules_to_save=['score']))
            baseline = root / 'baseline'
            baseline.mkdir()
            (baseline / 'manifest.json').write_text(json.dumps({'config': cfg}))
            (baseline / 'data.json').write_text(json.dumps({
                part: {'Chile': {'rows': rows, 'pairs': pairs}} for part in ('train', 'test')}))
            (baseline / 'report.json').write_text(json.dumps({'formats': {'legacy-pair': {
                'countries': {'Chile': {'rewards': reference}}}}}))
            output = root / 'output'
            output.mkdir()
            with patch('scripts.tulu_replication.load_model', side_effect=lambda *a: LlamaForSequenceClassification.from_pretrained(base)):
                train(cfg, output, baseline, 'Chile', 'legacy-pair')
            result = json.loads((output / 'report.json').read_text())
            audit = result['training_update_audit']
            self.assertFalse(result['bf16_autocast'])
            self.assertGreater(audit['score_head']['first_observed_grad_l2'], 0)
            self.assertGreater(audit['score_head']['changed_elements'], 0)
            self.assertGreater(audit['lora_B']['changed_elements'], 0)
            self.assertEqual(result['evaluation_mode'], 'fresh adapter reload, eval, no autocast')
            self.assertEqual(result['reload_pair_ranking_changes'], 0)
            self.assertTrue(json.loads((output / 'validation.json').read_text())['passed'])
