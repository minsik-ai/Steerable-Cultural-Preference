"""Offline checks for reconstructed Tulu pairing, formatting, and adapter lifecycle."""
import tempfile
import unittest
from pathlib import Path
from scripts.tulu_reconstructed import build_data, encode_rows, metrics


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return '|'.join(m['role'] + ':' + m['content'] for m in messages)

    def __call__(self, texts, pairs=None, **kwargs):
        self.calls.append((texts, pairs, kwargs))
        lengths = [len(t) + (len(pairs[i]) if pairs else 0) for i, t in enumerate(texts)]
        if kwargs.get('truncation'):
            lengths = [min(n, kwargs['max_length']) for n in lengths]
        return {'input_ids': [[1] * n for n in lengths], 'attention_mask': [[1] * n for n in lengths]}


class TuluReplicationTests(unittest.TestCase):
    def test_pairs_stay_within_user_interaction_and_drop_ties(self):
        survey = {'train': [{'user_id': 'a', 'location': {'reside_country': 'Chile'}}],
                  'test': [{'user_id': 'b', 'location': {'reside_country': 'Chile'}}]}
        rows = [dict(user_id=u, interaction_id='same', utterance_id=str(i), user_prompt='p',
                     model_response='a', score=s) for i, (u, s) in enumerate([('a', 1), ('a', 2),
                                                                           ('a', 2), ('b', 3), ('b', 1)])]
        rows.append(dict(user_id='not-surveyed', interaction_id='same', utterance_id='orphan',
                         user_prompt='p', model_response='a', score=1))
        result = build_data(survey, rows)
        self.assertEqual(result['_diagnostics']['utterances_without_survey_user'], 1)
        self.assertEqual(len(result['train']['Chile']['pairs']), 2)
        self.assertEqual(result['test']['Chile']['pairs'], [{'chosen_id': '3', 'rejected_id': '4'}])
        survey['test'].append(survey['train'][0])
        with self.assertRaises(ValueError):
            build_data(survey, rows)

    def test_formats_and_truncation_are_explicit(self):
        tokenizer = FakeTokenizer()
        rows = [{'user_prompt': 'prompt', 'model_response': 'answer'}]
        features, count = encode_rows(tokenizer, rows, 'legacy-pair', 4)
        self.assertEqual(tokenizer.calls[0][1], ['answer'])
        self.assertEqual(len(features[0]['input_ids']), 4)
        self.assertEqual(count['truncated_rows'], 1)
        encode_rows(tokenizer, rows, 'chat-template', 100)
        self.assertFalse(tokenizer.calls[-1][2]['add_special_tokens'])
        self.assertEqual(tokenizer.calls[-1][0], ['user:prompt|assistant:answer'])

    def test_reference_ties_are_not_misranked(self):
        pairs = [{'chosen_id': 'a', 'rejected_id': 'b'}, {'chosen_id': 'c', 'rejected_id': 'd'}]
        result = metrics(pairs, dict(a=1, b=0, c=1, d=0), dict(a=0, b=0, c=-1, d=1))
        self.assertEqual(result['accuracy'], 100)
        self.assertEqual(result['reference_misranked']['pairs'], 1)

    def test_tiny_lora_weighted_training_save_reload(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        from transformers import LlamaConfig, LlamaForSequenceClassification, BertTokenizer
        from datasets import Dataset
        from trl import RewardConfig
        from our_reward_trainer import OurRewardTrainer
        from our_data_collator import OurRewardDataCollatorWithPadding
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
            tokenizer = BertTokenizer(str(root / 'vocab.txt'))
            cfg = LlamaConfig(vocab_size=7, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                              num_attention_heads=2, num_key_value_heads=2, num_labels=1, pad_token_id=0)
            base = LlamaForSequenceClassification(cfg)
            base.save_pretrained(root / 'base')
            model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=2, lora_alpha=4,
                                                  lora_dropout=.1, target_modules=['q_proj', 'v_proj'],
                                                  modules_to_save=['score']))
            sample = dict(input_ids_chosen=[2, 5, 3], attention_mask_chosen=[1]*3,
                          input_ids_rejected=[2, 6, 3], attention_mask_rejected=[1]*3,
                          ref_rewards_chosen=[-1.], ref_rewards_rejected=[1.])
            trainer = OurRewardTrainer(model=model, args=RewardConfig(output_dir=str(root / 'train'),
                                      max_steps=1, per_device_train_batch_size=2, report_to='none',
                                      save_strategy='no', remove_unused_columns=False, max_length=8),
                                      beta=1., data_collator=OurRewardDataCollatorWithPadding(tokenizer),
                                      train_dataset=Dataset.from_list([sample, sample]))
            trainer.train()
            model.eval()
            ids = torch.tensor([[2, 5, 3]], device=model.device)
            with torch.no_grad():
                expected = model(input_ids=ids).logits.cpu()
            model.save_pretrained(root / 'adapter')
            loaded = PeftModel.from_pretrained(LlamaForSequenceClassification.from_pretrained(root / 'base'),
                                              root / 'adapter').eval()
            with torch.no_grad():
                actual = loaded(input_ids=ids.cpu()).logits
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(actual).all())


if __name__ == '__main__':
    unittest.main()
