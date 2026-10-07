"""Offline regression tests for the actual SCPO loss/collator (no model downloads)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import BertTokenizer

from our_data_collator import OurRewardDataCollatorWithPadding
from our_reward_trainer import OurRewardTrainer


class RewardPipelineTests(unittest.TestCase):
    def test_collator_preserves_per_pair_reference_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            vocab = Path(folder) / 'vocab.txt'
            vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
            tokenizer = BertTokenizer(str(vocab))
            features = [dict(input_ids_chosen=[2, 5, 3], attention_mask_chosen=[1]*3,
                             input_ids_rejected=[2, 6, 5, 3], attention_mask_rejected=[1]*4,
                             ref_rewards_chosen=[float(i)], ref_rewards_rejected=[-1.])
                        for i in (0, 2)]
            batch = OurRewardDataCollatorWithPadding(tokenizer)(features)
            self.assertEqual(tuple(batch['ref_rewards_chosen'].shape), (2, 1))
            torch.testing.assert_close(batch['ref_rewards_chosen'], torch.tensor([[0.], [2.]]))
            self.assertEqual(tuple(batch['input_ids_rejected'].shape), (2, 4))

    def test_weighted_loss_matches_pairwise_formula_and_backpropagates(self):
        trainer = object.__new__(OurRewardTrainer)
        trainer.beta = 1.1
        trainer.use_reward_data_collator = True
        trainer.args = SimpleNamespace(center_rewards_coefficient=None)
        parameter = torch.tensor(0.7, requires_grad=True)

        def model(input_ids, **kwargs):
            return {'logits': input_ids.float()[:, :1] * parameter}

        inputs = dict(input_ids_chosen=torch.tensor([[1], [3]]),
                      input_ids_rejected=torch.tensor([[2], [1]]),
                      attention_mask_chosen=torch.ones(2, 1),
                      attention_mask_rejected=torch.ones(2, 1),
                      ref_rewards_chosen=torch.tensor([[-2.], [2.]]),
                      ref_rewards_rejected=torch.zeros(2, 1))
        loss = trainer.compute_loss(model, inputs)
        expected = -(torch.tensor([(-2 / 1.1), 0.]).exp()
                     * torch.nn.functional.logsigmoid(torch.tensor([-1., 2.]) * parameter)).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad))


if __name__ == '__main__':
    unittest.main()
