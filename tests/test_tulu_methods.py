import unittest
from types import SimpleNamespace
from scripts.tulu_reconstructed import select_pairs, loss_references


class TuluMethodsTests(unittest.TestCase):
    def test_full_and_weighted_pools_match_and_filter_is_exact(self):
        pairs = [dict(chosen_id='a', rejected_id='b'), dict(chosen_id='c', rejected_id='d'),
                 dict(chosen_id='e', rejected_id='f')]
        reference = dict(a=-1., b=0., c=0., d=0., e=1., f=0.)
        self.assertEqual(select_pairs(pairs, reference, 'full_data', .5), pairs)
        self.assertEqual(select_pairs(pairs, reference, 'weighted_only', .5), pairs)
        self.assertEqual(select_pairs(pairs, reference, 'filtered_weighted', .5), pairs[:1])
        self.assertEqual(loss_references(pairs[0], reference, 'full_data'), dict(chosen=0., rejected=0.))
        self.assertEqual(reference['a'], -1.)
        self.assertEqual(loss_references(pairs[0], reference, 'weighted_only'), dict(chosen=-1., rejected=0.))

    def test_unit_and_weighted_loss_and_gradients(self):
        import torch
        from our_reward_trainer import OurRewardTrainer
        trainer = object.__new__(OurRewardTrainer)
        trainer.beta = 1.
        trainer.use_reward_data_collator = True
        trainer.args = SimpleNamespace(center_rewards_coefficient=None)
        for method in ('full_data', 'weighted_only'):
            parameter = torch.tensor(.7, requires_grad=True)
            refs = [1., 0., -2.] if method == 'weighted_only' else [0., 0., 0.]
            inputs = dict(input_ids_chosen=torch.tensor([[1], [2], [3]]),
                          input_ids_rejected=torch.tensor([[2], [1], [1]]),
                          attention_mask_chosen=torch.ones(3, 1), attention_mask_rejected=torch.ones(3, 1),
                          ref_rewards_chosen=torch.tensor(refs).reshape(3, 1),
                          ref_rewards_rejected=torch.zeros(3, 1))
            model = lambda input_ids, **kwargs: {'logits': input_ids.float() * parameter}
            actual = trainer.compute_loss(model, inputs)
            weights = torch.tensor([1., 1., torch.exp(torch.tensor(-2.)).item()]) if method == 'weighted_only' else torch.ones(3)
            expected = -(weights * torch.nn.functional.logsigmoid(torch.tensor([-1., 1., 2.]) * parameter)).mean()
            torch.testing.assert_close(actual, expected)
            actual_grad = torch.autograd.grad(actual, parameter, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected, parameter)[0]
            torch.testing.assert_close(actual_grad, expected_grad)
