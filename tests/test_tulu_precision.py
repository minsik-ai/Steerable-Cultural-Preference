"""Pure offline checks for precision-audit comparisons."""
import unittest
from scripts.tulu_precision_diagnostic import compare_rewards, reference_weights, summarize


class PrecisionDiagnosticTests(unittest.TestCase):
    def test_ranking_changes_include_ties(self):
        pairs = [dict(chosen_id='a', rejected_id='b'), dict(chosen_id='c', rejected_id='d')]
        before = dict(a=0., b=0., c=2., d=1.)
        after = dict(a=1., b=0., c=1., d=1.)
        result = compare_rewards(pairs, before, after)
        self.assertEqual(result['changed_pair_ranks'], 2)
        self.assertEqual(result['correctness_flips'], 2)
        self.assertEqual(result['fixed'], 1)
        self.assertEqual(result['regressed'], 1)
        self.assertEqual(result['before_ties'], 1)
        self.assertEqual(result['after_ties'], 1)
        with self.assertRaises(ValueError):
            compare_rewards(pairs, before, dict(a=1.))

    def test_weight_diagnostic_uses_strict_filter(self):
        pairs = [dict(chosen_id='a', rejected_id='b'), dict(chosen_id='c', rejected_id='d')]
        result = reference_weights(pairs, dict(a=0., b=0., c=0., d=1.), .5, 1.)
        self.assertEqual(result['retained_weights']['count'], 1)
        self.assertAlmostEqual(result['effective_sample_size'], 1.)
        self.assertEqual(summarize([]), {'count': 0})
        self.assertEqual(summarize([1., 3.])['median'], 2.)


if __name__ == '__main__':
    unittest.main()
