import unittest
from scripts.select_tulu_format import select_format


def report(legacy, chat):
    return {'formats': {name: {'countries': {str(i): {} for i in range(7)},
                              'mean_absolute_difference_pp': error,
                              'maximum_absolute_error_pp': error}
                        for name, error in [('legacy-pair', legacy), ('chat-template', chat)]}}


class FormatSelectionTests(unittest.TestCase):
    def test_select_verified_baseline_match(self):
        result = select_format(report(.01, 3.))
        self.assertEqual(result['input_format'], 'legacy-pair')
        self.assertTrue(result['baseline_matches_paper'])
        self.assertFalse(result['historical_training_recipe_verified'])

    def test_nonmatching_fallback_is_explicitly_provisional(self):
        result = select_format(report(.5, 2.))
        self.assertEqual(result['input_format'], 'chat-template')
        self.assertFalse(result['baseline_matches_paper'])

    def test_partial_baseline_is_rejected(self):
        data = report(.01, .02)
        data['formats']['legacy-pair']['countries'].pop('0')
        with self.assertRaises(ValueError):
            select_format(data)
