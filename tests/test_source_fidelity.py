"""Guard byte-for-byte fidelity to the supplied experiment implementation."""
import subprocess
import unittest
from pathlib import Path


class SourceFidelityTests(unittest.TestCase):
    def test_experiment_sources_match_original_revision(self):
        root = Path(__file__).resolve().parents[1]
        original = '94f8430054835aa919b226b4344b8f0e8424f29b'
        for name in ('preferences.py', 'our_reward_trainer.py',
                     'our_inverse_reward_trainer.py', 'our_data_collator.py'):
            with self.subTest(file=name):
                expected = subprocess.check_output(['git', 'show', f'{original}:{name}'], cwd=root)
                self.assertEqual((root / name).read_bytes(), expected)


if __name__ == '__main__':
    unittest.main()
