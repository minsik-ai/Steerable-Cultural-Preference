import json
from pathlib import Path
import tempfile
import unittest
from scripts.summarize_original_protocol import summarize


class OriginalSummaryTests(unittest.TestCase):
    def test_incomplete_runs_never_produce_macro(self):
        with tempfile.TemporaryDirectory() as folder:
            result=summarize(Path(folder))
            self.assertFalse(result['complete'])
            self.assertEqual(len(result['missing']),14)
            self.assertNotIn('macro_accuracy',result)

    def test_old_protocol_cannot_enter_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p=root/'deberta/chile';p.mkdir(parents=True)
            (p/'manifest.json').write_text(json.dumps(dict(status='completed',protocol='reconstructed')))
            with self.assertRaisesRegex(ValueError,'Refusing to compare'):
                summarize(root)
