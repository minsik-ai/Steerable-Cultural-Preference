"""Verify chat inputs cannot silently change shared evaluation semantics."""
import ast
import json
from pathlib import Path
import unittest

from scripts.original_protocol import derive_source, normalized_config

ROOT = Path(__file__).resolve().parents[1]


class ChatProtocolTests(unittest.TestCase):
    def test_only_training_formatter_and_scoring_inputs_change(self):
        original = (ROOT / 'preferences.py').read_text()
        config = json.loads((ROOT / 'configs/tulu_chat_protocol.json').read_text())
        chat = ast.parse(derive_source(original, 'tulu', 'Chile', config))
        config.pop('execution_profile')
        legacy = ast.parse(derive_source(original, 'tulu', 'Chile', config))

        class NormalizeInputs(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == 'format_func_train':
                    node.body = [ast.Pass()]
                    return node
                return self.generic_visit(node)

            def visit_Assign(self, node):
                if ast.unparse(node.targets[0]) == 'inputs':
                    node.value = ast.Constant('scoring-inputs')
                return self.generic_visit(node)

        self.assertEqual(ast.dump(NormalizeInputs().visit(chat)),
                         ast.dump(NormalizeInputs().visit(legacy)))

    def test_scoring_has_no_length_cap_mode_reset_or_reload(self):
        config = json.loads((ROOT / 'configs/tulu_chat_protocol.json').read_text())
        source = derive_source((ROOT / 'preferences.py').read_text(), 'tulu', 'Chile', config)
        scoring = [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == 'inputs']
        self.assertEqual(len(scoring), 3)
        for expression in scoring:
            token_call = expression.func.value
            keywords = {k.arg: ast.literal_eval(k.value) for k in token_call.keywords}
            self.assertEqual(keywords, dict(add_special_tokens=False, padding=True, return_tensors='pt'))
        self.assertNotIn('model.eval()', source)
        self.assertNotIn('torch.no_grad()', source)
        self.assertNotIn('PeftModel.from_pretrained', source)

    def test_deberta_rejects_chat_profile(self):
        with self.assertRaises(ValueError):
            normalized_config('deberta', {'execution_profile': 'tulu-chat-v1'})

    def test_training_uses_chat_and_dynamic_padding(self):
        config = json.loads((ROOT / 'configs/tulu_chat_protocol.json').read_text())
        source = derive_source((ROOT / 'preferences.py').read_text(), 'tulu', 'Chile', config)
        function = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
                        and n.name == 'format_func_train')
        text = ast.unparse(function)
        self.assertIn('max_length\': 2048', text)
        self.assertIn('padding\': False', text)
        self.assertEqual(text.count('apply_chat_template'), 2)


if __name__ == '__main__':
    unittest.main()
