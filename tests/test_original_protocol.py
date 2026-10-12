"""Protect source-derived execution from reconstructed scoring/training changes."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.country_search import BASE, trial_source
from scripts.original_protocol import (derive_source, normalized_config, primary_metrics,
                                       resolve_config, run_country)

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = (REPO / 'preferences.py').read_text()
LORA = dict(r=8, lora_alpha=64, lora_dropout=.1,
            target_modules=['q_proj', 'v_proj'], modules_to_save=['score'])
TULU = dict(BASE, lora=LORA)


def canonical(source):
    """Remove only documented modifications, retaining every other AST node."""
    class ApprovedChanges(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'peft':
                node.names = [name for name in node.names if name.name != 'get_peft_model']
            return node

        def visit_Assign(self, node):
            if len(node.targets) != 1:
                return self.generic_visit(node)
            target = ast.unparse(node.targets[0])
            if target == 'model.config.pad_token_id':
                if ast.unparse(node.value) != 'tokenizer.pad_token_id':
                    raise AssertionError('Unexpected pad adaptation')
                return None
            if target == 'model' and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'get_peft_model':
                if ast.unparse(node.value.args[0]) != 'model':
                    raise AssertionError('Unexpected model wrapping')
                return None
            if target in ('ref_rewards_chosen', 'ref_rewards_rejected') and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'torch.zeros_like':
                if ast.unparse(node.value.args[0]) != target:
                    raise AssertionError('Unexpected training reference adaptation')
                return None
            if target in ('BATCH_SIZE', 'RANDOM_DATA_SHUFFLE_SEED', 'EDITED_LR',
                          'REWARD_THRES', 'WEIGHTING_BETA', 'MODEL_NAME'):
                node.value = ast.Constant('approved-setting')
            # Only the final original country assignment has seven entries.
            if target == 'countries' and isinstance(node.value, ast.List) and (
                    len(node.value.elts) == 7 or len(node.value.elts) == 1):
                node.value = ast.Constant('approved-country')
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == 'RewardConfig':
                for kw in node.keywords:
                    if kw.arg == 'max_steps':
                        kw.value = ast.Constant('approved-budget')
            if ast.unparse(node.func) == 'Dataset.from_list' and ast.unparse(node.args[0]) in (
                    'utt_pref_train[count]', 'utt_pref_edited_train[count]'):
                node.args[0] = ast.Constant('approved-training-pool')
            return self.generic_visit(node)

    return ast.dump(ApprovedChanges().visit(ast.parse(source)))


def training_loop(source):
    main = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    return next(node for node in main.body if isinstance(node, ast.For)
                and ast.unparse(node.target) == 'count' and ast.unparse(node.iter) == 'countries'
                and any(isinstance(item, ast.FunctionDef) and item.name == 'format_func_train' for item in node.body))


class OriginalProtocolTests(unittest.TestCase):
    def test_deberta_default_derivation_is_existing_trial_source(self):
        self.assertEqual(derive_source(ORIGINAL, 'deberta', 'Israel', BASE),
                         trial_source(ORIGINAL, 'Israel', BASE))

    def test_ast_allows_only_declared_adaptations(self):
        for family in ('deberta', 'tulu'):
            for method in ('filtered_weighted', 'weighted_only', 'full_data'):
                config = dict(TULU if family == 'tulu' else BASE, method=method,
                              max_steps=128, batch_size=4, shuffle_seed=2, tau=.5)
                generated = derive_source(ORIGINAL, family, 'Australia', config)
                self.assertEqual(canonical(ORIGINAL), canonical(generated))
                self.assertEqual(generated.count('model.config.pad_token_id = tokenizer.pad_token_id'),
                                 2 if family == 'tulu' else 0)
                self.assertNotIn('model.eval()', generated)
                self.assertNotIn('torch.no_grad()', generated)
                self.assertNotIn('gradient_checkpointing', generated)
                self.assertNotIn('torch_dtype', generated)
                self.assertNotIn('apply_chat_template', generated)
        with self.assertRaises(ValueError):
            derive_source(ORIGINAL + '# drift\n', 'tulu', 'Australia', TULU)

    def test_original_format_functions_and_trainer_config_preserved(self):
        source = derive_source(ORIGINAL, 'tulu', 'Chile', TULU)
        original_loop, adapted_loop = training_loop(ORIGINAL), training_loop(source)
        original_functions = [ast.dump(node) for node in original_loop.body if isinstance(node, ast.FunctionDef)]
        adapted_functions = [ast.dump(node) for node in adapted_loop.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(original_functions, adapted_functions)
        for name in ('training_args', 'data_collator', 'trainer'):
            def assignment(loop):
                return next(ast.dump(node) for node in loop.body if isinstance(node, ast.Assign)
                            and ast.unparse(node.targets[0]) == name)
            self.assertEqual(assignment(original_loop), assignment(adapted_loop))

    def test_original_and_tulu_primary_scoring_have_identical_execution_trace(self):
        def trace(source):
            body = training_loop(source).body
            start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                         and ast.unparse(node.targets[0]) == 'rewards')
            end = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
                       and ast.unparse(node) == "print('Evaluating the test set!')")
            module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(body[start:end]), type_ignores=[]))
            calls = []

            class Inputs(dict):
                def to(self, device):
                    calls.append(('device', device))
                    return self

            class Scores(list):
                def cpu(self):
                    calls.append('cpu')
                    return self

                def detach(self):
                    calls.append('detach')
                    return self

            class Model:
                training = True
                rng = 99

                def __call__(self, **inputs):
                    calls.append(('forward', self.training, self.rng))
                    self.rng += 1
                    return SimpleNamespace(logits=Scores([self.rng] * len(inputs['questions'])))

            def tokenizer(questions, answers, **kwargs):
                calls.append(('tokenize', questions, answers, kwargs))
                return Inputs(questions=questions)

            groups = {'one': [dict(user_prompt='q', model_response=str(i), utterance_id=i) for i in range(5)],
                      'two': [dict(user_prompt='other', model_response='a', utterance_id=5)]}
            namespace = dict(count='Chile', test_baseline_rewards={'Chile': {}}, utt_test={'Chile': groups},
                             BATCH_SIZE=2, tqdm=lambda x: x, tokenizer=tokenizer, model=Model(),
                             device='cuda:0', print=lambda *a: None)
            exec(compile(module, '<scoring-trace>', 'exec'), namespace)
            return calls, namespace['test_baseline_rewards']

        original = trace(ORIGINAL)
        derived = trace(derive_source(ORIGINAL, 'tulu', 'Chile', TULU))
        self.assertEqual(original, derived)
        forwards = [x for x in derived[0] if isinstance(x, tuple) and x[0] == 'forward']
        self.assertEqual(forwards, [('forward', True, 99), ('forward', True, 100),
                                    ('forward', True, 101), ('forward', True, 102)])

    def test_full_data_only_neutralizes_training_references(self):
        source = derive_source(ORIGINAL, 'tulu', 'Chile', dict(TULU, method='full_data'))
        loop = training_loop(source)
        functions = {node.name: node for node in loop.body if isinstance(node, ast.FunctionDef)}
        original_test = next(node for node in training_loop(ORIGINAL).body
                             if isinstance(node, ast.FunctionDef) and node.name == 'format_func_test')
        self.assertEqual(ast.dump(original_test), ast.dump(functions['format_func_test']))
        class Tokenizer:
            def encode_plus(self, text, **kwargs):
                self.last = text, kwargs
                return {'input_ids': [[1, 2]], 'attention_mask': [[1, 1]]}
        tokenizer = Tokenizer()
        refs = {'a': 10, 'b': 20}
        namespace = dict(tokenizer=tokenizer, utt_train_rewards=refs,
                         torch=SimpleNamespace(zeros_like=lambda x: 0))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[functions['format_func_train']], type_ignores=[])),
                     '<format>', 'exec'), namespace)
        formatted = namespace['format_func_train'](dict(chosen=[{'content': 'q'}, {'content': 'yes'}],
                                                       rejected=[{'content': 'q'}, {'content': 'no'}],
                                                       chosen_id='a', rejected_id='b'))
        self.assertEqual((formatted['ref_rewards_chosen'], formatted['ref_rewards_rejected']), (0, 0))
        self.assertEqual(refs, {'a': 10, 'b': 20})
        self.assertEqual(tokenizer.last, ('q [SEP] no', dict(padding='max_length', truncation=True,
                                                          max_length=512, return_tensors='pt')))

    def test_config_requires_explicit_lora_and_rejects_semantic_overrides(self):
        document = json.loads((REPO / 'configs/country_defaults.json').read_text())
        self.assertEqual(normalized_config('deberta', resolve_config(document, 'Israel'))['shuffle_seed'], 2)
        for key in ('torch_dtype', 'max_length', 'gradient_checkpointing', 'input_format',
                    'score_batch_size', 'attn_implementation', 'evaluation_mode'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalized_config('tulu', dict(TULU, **{key: False}))
        for config in (BASE, dict(TULU, lora=dict(LORA, r=0)),
                       dict(TULU, lora=dict(LORA, target_modules=[])), dict(TULU, split_seed=1)):
            with self.assertRaises(ValueError):
                normalized_config('tulu', config)

    def test_preparation_and_subprocess_preserve_primary_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            def fake_run(command, **kwargs):
                self.assertEqual(kwargs['cwd'], REPO)
                self.assertEqual(command[1], str(root / 'chile/preferences_trial.py'))
                report = Path(command[2]) / 'experiment/seed/0'
                report.mkdir(parents=True)
                for filename in ('full_report.json', 'partial_report.json'):
                    (report / filename).write_text(json.dumps({'Chile': dict(start=50., ours_partial=60.),
                                                              'avg': dict(start=50., ours_partial=60.)}))
                return SimpleNamespace(returncode=0)
            with patch('scripts.original_protocol.subprocess.run', side_effect=fake_run) as run, \
                 patch('scripts.original_protocol.importlib.metadata.distributions', return_value=[]):
                result = run_country(REPO, root, 'tulu', 'Chile', TULU)
                self.assertEqual(run.call_count, 1)  # no reloading, diagnostics or extra child RNG work
            self.assertEqual(result['primary_metrics']['overall'], 60.)
            self.assertFalse(result['checkpoint_reload_validated'])
            self.assertEqual(result['status'], 'completed')
            self.assertTrue((root / 'chile/source.patch').is_file())
            with self.assertRaises(FileExistsError):
                run_country(REPO, root, 'tulu', 'Chile', TULU, prepare_only=True)
            report = root / 'chile/run/experiment/seed/0/full_report.json'
            report.write_text('{"Chile":{"start":NaN,"ours_partial":60},"avg":{"start":50,"ours_partial":60}}')
            with self.assertRaises(ValueError):
                primary_metrics(root / 'chile/run', 'Chile')


if __name__ == '__main__':
    unittest.main()
