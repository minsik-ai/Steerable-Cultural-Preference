"""Exercise the source-backed Tulu path with real tiny Llama/PEFT objects."""
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


class OriginalTuluSmokeTests(unittest.TestCase):
    def test_original_modes_gradients_padding_and_adapter_export(self):
        import torch
        from datasets import Dataset
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import (PreTrainedTokenizerFast, LlamaConfig, LlamaForSequenceClassification,
                                  AutoModelForSequenceClassification)
        from scripts.original_protocol import derive_source
        torch.set_num_threads(2)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'base';base.mkdir()
            raw=Tokenizer(models.WordLevel({'[PAD]':0,'[UNK]':1,'[SEP]':2,'question':3,
                                            'one':4,'two':5,'[BOS]':6},unk_token='[UNK]'))
            raw.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
            raw.post_processor=processors.TemplateProcessing(single='[BOS] $A',pair='[BOS] $A $B',
                                                               special_tokens=[('[BOS]',6)])
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',unk_token='[UNK]',
                                             model_input_names=['input_ids','attention_mask'])
            if getattr(self, 'chat_profile', False):
                tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            tokenizer.save_pretrained(base)
            model=LlamaForSequenceClassification(LlamaConfig(vocab_size=7,hidden_size=16,
                intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
                num_key_value_heads=2,num_labels=1,pad_token_id=0))
            model.eval();model.save_pretrained(base)
            with torch.no_grad():
                scores=model(**tokenizer(['question','question'],['one','two'],return_tensors='pt')).logits.flatten()
            self.assertNotEqual(scores[0].item(),scores[1].item())
            chosen=int(scores.argmin())
            survey=Dataset.from_list([dict(user_id=str(i),location=dict(reside_country='Chile',birth_country='Chile'))
                                      for i in range(20)])
            rows=Dataset.from_list([dict(user_id=str(i),interaction_id=str(i),utterance_id=f'{i}-{j}',
                                         user_prompt='question',model_response=a,score=int(j==chosen))
                                    for i in range(20) for j,a in enumerate(('one','two'))])
            repo=Path(__file__).resolve().parents[1]
            config=dict(max_steps=1,learning_rate=1e-6,tau=.7,beta=1.1,shuffle_seed=0,
                        lora=dict(r=2,lora_alpha=4,lora_dropout=.1,target_modules=['q_proj','v_proj'],
                                  modules_to_save=['score']))
            if getattr(self, 'chat_profile', False):
                config['execution_profile'] = 'tulu-chat-v1'
            path=root/'original_tulu.py'
            path.write_text(derive_source((repo/'preferences.py').read_text(),'tulu','Chile',config))
            output=root/'run'
            with patch.object(sys,'argv',[str(path),str(output)]):
                main=runpy.run_path(str(path))['main']
            main.__globals__['MODEL_NAME']=str(base)  # test-only local model substitution
            records=[]
            original_loader=AutoModelForSequenceClassification.from_pretrained
            def load(*args,**kwargs):
                result=original_loader(*args,**kwargs)
                def observe(module,args,kwargs):
                    ids=kwargs.get('input_ids',args[0] if args else None)
                    records.append((module.training,torch.is_grad_enabled(),tuple(ids.shape),str(next(module.parameters()).dtype)))
                # PEFT delegates the classifier forward directly; observe its decoder.
                result.model.register_forward_pre_hook(observe,with_kwargs=True)
                return result
            with patch('datasets.load_dataset',side_effect=lambda name,config,**kw: survey if config=='survey' else rows), \
                 patch.object(AutoModelForSequenceClassification,'from_pretrained',side_effect=load):
                main(str(output))
            self.assertTrue(all(grad for _,grad,_,_ in records))
            training=[r for r in records if r[2][0] == 8] if getattr(self, 'chat_profile', False) else [r for r in records if r[2][1]==512]
            self.assertEqual(len(training),2)
            self.assertTrue(all(mode for mode,_,_,_ in training))
            self.assertTrue(all(not r[0] for r in records[:20]))
            self.assertTrue(all(r[0] and r[2][0]==2 for r in records[-3:]))
            self.assertTrue(all(r[3]=='torch.float32' for r in records))
            checkpoint=output/'trunc/Chile/ours_3epoch'
            self.assertTrue((checkpoint/'adapter_model.safetensors').exists())
            self.assertEqual(json.loads((checkpoint/'adapter_config.json').read_text())['task_type'],'SEQ_CLS')
            report=json.loads((output/'experiment/seed/0/full_report.json').read_text())
            self.assertIn('ours_partial',report['Chile'])
