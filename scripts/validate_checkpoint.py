"""Validate an exported reward model without publishing it."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, local_files_only=True).to(device).eval()
    questions = ['What is a respectful way to disagree?', 'Explain the water cycle.']
    answers = ['Listen first and explain your reasons calmly.', 'Water evaporates, condenses, and falls as precipitation.']
    inputs = tokenizer(questions, answers, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.inference_mode():
        first = model(**inputs).logits
        second = model(**inputs).logits
    assert first.shape == (2, 1), first.shape
    assert torch.isfinite(first).all(), first
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    encoding_matches = []
    for question, answer in zip(questions, answers):
        paired = tokenizer(question, answer, truncation=True, max_length=512)['input_ids']
        concatenated = tokenizer(question + ' [SEP] ' + answer, truncation=True, max_length=512)['input_ids']
        encoding_matches.append(paired == concatenated)
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), device=device,
                  scores=first.cpu().tolist(), deterministic=True,
                  sample_pair_vs_concat_ids_equal=encoding_matches)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
