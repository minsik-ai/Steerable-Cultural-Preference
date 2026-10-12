"""User-authorized Tulu input adaptations, preserving original evaluation control flow."""


def adapt_chat_inputs(source):
    # Change only token construction inside the three original scoring loops.
    # Batch membership, ordering, truncation policy, mode and RNG remain untouched.
    before = "tokenizer(q_s, a_s, padding=True, return_tensors='pt')"
    after = ("tokenizer([tokenizer.apply_chat_template("
             "[{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}], "
             "tokenize=False, add_generation_prompt=False) for q, a in zip(q_s, a_s)], "
             "add_special_tokens=False, padding=True, return_tensors='pt')")
    if source.count(before) != 3:
        raise ValueError('Expected exactly three original scoring loops')
    source = source.replace(before, after)
    start = source.index('        def format_func_train(examples):')
    end = source.index('        def format_func_test(examples):', start)
    training = source[start:end]
    training = training.replace('"padding": "max_length", "truncation": True, "max_length": 512',
                                '"padding": False, "truncation": True, "max_length": 2048, "add_special_tokens": False')
    for side in ('chosen', 'rejected'):
        before = f'examples["{side}"][0]["content"] + " [SEP] " + examples["{side}"][1]["content"]'
        after = f'tokenizer.apply_chat_template(examples["{side}"], tokenize=False, add_generation_prompt=False)'
        if training.count(before) != 1:
            raise ValueError('Unexpected original training formatter')
        training = training.replace(before, after)
    return source[:start] + training + source[end:]
