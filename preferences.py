# from transformers.utils import logging
# logging.set_verbosity_error()
import csv
import warnings

from peft import LoraConfig, TaskType

warnings.filterwarnings('ignore')

import torch
from datasets import Dataset
from trl import RewardConfig, RewardTrainer
import collections
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import numpy as np

from our_data_collator import OurRewardDataCollatorWithPadding
from our_reward_trainer import OurRewardTrainer
# from our_inverse_reward_trainer import OurInverseRewardTrainer

BATCH_SIZE = 8
# EPOCHS = 2
EPOCHS = 1
LR = 1e-6
LR_OVERRIDE = {
}
# LR = 1e-5
EDITED_EPOCHS = 3
EDITED_LR = 1e-6
EDITED_LR_OVERRIDE = {
}

SKIP_ORIG_PARTIAL = ["South Africa", "Mexico", "Australia", "New Zealand", "Chile", "Canada", "Israel"]

REWARD_THRES = 0.7
REWARD_THRES_OVERRIDE = {
}
WEIGHTING_BETA = 1.10

MAX_STEPS = {
    # "United States": 1200,
    "New Zealand": 1024,
    "Mexico": 1024,
    "Chile": 1024,
    "South Africa": 1024,
    "Israel": 1024,
    "Canada": 1024,
    "Australia": 1024
}

REPORT = 1000000

RANDOM_DATA_SPLIT_SEED = 0
RANDOM_DATA_SHUFFLE_SEED = 0
RANDOM_MODEL_SEED = 0

from transformers import set_seed
set_seed(RANDOM_MODEL_SEED)
MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-base"
# MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B-RM"


def main(out_path):
    print("script starting")
    # Use reside_country

    # GPO list
    countries = ["United States", "Canada",
                 "Germany", "France", "Spain",
                 "Nigeria", "Egypt",
                 "India", "China", "Japan",
                 "Brazil", "Argentina",
                 "Australia", "New Zealand"]
    # Our list
    # Max 3 countries per continent, more than 10 users.
    # Must be also in GlobalOpinionsQA - South Africa, Korea not in the Global Opinions QA
    # North America : United States, Canada, Mexico
    # South America : Chile
    # Europe: United Kingdom, Spain, Belgium
    # Asia: Israel, Japan
    # Australia: Australia
    # countries = ["United States", "Mexico", "Canada",
    #              "Chile",
    #              "United Kingdom", "Spain", "Belgium",
    #              "Israel", "Japan",
    #              "Australia"]
    # Counter({'United States': 386, 'United Kingdom': 340, 'South Africa': 86, 'New Zealand': 77, 'Australia': 72,
    # 'Mexico': 67, 'Chile': 65, 'Israel': 61, 'Canada': 54, 'Spain': 18, 'Belgium': 17, 'Hungary': 16, 'Denmark': 15,
    # 'Norway': 15, 'Ireland': 15, 'Poland': 14, 'Czechia': 14, 'Switzerland': 14, 'Latvia': 14, 'Germany': 13,
    # 'Greece': 13, 'Finland': 13, 'Italy': 13, 'France': 12, 'Japan': 11, 'Slovenia': 10, 'Austria': 10, 'Estonia': 10,
    # 'Netherlands': 8, 'Portugal': 7, 'Korea, Republic of': 7, 'Sweden': 6, 'Luxembourg': 2,
    # 'Tanzania, United Republic of': 1, 'Uruguay': 1, 'Lesotho': 1, 'Iceland': 1, 'Prefer not to say': 1})

    # Or, just top-10 responding countries with more than 20 users
    # countries = ["United States", "United Kingdom", "South Africa", "New Zealand", "Australia", "Mexico", "Chile",
    #              "Israel", "Canada"]
    # countries = ["United States"]

    # countries = ["New Zealand", "Mexico", "South Africa"]
    # countries = ["South Africa"]
    # countries = ["Australia"]

    # countries = ["Japan"]
    countries = ["Chile", "South Africa", "New Zealand", "Australia", "Mexico", "Israel", "Canada"]

    print("dataset starting")
    from datasets import load_dataset
    # Load datasets
    utts = load_dataset("HannahRoseKirk/prism-alignment", "utterances", split="train")
    infos = load_dataset("HannahRoseKirk/prism-alignment", "survey", split="train")

    info_split = infos.train_test_split(test_size=0.15, seed=RANDOM_DATA_SPLIT_SEED)
    info_split.save_to_disk(f"{out_path}/dataset/PRISM/seed/{RANDOM_DATA_SPLIT_SEED}/")

    info_train = info_split["train"]
    info_test = info_split["test"]

    reside_countries = collections.Counter()
    birth_countries = collections.Counter()

    info_train_ids = collections.defaultdict(set)

    # map users to countries for the training set
    # count number of users that reside/were born in each country
    for info in info_train:
        country = info["location"]["reside_country"]
        b_country = info["location"]["birth_country"]
        reside_countries[country] += 1
        birth_countries[b_country] += 1
        if country in countries:
            info_train_ids[country].add(info["user_id"])

    # map users to countries for the test set
    # count number of users that reside or were born in each country
    info_test_ids = collections.defaultdict(set)
    for info in info_test:
        country = info["location"]["reside_country"]
        b_country = info["location"]["birth_country"]
        reside_countries[country] += 1
        birth_countries[b_country] += 1
        if country in countries:
            info_test_ids[country].add(info["user_id"])

    print("**** RESIDE COUNTRIES ****")
    print(reside_countries)
    # print(birth_countries)

    full_perfs = {count: dict() for count in countries}
    # Different from average
    full_perfs["avg"] = dict()

    partial_perfs = {count: dict() for count in countries}
    # Different from average
    partial_perfs["avg"] = dict()

    print("**** USERS ****")
    for country in countries:
        print(f"{country}: train {len(info_train_ids[country])} test {len(info_test_ids[country])}")

    # Keyed per user & turn
    utt_train = {country: collections.defaultdict(list) for country in countries}
    utt_test = {country: collections.defaultdict(list) for country in countries}
    for utt in utts:
        user_id = utt["user_id"]
        turn = utt["interaction_id"]
        for country in countries:
            key = f"{user_id}_{turn}"
            if user_id in info_train_ids[country]:
                utt_train[country][key].append(utt)
            elif user_id in info_test_ids[country]:
                utt_test[country][key].append(utt)
            else:
                continue
    print("**** Raw Utterances ****")
    print("**** Train ****")
    print({count: len(items) for count, items in utt_train.items()})
    print("**** Test ****")
    print({count: len(items) for count, items in utt_test.items()})

    utt_score_train = collections.defaultdict(list)
    utt_score_test = collections.defaultdict(list)

    def format_text(chosen, rejected):
        assert chosen["interaction_id"] == rejected["interaction_id"]
        return {
            "chosen": [{"content": chosen["user_prompt"], "role": "user"},
                       {"content": chosen["model_response"], "role": "assistant"}],
            "rejected": [{"content": rejected["user_prompt"], "role": "user"},
                         {"content": rejected["model_response"], "role": "assistant"}],
            "chosen_id": chosen["utterance_id"],
            "rejected_id": rejected["utterance_id"],
            "interaction_id": chosen["interaction_id"]
        }

    # import model and tokenizer
    # move to GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model predicts which generated answer is better judged by a human
    # returns a score of how good a response is given a question (?)
    model, tokenizer = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME), \
                       AutoTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(device)

    ### BASELINE ###

    # Batched utterance score processing
    # get scores/rewards for each utterance in the train set
    # utterance consists of both a user's prompt and the model's response
    utt_train_rewards = {}
    for country, keyed_per_turn in utt_train.items():
        print(f"Processing {country} - train")
        for raw_utts in tqdm(keyed_per_turn.values()):
            for i in range(0, len(raw_utts), BATCH_SIZE):
                utt_batch = raw_utts[i:min(len(raw_utts), i + BATCH_SIZE)]

                # questions are prompts from the users
                q_s = []
                # answers are how the model responds
                a_s = []

                # create list of all user prompts
                # create list of all model responses
                for utt in utt_batch:
                    q = utt["user_prompt"]
                    a = utt["model_response"]
                    q_s.append(q)
                    a_s.append(a)

                # tokenize the inputs
                inputs = tokenizer(q_s, a_s, padding=True, return_tensors='pt').to(device)
                # given question, answer
                # get a score for how good the answer is
                # save the score for the given utterance
                scores = model(**inputs).logits.cpu().detach()
                for j in range(len(utt_batch)):
                    utt_id = utt_batch[j]["utterance_id"]
                    score = scores[j]
                    utt_train_rewards[utt_id] = score

    # get scores/rewards for each utterance in the test set
    # this scoring loop could probably be decomposed with the one starting on line 161

    utt_test_rewards = {}
    for country, keyed_per_turn in utt_test.items():
        print(f"Processing {country} - test")
        for raw_utts in tqdm(keyed_per_turn.values()):
            for i in range(0, len(raw_utts), BATCH_SIZE):
                utt_batch = raw_utts[i:min(len(raw_utts), i + BATCH_SIZE)]
                q_s = []
                a_s = []
                for utt in utt_batch:
                    q = utt["user_prompt"]
                    a = utt["model_response"]
                    q_s.append(q)
                    a_s.append(a)
                inputs = tokenizer(q_s, a_s, padding=True, return_tensors='pt').to(device)

                scores = model(**inputs).logits.cpu().detach()
                for j in range(len(utt_batch)):
                    utt = utt_batch[j]
                    utt_id = utt["utterance_id"]
                    score = scores[j]
                    utt_test_rewards[utt_id] = score


    utt_pref_train = collections.defaultdict(list)
    utt_pref_test = collections.defaultdict(list)

    utt_pref_edited_train = collections.defaultdict(list)
    utt_pref_edited_test = collections.defaultdict(list)

    print("Processing edits")
    score_same_cnt = 0

    # utt_train is keyed per user and turn
    # {country: {key: [utterances]}}
    # key = f"{user_id}_{turn}"
    # turn is not "turn" in the dataset, but rather interaction_id
    # see line 117: turn = utt["interaction_id"]
    # what is interaction_id?

    def score_thres(bigger, smaller, country):
        return (torch.exp(bigger) / (torch.exp(bigger) + torch.exp(smaller))) < (REWARD_THRES if country not in REWARD_THRES_OVERRIDE else REWARD_THRES_OVERRIDE[country])

    # utt_pref_train = {country: [(pref_utt_id, dispref_utt_id)...]}
    for country, keyed_per_turn in utt_train.items():

        # prefs is a reference to utt_pref_train
        # when we modify prefs in the loop below, this also populates utt_pref_train
        prefs = utt_pref_train[country]
        edited_prefs = utt_pref_edited_train[country]
        scores = utt_score_train[country]
        # Ignoring key which is user / turn info.
        for raw_utts in keyed_per_turn.values():
            for i, utt1 in enumerate(raw_utts):
                utt1r = utt_train_rewards[utt1["utterance_id"]]
                utt1s = utt1["score"]
                scores.append([utt1, utt1s])
                for j in range(i + 1, len(raw_utts)):
                    utt2 = raw_utts[j]
                    utt2r = utt_train_rewards[utt2["utterance_id"]]
                    utt2s = utt2["score"]
                    if utt1s > utt2s:
                        prefs.append(format_text(utt1, utt2))
                        if score_thres(utt1r, utt2r, country):
                            edited_prefs.append(format_text(utt1, utt2))
                    elif utt2s > utt1s:
                        prefs.append(format_text(utt2, utt1))
                        if score_thres(utt2r, utt1r, country):
                            edited_prefs.append(format_text(utt2, utt1))
                    else:
                        score_same_cnt += 1

    print("Evaluating test results")
    for country, keyed_per_turn in utt_test.items():
        prefs = utt_pref_test[country]
        edited_prefs = utt_pref_edited_test[country]
        scores = utt_score_test[country]
        # Ignoring key which is user / turn info.

        # for each pair of utterances
        for raw_utts in keyed_per_turn.values():
            for i, utt1 in enumerate(raw_utts):
                # utt1s stands for utt1_score
                utt1r = utt_test_rewards[utt1["utterance_id"]]
                utt1s = utt1["score"]

                # scores = [[utt, utt_score], ...]
                scores.append([utt1, utt1s])

                for j in range(i + 1, len(raw_utts)):
                    utt2 = raw_utts[j]
                    utt2r = utt_test_rewards[utt2["utterance_id"]]
                    utt2s = utt2["score"]

                    # prefs stores all pairs of utterances
                    # with the higher scored one at idx 0
                    if utt1s > utt2s:
                        prefs.append(format_text(utt1, utt2))
                        if utt1r < utt2r:
                            edited_prefs.append(format_text(utt1, utt2))
                    elif utt2s > utt1s:
                        prefs.append(format_text(utt2, utt1))
                        if utt2r < utt1r:
                            edited_prefs.append(format_text(utt2, utt1))
                    else:
                        # if scored the same, count it, but leave out of prefs
                        # the count of how many have been "removed"
                        score_same_cnt += 1

    print("**** PAIRS ****")
    print("**** TRAIN ****")
    print({count: len(pairs) for count, pairs in utt_pref_train.items()})
    print("**** TEST ****")
    print({count: len(pairs) for count, pairs in utt_pref_test.items()})
    print("**** EDITED TRAIN ****")
    print({count: len(pairs) for count, pairs in utt_pref_edited_train.items()})
    print("**** EDITED TEST ****")
    print({count: len(pairs) for count, pairs in utt_pref_edited_test.items()})
    print(f"Scored same : {score_same_cnt}")

    # Compute edited pairs ratio (edited / total) for train and test
    edited_ratios = {"train": {}, "test": {}}
    total_train_pairs = 0
    total_train_edited = 0
    total_test_pairs = 0
    total_test_edited = 0
    for count in countries:
        train_total = len(utt_pref_train[count])
        train_edited = len(utt_pref_edited_train[count])
        test_total = len(utt_pref_test[count])
        test_edited = len(utt_pref_edited_test[count])
        edited_ratios["train"][count] = train_edited / train_total if train_total > 0 else 0.0
        edited_ratios["test"][count] = test_edited / test_total if test_total > 0 else 0.0
        total_train_pairs += train_total
        total_train_edited += train_edited
        total_test_pairs += test_total
        total_test_edited += test_edited
    # Compute average ratios based on total counts
    edited_ratios["train"]["avg"] = total_train_edited / total_train_pairs if total_train_pairs > 0 else 0.0
    edited_ratios["test"]["avg"] = total_test_edited / total_test_pairs if total_test_pairs > 0 else 0.0

    print("**** EDITED PAIRS RATIO ****")
    print("**** TRAIN ****")
    for count in countries:
        print(f"{count}: {edited_ratios['train'][count]:.4f}")
    print(f"Average: {edited_ratios['train']['avg']:.4f}")
    print("**** TEST ****")
    for count in countries:
        print(f"{count}: {edited_ratios['test'][count]:.4f}")
    print(f"Average: {edited_ratios['test']['avg']:.4f}")

    print("**** Starting reward model Evals ****")
    print("**** Percentage of pairs correctly evaluated ****")
    avg_perf = 0.
    for count, pairs in utt_pref_test.items():
        correct = 0
        wrong = 0
        for pair in pairs:
            chosen_score = utt_test_rewards[pair["chosen_id"]]
            rejected_score = utt_test_rewards[pair["rejected_id"]]

            # model (deberta) gave higher score to preferred response
            if chosen_score > rejected_score:
                correct += 1
            # model (deberta) gave higher score to rejected response
            else:
                wrong += 1
        perf = correct * 1.0 / (correct + wrong) * 100
        print(f"{count}: {perf:.2f}%")
        full_perfs[count]["start"] = perf
        avg_perf += perf
    avg_perf /= len(countries)
    full_perfs["avg"]["start"] = avg_perf
    print(f"Starting Performance: {avg_perf:.2f}%")

    avg_perf = 0.
    for count, pairs in utt_pref_edited_test.items():
        correct = 0
        wrong = 0
        for pair in pairs:
            chosen_score = utt_test_rewards[pair["chosen_id"]]
            rejected_score = utt_test_rewards[pair["rejected_id"]]

            # model (deberta) gave higher score to preferred response
            if chosen_score > rejected_score:
                correct += 1
            # model (deberta) gave higher score to rejected response
            else:
                wrong += 1
        perf = correct * 1.0 / (correct + wrong) * 100
        print(f"{count}: {perf:.2f}%")
        partial_perfs[count]["start"] = perf
        avg_perf += perf
    avg_perf /= len(countries)
    partial_perfs["avg"]["start"] = avg_perf
    print(f"Starting Partial Performance: {avg_perf:.2f}%")

    print("**** Training Country-Specific Models ****")

    ### TRAIN THE REWARD MODEL (DEBERTA) WITH EDITED PREFS ###
    # this training loop can probably be decomposed with the one starting on line 328
    print("**** Ablation reward model training ****")

    avg_perf = 0.
    avg_partial_perf = 0.
    test_baseline_rewards = collections.defaultdict(dict)
    # Train reward model
    for count in countries:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        def format_func_train(examples):
            kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
            chosen_chat = examples["chosen"][0]["content"] + " [SEP] " + examples["chosen"][1]["content"]
            rejected_chat = examples["rejected"][0]["content"] + " [SEP] " + examples["rejected"][1]["content"]
            chosen_tokens = tokenizer.encode_plus(chosen_chat, **kwargs)
            rejected_tokens = tokenizer.encode_plus(rejected_chat, **kwargs)
            ref_rewards_chosen = utt_train_rewards[examples["chosen_id"]]
            ref_rewards_rejected = utt_train_rewards[examples["rejected_id"]]
            return {
                "input_ids_chosen": chosen_tokens["input_ids"][0],
                "attention_mask_chosen": chosen_tokens["attention_mask"][0],
                "input_ids_rejected": rejected_tokens["input_ids"][0],
                "attention_mask_rejected": rejected_tokens["attention_mask"][0],
                "ref_rewards_chosen": ref_rewards_chosen,
                "ref_rewards_rejected": ref_rewards_rejected,
            }

        def format_func_test(examples):
            kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
            chosen_chat = examples["chosen"][0]["content"] + " [SEP] " + examples["chosen"][1]["content"]
            rejected_chat = examples["rejected"][0]["content"] + " [SEP] " + examples["rejected"][1]["content"]
            chosen_tokens = tokenizer.encode_plus(chosen_chat, **kwargs)
            rejected_tokens = tokenizer.encode_plus(rejected_chat, **kwargs)
            ref_rewards_chosen = utt_test_rewards[examples["chosen_id"]]
            ref_rewards_rejected = utt_test_rewards[examples["rejected_id"]]
            return {
                "input_ids_chosen": chosen_tokens["input_ids"][0],
                "attention_mask_chosen": chosen_tokens["attention_mask"][0],
                "input_ids_rejected": rejected_tokens["input_ids"][0],
                "attention_mask_rejected": rejected_tokens["attention_mask"][0],
                "ref_rewards_chosen": ref_rewards_chosen,
                "ref_rewards_rejected": ref_rewards_rejected,
            }

        train_reward_dataset = Dataset.from_list(utt_pref_edited_train[count]).map(format_func_train)
        train_reward_dataset = train_reward_dataset.shuffle(seed=RANDOM_DATA_SHUFFLE_SEED)
        test_reward_dataset = Dataset.from_list(utt_pref_test[count]).map(format_func_test)
        test_reward_dataset = test_reward_dataset.shuffle(seed=RANDOM_DATA_SHUFFLE_SEED)

        training_args = RewardConfig(
            output_dir=f"{out_path}/trunc/{count}/ours_{EDITED_EPOCHS}epoch/",
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=EDITED_EPOCHS,
            max_steps=MAX_STEPS[count],
            learning_rate=EDITED_LR,
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=REPORT,
            save_strategy="no",
            max_length=2048,
            report_to="none",
            remove_unused_columns=False
        )
        data_collator = OurRewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=training_args.max_length
        )
        trainer = OurRewardTrainer(
            model=model,
            args=training_args,
            beta=WEIGHTING_BETA,
            data_collator=data_collator,
            train_dataset=train_reward_dataset,
            eval_dataset=test_reward_dataset
        )

        trainer.train()

        trainer.save_model(training_args.output_dir)

        rewards = test_baseline_rewards[count]
        print("Scoring the test set!")
        keyed_per_turn = utt_test[count]
        for raw_utts in tqdm(keyed_per_turn.values()):
            for i in range(0, len(raw_utts), BATCH_SIZE):
                utt_batch = raw_utts[i:min(len(raw_utts), i + BATCH_SIZE)]
                q_s = []
                a_s = []
                for utt in utt_batch:
                    q = utt["user_prompt"]
                    a = utt["model_response"]
                    q_s.append(q)
                    a_s.append(a)
                inputs = tokenizer(q_s, a_s, padding=True, return_tensors='pt').to(device)
                scores = model(**inputs).logits.cpu().detach()
                for j in range(len(utt_batch)):
                    utt_id = utt_batch[j]["utterance_id"]
                    score = scores[j]
                    rewards[utt_id] = score

        print("Evaluating the test set!")

        # this evaluation loop can probably be decomposed with the one starting on line 300
        # currently train on the edited set, test on the full set
        pairs = utt_pref_test[count]

        correct = 0
        wrong = 0
        for pair in pairs:
            chosen_score = rewards[pair["chosen_id"]]
            rejected_score = rewards[pair["rejected_id"]]
            if chosen_score > rejected_score:
                correct += 1
            else:
                wrong += 1
        perf = correct * 1.0 / (correct + wrong) * 100
        print(f"{count}: {perf:.2f}%")
        full_perfs[count]["ours_partial"] = perf
        avg_perf += perf

        pairs = utt_pref_edited_test[count]

        correct = 0
        wrong = 0
        for pair in pairs:
            chosen_score = rewards[pair["chosen_id"]]
            rejected_score = rewards[pair["rejected_id"]]

            # model (deberta) gave higher score to preferred response
            if chosen_score > rejected_score:
                correct += 1
            # model (deberta) gave higher score to rejected response
            else:
                wrong += 1
        perf = correct * 1.0 / (correct + wrong) * 100
        print(f"{count}: {perf:.2f}%")
        partial_perfs[count]["ours_partial"] = perf
        avg_partial_perf += perf

    print("**** Ablation reward model Evals ****")
    print("**** Percentage of pairs correctly evaluated ****")
    avg_perf /= len(countries)
    full_perfs["avg"]["ours_partial"] = avg_perf
    print(f"Ablation Model Performance: {avg_perf:.2f}%")

    avg_partial_perf /= len(countries)
    partial_perfs["avg"]["ours_partial"] = avg_partial_perf
    print(f"Ablation Model Partial Performance: {avg_partial_perf:.2f}%")

    import json
    path = f"{out_path}/experiment/seed/{RANDOM_MODEL_SEED}/"
    if not os.path.exists(path):
        os.makedirs(path)

    print("************* OVERALL PERFORMANCE *****************")
    print(json.dumps(full_perfs, indent=4))
    with open(f"{out_path}/experiment/seed/{RANDOM_MODEL_SEED}/full_report.json", 'w') as f:
        json.dump(full_perfs, f, ensure_ascii=False, indent=4)

    print("************* OVERALL Partial PERFORMANCE *****************")
    print(json.dumps(partial_perfs, indent=4))
    with open(f"{out_path}/experiment/seed/{RANDOM_MODEL_SEED}/partial_report.json", 'w') as f:
        json.dump(partial_perfs, f, ensure_ascii=False, indent=4)

    print("************* EDITED PAIRS RATIO *****************")
    print(json.dumps(edited_ratios, indent=4))
    with open(f"{out_path}/experiment/seed/{RANDOM_MODEL_SEED}/edited_ratios.json", 'w') as f:
        json.dump(edited_ratios, f, ensure_ascii=False, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("out_path")
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(args.out_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
