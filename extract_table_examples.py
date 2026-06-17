"""
Extract examples for Table 13 (Filtering) and Table 14 (Weighting) from the paper.

Table 13: Examples of SCPO Filtering Mechanism
- Shows retained examples (Global RM disagrees with minority → culturally distinctive)
- Shows filtered examples (Global RM agrees with minority → universal preference)

Table 14: Examples of SCPO Weighting Mechanism
- Shows examples across extremeness levels with computed weights
"""

import torch
import json
import collections
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Hyperparameters (matching preferences.py)
BATCH_SIZE = 8
REWARD_THRES = 0.7  # τ threshold for filtering
WEIGHTING_BETA = 1.0  # β temperature for weighting

MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-base"

def compute_filtering_ratio(r_chosen, r_rejected):
    """
    Compute p_glo(y+ > y-|x) = exp(r+) / (exp(r+) + exp(r-))
    This is the probability that global model prefers chosen over rejected.
    """
    p_plus = torch.exp(r_chosen) / (torch.exp(r_chosen) + torch.exp(r_rejected))
    return p_plus.item()

def compute_weight(r_chosen, r_rejected, beta=WEIGHTING_BETA):
    """
    Compute weight W(y+, y-) = min(exp((r+ - r-) / β), 1)
    Lower weight for more extreme preferences (where global RM strongly disagrees).
    """
    weight = torch.minimum(
        torch.exp((r_chosen - r_rejected) / beta),
        torch.ones_like(r_chosen)
    )
    return weight.item()

def is_retained_by_filter(r_chosen, r_rejected, tau=REWARD_THRES):
    """
    Check if pair should be retained (True) or filtered out (False).
    Retained if p_glo(y+ > y-|x) < τ (global RM disagrees with minority preference)
    """
    p_plus = compute_filtering_ratio(r_chosen, r_rejected)
    return p_plus < tau

def extract_table_examples(out_path="./table_examples"):
    """
    Extract examples for Table 13 and Table 14 from PRISM dataset.
    """
    import os
    os.makedirs(out_path, exist_ok=True)

    # Countries used in the paper
    countries = ["Chile", "South Africa", "New Zealand", "Australia", "Mexico", "Israel", "Canada"]

    print("Loading datasets...")
    utts = load_dataset("HannahRoseKirk/prism-alignment", "utterances", split="train")
    infos = load_dataset("HannahRoseKirk/prism-alignment", "survey", split="train")

    # Split by users
    info_split = infos.train_test_split(test_size=0.15, seed=0)
    info_train = info_split["train"]

    # Map users to countries
    info_train_ids = collections.defaultdict(set)
    for info in info_train:
        country = info["location"]["reside_country"]
        if country in countries:
            info_train_ids[country].add(info["user_id"])

    # Organize utterances by country and interaction
    utt_train = {country: collections.defaultdict(list) for country in countries}
    for utt in utts:
        user_id = utt["user_id"]
        turn = utt["interaction_id"]
        for country in countries:
            key = f"{user_id}_{turn}"
            if user_id in info_train_ids[country]:
                utt_train[country][key].append(utt)

    # Load model and tokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    # Score all utterances
    print("Scoring utterances with global reward model...")
    utt_rewards = {}
    for country, keyed_per_turn in utt_train.items():
        print(f"Processing {country}...")
        for raw_utts in tqdm(keyed_per_turn.values()):
            for i in range(0, len(raw_utts), BATCH_SIZE):
                utt_batch = raw_utts[i:min(len(raw_utts), i + BATCH_SIZE)]
                q_s = [utt["user_prompt"] for utt in utt_batch]
                a_s = [utt["model_response"] for utt in utt_batch]

                with torch.no_grad():
                    inputs = tokenizer(q_s, a_s, padding=True, truncation=True,
                                       max_length=512, return_tensors='pt').to(device)
                    scores = model(**inputs).logits.cpu()

                for j, utt in enumerate(utt_batch):
                    utt_rewards[utt["utterance_id"]] = scores[j]

    # Collect examples for tables
    table_13_retained = []  # Culturally distinctive (filtered IN)
    table_13_filtered = []  # Universal preferences (filtered OUT)
    table_14_examples = []  # All examples with weights

    print("Extracting preference pair examples...")
    for country, keyed_per_turn in utt_train.items():
        for raw_utts in keyed_per_turn.values():
            for i, utt1 in enumerate(raw_utts):
                for j in range(i + 1, len(raw_utts)):
                    utt2 = raw_utts[j]

                    # Get human preference scores
                    utt1s = utt1["score"]
                    utt2s = utt2["score"]

                    if utt1s == utt2s:
                        continue  # Skip ties

                    # Determine chosen/rejected based on human preference
                    if utt1s > utt2s:
                        chosen, rejected = utt1, utt2
                    else:
                        chosen, rejected = utt2, utt1

                    # Get global RM rewards
                    r_chosen = utt_rewards[chosen["utterance_id"]]
                    r_rejected = utt_rewards[rejected["utterance_id"]]

                    # Compute metrics
                    ratio_F = compute_filtering_ratio(r_chosen, r_rejected)
                    weight = compute_weight(r_chosen, r_rejected, WEIGHTING_BETA)
                    retained = is_retained_by_filter(r_chosen, r_rejected, REWARD_THRES)

                    example = {
                        "country": country,
                        "user_prompt": chosen["user_prompt"][:200],  # Truncate for display
                        "preferred_response": chosen["model_response"][:200],
                        "rejected_response": rejected["model_response"][:200],
                        "r_chosen": round(r_chosen.item(), 2),
                        "r_rejected": round(r_rejected.item(), 2),
                        "ratio_F": round(ratio_F, 2),
                        "weight_W": round(weight, 2),
                        "retained": retained,
                        "human_score_chosen": chosen["score"],
                        "human_score_rejected": rejected["score"],
                    }

                    # Categorize for Table 13
                    if retained:
                        table_13_retained.append(example)
                    else:
                        table_13_filtered.append(example)

                    # All examples for Table 14
                    table_14_examples.append(example)

    # Sort and select diverse examples
    print("\nSelecting representative examples...")

    # Table 13: Select examples around F values 0.0, 0.25, 0.50, 0.75, 1.00
    # Include all countries once per F value
    table_13_output = {"retained": [], "filtered": []}
    target_F_values = [0.0, 0.25, 0.50, 0.75, 1.00]

    for target_F in target_F_values:
        for country in countries:
            # Retained examples (culturally distinctive)
            country_retained = [e for e in table_13_retained if e["country"] == country]
            if country_retained:
                # Find example closest to target F value
                closest = min(country_retained, key=lambda x: abs(x["ratio_F"] - target_F))
                table_13_output["retained"].append(closest)

            # Filtered examples (universal preferences)
            country_filtered = [e for e in table_13_filtered if e["country"] == country]
            if country_filtered:
                # Find example closest to target F value
                closest = min(country_filtered, key=lambda x: abs(x["ratio_F"] - target_F))
                table_13_output["filtered"].append(closest)

    # Table 14: Select examples around W values 0.0, 0.25, 0.50, 0.75, 1.00
    # Include all countries once per W value
    table_14_output = []
    target_W_values = [0.0, 0.25, 0.50, 0.75, 1.00]

    for target_W in target_W_values:
        for country in countries:
            country_examples = [e for e in table_14_examples if e["country"] == country]
            if country_examples:
                # Find example closest to target W value
                closest = min(country_examples, key=lambda x: abs(x["weight_W"] - target_W))
                table_14_output.append(closest)

    # Sort Table 14 by weight for presentation
    table_14_output.sort(key=lambda x: -x["weight_W"])

    # Save results
    with open(f"{out_path}/table_13_examples.json", 'w') as f:
        json.dump(table_13_output, f, indent=2, ensure_ascii=False)

    with open(f"{out_path}/table_14_examples.json", 'w') as f:
        json.dump(table_14_output, f, indent=2, ensure_ascii=False)

    # Print summary for paper
    print("\n" + "="*80)
    print("TABLE 13: SCPO Filtering Mechanism Examples")
    print("="*80)

    print("\n--- RETAINED Examples (Global RM disagrees → culturally distinctive) ---")
    for ex in table_13_output["retained"][:3]:
        print(f"\nCountry: {ex['country']}")
        print(f"  Prompt: {ex['user_prompt'][:80]}...")
        print(f"  Preferred: {ex['preferred_response'][:80]}...")
        print(f"  Rejected: {ex['rejected_response'][:80]}...")
        print(f"  r_chosen: {ex['r_chosen']}, r_rejected: {ex['r_rejected']}, F: {ex['ratio_F']}")

    print("\n--- FILTERED Examples (Global RM agrees → universal preference) ---")
    for ex in table_13_output["filtered"][:3]:
        print(f"\nCountry: {ex['country']}")
        print(f"  Prompt: {ex['user_prompt'][:80]}...")
        print(f"  Preferred: {ex['preferred_response'][:80]}...")
        print(f"  Rejected: {ex['rejected_response'][:80]}...")
        print(f"  r_chosen: {ex['r_chosen']}, r_rejected: {ex['r_rejected']}, F: {ex['ratio_F']}")

    print("\n" + "="*80)
    print("TABLE 14: SCPO Weighting Mechanism Examples")
    print("="*80)

    for ex in table_14_output[:7]:
        print(f"\nCountry: {ex['country']}")
        print(f"  Prompt: {ex['user_prompt'][:60]}...")
        print(f"  Preferred: {ex['preferred_response'][:60]}...")
        print(f"  Rejected: {ex['rejected_response'][:60]}...")
        print(f"  r_chosen: {ex['r_chosen']}, r_rejected: {ex['r_rejected']}, W: {ex['weight_W']}")

    print(f"\n\nResults saved to {out_path}/")
    print(f"  - table_13_examples.json ({len(table_13_output['retained'])} retained, {len(table_13_output['filtered'])} filtered)")
    print(f"  - table_14_examples.json ({len(table_14_output)} examples)")

    return table_13_output, table_14_output


def generate_latex_table_13(examples):
    """Generate LaTeX code for Table 13."""
    latex = []
    latex.append("\\begin{table*}")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{p{1.2cm}p{3cm}p{3cm}p{3cm}ccccc}")
    latex.append("\\toprule")
    latex.append("Country & User Prompt & Preferred Response $y^+$ & Rejected Response $y^-$ & $r^+$ & $r^-$ & $s^+$ & $s^-$ & $F$ \\\\")
    latex.append("\\midrule")
    latex.append("\\multicolumn{9}{l}{\\textit{Retained Examples (Global RM disagrees with minority $\\rightarrow$ culturally distinctive)}} \\\\")

    for ex in examples["retained"][:3]:
        prompt = ex["user_prompt"][:80].replace("&", "\\&").replace("%", "\\%")
        pref = ex["preferred_response"][:80].replace("&", "\\&").replace("%", "\\%")
        rej = ex["rejected_response"][:80].replace("&", "\\&").replace("%", "\\%")
        latex.append(f"{ex['country']} & \"{prompt}...\" & \"{pref}...\" & \"{rej}...\" & {ex['r_chosen']} & {ex['r_rejected']} & {ex['human_score_chosen']} & {ex['human_score_rejected']} & {ex['ratio_F']} \\\\")

    latex.append("\\midrule")
    latex.append("\\multicolumn{9}{l}{\\textit{Filtered Examples (Global RM agrees with minority $\\rightarrow$ universal preference)}} \\\\")

    for ex in examples["filtered"][:3]:
        prompt = ex["user_prompt"][:80].replace("&", "\\&").replace("%", "\\%")
        pref = ex["preferred_response"][:80].replace("&", "\\&").replace("%", "\\%")
        rej = ex["rejected_response"][:80].replace("&", "\\&").replace("%", "\\%")
        latex.append(f"{ex['country']} & \"{prompt}...\" & \"{pref}...\" & \"{rej}...\" & {ex['r_chosen']} & {ex['r_rejected']} & {ex['human_score_chosen']} & {ex['human_score_rejected']} & {ex['ratio_F']} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Examples of SCPO Filtering Mechanism.}")
    latex.append("\\label{tab:filtering_examples}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


def generate_latex_table_14(examples):
    """Generate LaTeX code for Table 14."""
    latex = []
    latex.append("\\begin{table*}")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{p{1.2cm}p{3cm}p{3cm}p{3cm}ccccc}")
    latex.append("\\toprule")
    latex.append("Country & User Prompt & Preferred Response $y^+$ & Rejected Response $y^-$ & $r^+$ & $r^-$ & $s^+$ & $s^-$ & $W$ \\\\")
    latex.append("\\midrule")

    for ex in examples[:7]:
        prompt = ex["user_prompt"][:80].replace("&", "\\&").replace("%", "\\%")
        pref = ex["preferred_response"][:80].replace("&", "\\&").replace("%", "\\%")
        rej = ex["rejected_response"][:80].replace("&", "\\&").replace("%", "\\%")
        latex.append(f"{ex['country']} & \"{prompt}...\" & \"{pref}...\" & \"{rej}...\" & {ex['r_chosen']} & {ex['r_rejected']} & {ex['human_score_chosen']} & {ex['human_score_rejected']} & {ex['weight_W']} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Examples of SCPO Weighting Mechanism across extremeness levels.}")
    latex.append("\\label{tab:weighting_examples}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


if __name__ == "__main__":
    table_13, table_14 = extract_table_examples()

    # Generate LaTeX
    print("\n" + "="*80)
    print("LATEX for Table 13:")
    print("="*80)
    print(generate_latex_table_13(table_13))

    print("\n" + "="*80)
    print("LATEX for Table 14:")
    print("="*80)
    print(generate_latex_table_14(table_14))
