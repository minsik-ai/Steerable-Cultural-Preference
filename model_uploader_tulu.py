import argparse
import os

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str)
parser.add_argument("--hf_out_path", type=str)
args = parser.parse_args()

BASE_ID = "allenai/Llama-3.1-Tulu-3-8B-RM"
TOKEN = os.environ["HF_TOKEN"]

def main(load_path, hf_out_path):
    model = AutoModelForSequenceClassification.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    peft_model = PeftModel.from_pretrained(model, load_path, device_map="auto")

    merged_model = peft_model.merge_and_unload()

    repo_id = hf_out_path
    merged_model.push_to_hub(repo_id, token=TOKEN, max_shard_size="5GB", safe_serialization=True)
    tokenizer.push_to_hub(repo_id, token=TOKEN)


if __name__ == '__main__':
    main(args.load_path, args.hf_out_path)
