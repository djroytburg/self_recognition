from data import load_data, save_to_json
from models import get_summary
from time import sleep
from transformers import pipeline
from tqdm import tqdm
import os
import sys
from prompts import DATASET_SYSTEM_PROMPTS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("target", type=str)
parser.add_argument("-N", type=int, default=1000)
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument("--extras", action="store_true", default=False)
args = parser.parse_args()

TARGET = args.target
N = args.N or 1000
SOURCES = ['human']
xsum_responses, xsum_articles, xsum_keys = load_data("xsum", sources = SOURCES, target_model=TARGET, num_samples=N, extras=args.extras)
cnn_responses, cnn_articles, cnn_keys = load_data("cnn", sources = SOURCES, target_model=TARGET, num_samples=N, extras=args.extras)
main_models = [TARGET]

def preprocess_summary_data(dataset_name, dataset, pipe):
    preprocessed_data = []
    for data in dataset:
        messages = [
            {"role": "system", "content": DATASET_SYSTEM_PROMPTS[dataset_name]},
            {
                "role": "user",
                "content": f"Article:\n{data}\n\nProvide only the summary with no other text.",
            },
        ]
        tokenizer = pipe.tokenizer
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        preprocessed_data.append(formatted_prompt)
    return preprocessed_data


print("Starting...")
for model in main_models:
    if args.overwrite or f"xsum_train_{model}_responses{'_' + str(N) if N != 1000 else ''}.json" not in os.listdir("summaries/xsum"):
        results = {}
        for key in tqdm(xsum_keys[:N]):
            results[key] = get_summary(xsum_articles[key], "xsum", model)[0]
            # print(key)
            # print(results[key])
            save_to_json(results, f"summaries/xsum/xsum_train_{model}_responses{'_' + str(N) if N != 1000 else ''}.json")
    else: 
        print(f"xsum_train_{model}_responses{'_' + str(N) if N != 1000 else ''}{'_extra' if args.extras else ''}.json already exists")
    if args.overwrite or f"cnn_train_{model}_responses{'_' + str(N) if N != 1000 else ''}{'_extra' if args.extras else ''}.json" not in os.listdir("summaries/cnn"):
        results = {}
        for key in tqdm(cnn_keys[:N]):
            results[key] = get_summary(cnn_articles[key], "cnn", model)[0]
            # print(key)
            # print(results[key])[0]
            save_to_json(results, f"summaries/cnn/cnn_train_{model}_responses{'_' + str(N) if N != 1000 else ''}{'_extra' if args.extras else ''}.json")
    else: 
        print(f"cnn_train_{model}_responses{'_' + str(N) if N != 1000 else ''}{'_extra' if args.extras else ''}.json already exists")
    print(model, "done!")

print("Done!")
