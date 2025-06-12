import json
from data import load_data, save_to_json, TARGET, N, SOURCES
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
parser.add_argument("--N", type=int, default=1000)
parser.add_argument("--overwrite", action="store_true", default=False)
args = parser.parse_args()

TARGET = args.target
N = args.N

xsum_responses, xsum_articles, xsum_keys = load_data("xsum", sources = SOURCES)
cnn_responses, cnn_articles, cnn_keys = load_data("cnn", sources = SOURCES)

main_models = [TARGET]
xsum_models_gpt35 = [
    "xsum_2_ft_gpt35",
    "xsum_10_ft_gpt35",
    "xsum_500_ft_gpt35",
    "xsum_always_1_ft_gpt35",
    "xsum_random_ft_gpt35",
    "xsum_readability_ft_gpt35",
    "xsum_length_ft_gpt35",
    "xsum_vowelcount_ft_gpt35",
]
cnn_models_gpt35 = [
    "cnn_2_ft_gpt35",
    "cnn_10_ft_gpt35",
    "cnn_500_ft_gpt35",
    "cnn_always_1_ft_gpt35",
    "cnn_random_ft_gpt35",
    "cnn_readability_ft_gpt35",
    "cnn_length_ft_gpt35",
    "cnn_vowelcount_ft_gpt35",
]

xsum_models_llama = [
    "xsum_2_ft_llama",
    "xsum_10_ft_llama",
    "xsum_500_ft_llama",
    "xsum_always_1_ft_llama",
    "xsum_random_ft_llama",
    "xsum_readability_ft_llama",
    "xsum_length_ft_llama",
    "xsum_vowelcount_ft_llama",
]
cnn_models_llama = [
    "cnn_2_ft_llama",
    "cnn_10_ft_llama",
    "cnn_500_ft_llama",
    "cnn_always_1_ft_llama",
    "cnn_random_ft_llama",
    "cnn_readability_ft_llama",
    "cnn_length_ft_llama",
    "cnn_vowelcount_ft_llama",
]

# models = (
#     main_models
#     + xsum_models_gpt35
#     + cnn_models_gpt35
#     + xsum_models_llama
#     + cnn_models_llama
# )

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
    # print(model)
    # if "llama" in model.lower() and False:
    #     results = {}
    #     pipe = pipeline("text-generation", model=model, token=os.getenv("HF_TOKEN"), device_map="auto")
    #     pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    #     pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    #     xsum_pipe_data = preprocess_summary_data("xsum", [xsum_articles[k] for k in xsum_keys], pipe)
    #     xsum_summaries = []
    #     for output in pipe(xsum_pipe_data,
    #                     max_new_tokens=100, # Set max_new_tokens for generation
    #                     return_full_text=False # Return only the generated part
    #                    ):
    #         generated_text = output[0]["generated_text"]
    #         print(generated_text)
    #         xsum_summaries.append(generated_text) # Store the raw generated text
    #     for k, v in zip(xsum_keys, xsum_summaries):
    #         results[k] = v        

    #     save_to_json(results, f"summaries/xsum/{model}_responses.json")
    #     results = {}
        

    #     cnn_pipe_data = preprocess_summary_data("cnn", [cnn_articles[k] for k in cnn_keys], pipe)
    #     cnn_summaries = []
    #     for output in pipe(cnn_pipe_data,
    #                     max_new_tokens=100, # Set max_new_tokens for generation
    #                     batch_size=8,      # Process in batches for efficiency
    #                     return_full_text=False # Return only the generated part
    #                    ):
    #         generated_text = output[0]["generated_text"]
    #         print(generated_text)
    #         cnn_summaries.append(generated_text)
    #     for k, v in zip(cnn_keys, cnn_summaries):
    #         results[k] = v
    #     save_to_json(results, f"summaries/cnn/{model}_responses.json")
    #     continue
    # if args.overwrite or f"xsum_train_{model}_responses{'_' + str(N) if N != 1000 else ''}.json" not in os.listdir("summaries/xsum"):
    #     results = {}
    #     for key in tqdm(xsum_keys[:N]):
    #         results[key] = get_summary(xsum_articles[key], "xsum", model)[0]
    #         # print(key)
    #         # print(results[key])
    #         save_to_json(results, f"summaries/xsum/xsum_train_{model}_responses{'_' + str(N) if N != 1000 else ''}.json")
    # else: 
    #     print(f"xsum_train_{model}_responses{'_' + str(N) if N != 1000 else ''}.json already exists")
    results = {}
    ambivalent_keys = json.load(open(f"ambivalent_keys_cnn.json", "r"))
    for key in tqdm(ambivalent_keys):
        results[key] = get_summary(cnn_articles[key], "cnn", model)[0]
        # print(key)
        # print(results[key])[0]
        save_to_json(results, f"summaries/cnn/cnn_train_{model}_ambivalents.json")
    print(model, "done!")

print("Done!")
