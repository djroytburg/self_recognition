import sys
from tqdm import tqdm
from data import load_data, save_to_json, load_from_json
from models import (
    get_gpt_recognition_logprobs,
    get_model_choice,
    get_logprobs_choice_with_sources,
    get_gpt_score,
)
from math import exp
from pprint import pprint
from random import shuffle
import json
import pandas as pd

from self_recognition import simplify_compares

# Parse SOURCES from command line
if len(sys.argv) < 3:
    print("Usage: python experiments.py <model_name> <N> <sources> [compare]")
    print("Example: python experiments.py llama3.2-3b-instruct 350 llama3.2-3b-instruct,deepseek-v3-0324 compare")
    sys.exit(1)

MODEL = sys.argv[1]
N = int(sys.argv[2])
SOURCES = sys.argv[3].split(',')
COMPARE = len(sys.argv) > 4 and sys.argv[4] == "compare"

# Only suitable for GPT models
def generate_gpt_logprob_results(
    dataset,
    model,
    starting_idx=0,
    detection_type="detection",
    comparison_type="comparison",
):
    # For retrieving summaries, the specific fine-tuning version isn't needed
    exact_model = model
    model = "gpt35" if model.endswith("gpt35") else model

    responses, articles, keys = load_data(dataset, sources=SOURCES, target_model=MODEL, num_samples=N)
    results = []  # load_from_json(f"results/{model}_results.json")

    for key in tqdm(keys[starting_idx:], desc=f"[generate_gpt_logprob_results] on {model} for {dataset}"):
        article = articles[key]

        source_summary = responses[model][key]
        for other in [s for s in SOURCES if s != model]:
            result = {"key": key, "model": other}
            other_summary = responses[other][key]

            # Detection
            forward_result = get_model_choice(
                source_summary,
                other_summary,
                article,
                detection_type,
                exact_model,
                return_logprobs=True,
            )
            backward_result = get_model_choice(
                other_summary,
                source_summary,
                article,
                detection_type,
                exact_model,
                return_logprobs=True,
            )

            forward_choice = forward_result[0].token
            backward_choice = backward_result[0].token

            result["forward_detection"] = forward_choice
            result["forward_detection_probability"] = exp(forward_result[0].logprob)
            result["backward_detection"] = backward_choice
            result["forward_detection_probability"] = exp(forward_result[0].logprob)

            match (forward_choice, backward_choice):
                case ("1", "2"):
                    result["detection_score"] = 0.5 * (
                        exp(forward_result[0].logprob) + exp(backward_result[0].logprob)
                    )
                case ("2", "1"):
                    result["detection_score"] = 0.5 * (
                        exp(forward_result[1].logprob) + exp(backward_result[1].logprob)
                    )
                case ("1", "1"):
                    result["detection_score"] = 0.5 * (
                        exp(forward_result[0].logprob) + exp(backward_result[1].logprob)
                    )
                case ("2", "2"):
                    result["detection_score"] = 0.5 * (
                        exp(forward_result[1].logprob) + exp(backward_result[0].logprob)
                    )

            # Comparison
            forward_result = get_model_choice(
                source_summary,
                other_summary,
                article,
                comparison_type,
                exact_model,
                return_logprobs=True,
            )
            backward_result = get_model_choice(
                other_summary,
                source_summary,
                article,
                comparison_type,
                exact_model,
                return_logprobs=True,
            )

            forward_choice = forward_result[0].token
            backward_choice = backward_result[0].token

            # If the comparison asked "Which is worse?" then reverse the options
            if comparison_type == "comparison_with_worse":
                forward_choice = "1" if forward_choice == "2" else "2"
                backward_choice = "1" if backward_choice == "2" else "2"

            result["forward_comparison"] = forward_choice
            result["forward_comparison_probability"] = exp(forward_result[0].logprob)
            result["backward_comparison"] = backward_choice
            result["backward_comparison_probability"] = exp(backward_result[0].logprob)

            match (forward_choice, backward_choice):
                case ("1", "2"):
                    result["self_preference"] = 0.5 * (
                        exp(forward_result[0].logprob) + exp(backward_result[0].logprob)
                    )
                case ("2", "1"):
                    result["self_preference"] = 0.5 * (
                        exp(forward_result[1].logprob) + exp(backward_result[1].logprob)
                    )
                case ("1", "1"):
                    result["self_preference"] = 0.5 * (
                        exp(forward_result[0].logprob) + exp(backward_result[1].logprob)
                    )
                case ("2", "2"):
                    result["self_preference"] = 0.5 * (
                        exp(forward_result[1].logprob) + exp(backward_result[0].logprob)
                    )

            results.append(result)
    return results


# Only suitable for GPT models
def generate_gpt_logprob_results_with_sources(
    dataset, model, reversed=False, randomized=False
):
    exact_model = model  # the specific fine-tuning version not needed for retrieval
    model = "gpt35" if model.endswith("gpt35") else model

    responses, articles, keys = load_data(dataset, sources=SOURCES, target_model=MODEL, num_samples=N)
    results = []  # load_from_json(f"prompting_results/{model}_results.json")

    for key in keys:
        article = articles[key]
        source_summary = responses[model][key]

        for other in [s for s in SOURCES if s != model]:
            result = {"key": key, "model": other}
            other_summary = responses[other][key]

            random_labels = [model, other]
            shuffle(random_labels)

            # Comparison
            forward_result = get_logprobs_choice_with_sources(
                source_summary,
                other_summary,
                random_labels[0] if randomized else other if reversed else model,
                random_labels[1] if randomized else model if reversed else other,
                article,
                exact_model,
            )
            backward_result = get_logprobs_choice_with_sources(
                other_summary,
                source_summary,
                random_labels[1] if randomized else model if reversed else other,
                random_labels[0] if randomized else other if reversed else model,
                article,
                exact_model,
            )

            forward_choice = forward_result[0].token
            backward_choice = backward_result[0].token

            if randomized:
                result["random_labels"] = random_labels

            result["forward_comparison"] = forward_choice
            result["forward_probability"] = exp(forward_result[0].logprob)
            result["backward_comparison"] = backward_choice
            result["backward_probability"] = exp(backward_result[0].logprob)

            match (forward_choice, backward_choice):
                case ("1", "2"):
                    result["self_preference"] = 0.5 * (
                        exp(forward_result[0].logprob) + exp(backward_result[0].logprob)
                    )
                case ("2", "1"):
                    result["self_preference"] = 0.5 * (
                        exp(forward_result[1].logprob) + exp(backward_result[1].logprob)
                    )
                case ("1", "1"):
                    result["self_preference"] = 0.5 * (
                        exp(forward_result[0].logprob) + exp(backward_result[1].logprob)
                    )
                case ("2", "2"):
                    result["self_preference"] = 0.5 * (
                        exp(forward_result[1].logprob) + exp(backward_result[0].logprob)
                    )

            results.append(result)
    return results


def generate_score_results(dataset, model, starting_idx=0):
    SCORES = ["1", "2", "3", "4", "5"]

    exact_model = model
    model = "gpt35" if model.endswith("gpt35") else model

    responses, articles, keys = load_data(dataset, sources=SOURCES, target_model=MODEL, num_samples=N)
    results = []

    for key in tqdm(keys[starting_idx:]):
        article = articles[key]
        for target_model in SOURCES:
            summary = responses[target_model][key]

            response = get_gpt_score(summary, article, exact_model)
            result = {i.token: exp(i.logprob) for i in response if i.token in SCORES}
            for score in SCORES:
                if score not in result:
                    result[score] = 0

            results.append(
                {
                    "key": key,
                    "model": model,
                    "target_model": target_model,
                    "scores": result,
                }
            )

    return results


def generate_recognition_results(dataset, model, starting_idx=0):
    exact_model = model
    model = "gpt35" if model.endswith("gpt35") else model

    responses, articles, keys = load_data(dataset, sources=SOURCES, target_model=MODEL, num_samples=N)
    results = []

    for key in tqdm(keys[starting_idx:]):
        article = articles[key]
        for target_model in SOURCES:
            summary = responses[target_model][key]

            res = get_gpt_recognition_logprobs(summary, article, exact_model)
            res = {i.token: exp(i.logprob) for i in res}

            if "Yes" not in res:
                print(key, exact_model, target_model, res)
            else:
                results.append(
                    {
                        "key": key,
                        "model": exact_model,
                        "target_model": target_model,
                        "recognition_score": res["Yes"],
                        "res": res,
                        "ground_truth": int(model == target_model),
                    }
                )

    return results

def simplify_scores(results):
    score = lambda x: [{a['target_model']: sum([int(k) * v for k, v in a['scores'].items()])} for a in results if a['key'] == x]
    keys = list(set([a['key'] for a in results]))
    return pd.DataFrame([[list(v.values())[0] for v in score(key)] for key in keys], columns = SOURCES, index=keys).mean(axis=0)

def simplify_recognition_results(results):
    keys = list(set([a['key'] for a in results]))
    keyset = {}
    for key in keys:
        keyset[key] = [c['recognition_score'] for c in results if c['key'] == key]
    recog_data = pd.DataFrame(keyset).T
    recog_data.columns = SOURCES
    recog_data.index = keys
    return recog_data.mean(axis=0)


# Main execution
for dataset in ["cnn", "xsum"]:
    number_string = '_' + str(N) if N != 1000 else ''
    
    # Individual Scoring (1 to 5) Experiment
    results = generate_score_results(dataset, MODEL, starting_idx=0)
    save_to_json(results, f"individual_setting/score_results/{dataset}/{MODEL}_results{number_string}.json")
    simplify_scores(results).to_csv(f"individual_setting/score_results/{dataset}/{MODEL}_results{number_string}_simple.csv")
    
    # Individual Recognition Experiment
    results = generate_recognition_results(dataset, MODEL, starting_idx=0)
    save_to_json(results, f"individual_setting/score_results/{dataset}/{MODEL}_recognition_results{number_string}.json")
    simplify_recognition_results(results).to_csv(f"individual_setting/score_results/{dataset}/{MODEL}_recognition_results{number_string}_simple.csv")
    
    # Pairwise Recognition AND Preference Experiment
    if COMPARE:
        results = generate_gpt_logprob_results(dataset, MODEL, starting_idx=0)
        base_output_filename = f"individual_setting/score_results/{dataset}/{MODEL}_comparison_results{number_string}"
        save_to_json(results, base_output_filename)
        mean_dc, mean_pc, detect_acc, prefer_rate = simplify_compares(
            results, model_name_being_evaluated=MODEL
        )
        mean_dc.to_csv(f"{base_output_filename}_mean_detect_conf_simple.csv", header=True)
        mean_pc.to_csv(f"{base_output_filename}_mean_prefer_conf_simple.csv", header=True)
        detect_acc.to_csv(f"{base_output_filename}_detect_accuracy_simple.csv", header=True)
        prefer_rate.to_csv(f"{base_output_filename}_self_prefer_rate_simple.csv", header=True)

print(f"Completed experiments for {MODEL}")


# Artifacts from old repo
"""
print("Starting results_with_worse CNN Experiments!")

model = "cnn_2_ft_gpt35"
results = generate_gpt_logprob_results(
    "cnn", model, comparison_type="comparison_with_worse", starting_idx=2
)
save_to_json(results, f"results_with_worse/cnn/{model}_results.json")
print(f"Done with {model}!")

model = "cnn_10_ft_gpt35"
results = generate_gpt_logprob_results(
    "cnn", model, comparison_type="comparison_with_worse", starting_idx=10
)
save_to_json(results, f"results_with_worse/cnn/{model}_results.json")
print(f"Done with {model}!")

models = [
    "cnn_500_ft_gpt35",
    "cnn_always_1_ft_gpt35",
    "cnn_random_ft_gpt35",
    "cnn_readability_ft_gpt35",
    "cnn_length_ft_gpt35",
    "cnn_vowelcount_ft_gpt35",
]

for model in models:
    print(f"Starting {model}")
    results = generate_gpt_logprob_results(
        "cnn", model, comparison_type="comparison_with_worse", starting_idx=500
    )
    save_to_json(results, f"results_with_worse/cnn/{model}_results.json")
    print("Done!")

print("All Done!")
"""

"""
print("Starting XSUM Scoring Experiments!")

model = "cnn_2_ft_gpt35"
results = generate_score_results("xsum", model, starting_idx=2)
save_to_json(results, f"individual_setting/score_results/xsum/{model}_results.json")
print(f"Done with {model}!")

model = "cnn_2_ft_gpt35"
results = generate_score_results("xsum", model, starting_idx=10)
save_to_json(results, f"individual_setting/score_results/xsum/{model}_results.json")
print(f"Done with {model}!")

models = [
    "cnn_500_ft_gpt35",
    "cnn_always_1_ft_gpt35",
    "cnn_random_ft_gpt35",
    "cnn_readability_ft_gpt35",
    "cnn_length_ft_gpt35",
    "cnn_vowelcount_ft_gpt35",
]

for model in models:
    print(f"Starting {model}")
    results = generate_score_results("xsum", model, starting_idx=500)
    save_to_json(results, f"individual_setting/score_results/xsum/{model}_results.json")
    print("Done!")

print("All Done!")


# model = "cnn_10_ft_gpt35"
# results = generate_score_results("cnn", model, starting_idx=10)
# save_to_json(results, f"individual_setting/score_results/cnn/{model}_results.json")
# print("3/5")

# model = "xsum_10_ft_gpt35"
# results = generate_score_results("cnn", model, starting_idx=10)
# save_to_json(results, f"individual_setting/score_results/cnn/{model}_results.json")
# print("4/5")

# model = "cnn_10_ft_gpt35"
# results = generate_score_results("xsum", model, starting_idx=10)
# save_to_json(results, f"individual_setting/score_results/xsum/{model}_results.json")
# print("5/5")

"""
