import os
import re
import glob
import json
from datasets import load_dataset
from models import GPT_MODEL_ID, code_datasets
from generate_summaries import process_dataset


def save_to_json(dictionary, file_name, force_overwrite=True):
    """Save a dictionary to a JSON file, creating directories if needed."""
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    if not force_overwrite and os.path.exists(file_name):
        return

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    """Load a dictionary from a JSON file."""
    with open(file_name, "r") as f:
        return json.load(f)


def load_data(dataset, sources, target_model, num_samples, load_summaries=True, extras=False, logger=None):
    """
    Load summaries and articles for a given dataset and set of sources.
    Returns (responses, articles, keys).
    """
    responses = {}
        
    print(num_samples)
    data_type = "articles" if dataset not in code_datasets else "code"
    if load_summaries:
        for source in sources:
            print(f"[DEBUG] Loading data for source: {source}")
            merged_file = f"summaries/{dataset}/{dataset}_train_{source}_responses_merged.json"
            print(f"[DEBUG] Checking merged file: {merged_file}")
            if os.path.exists(merged_file):
                print(f"[DEBUG] Merged file exists, loading...")
                summaries = load_from_json(merged_file)
                print(f"[DEBUG] Loaded {len(summaries)} samples from merged file")
                if len(summaries) < num_samples:
                    print(f"[DEBUG] Not enough samples: {len(summaries)} < {num_samples}")
                    raise FileNotFoundError(f"Merged {data_type} file for {source} has only {len(summaries)} samples, but {num_samples} requested.")
                print(f"[DEBUG] Using merged file for {source}")
                responses[source] = summaries
            else:
                # Fallback: find the best available non-merged file with at least num_samples
                pattern = f"summaries/{dataset}/{dataset}_train_{source}_responses*{'_extra' if extras else ''}.json"
                files = glob.glob(pattern)
                best_file = None
                for file in files:
                    with open(file, "r") as f:
                        data = json.load(f)
                    if len(data) >= num_samples:
                        best_file = file
                        break
                if best_file is None:
                    if logger is not None:
                        logger.warning(f"No suitable {data_type} file found for {source} with at least {num_samples} samples. Generating now.")
                    else:
                        print(f"No suitable {data_type} file found for {source} with at least {num_samples} samples. Generating now.")
                    process_dataset(dataset, source, num_samples)
                    # After generation, check for the merged file first, then fallback files
                    merged_file = f"summaries/{dataset}/{dataset}_train_{source}_responses_merged.json"
                    if os.path.exists(merged_file):
                        best_file = merged_file
                    else:
                        # Check for newly generated files
                        pattern = f"summaries/{dataset}/{dataset}_train_{source}_responses*{'_extra' if extras else ''}.json"
                        files = glob.glob(pattern)
                        for file in files:
                            with open(file, "r") as f:
                                data = json.load(f)
                            if len(data) >= num_samples:
                                best_file = file
                                break
                        if best_file is None:
                            raise FileNotFoundError(f"Failed to generate or find suitable {data_type} file for {source} with at least {num_samples} samples.")
                responses[source] = load_from_json(best_file)
    articles = load_from_json(f"{data_type}/{dataset}_train_{data_type}{'_extra' if extras else ''}.json")
    if target_model in sources:
        all_keys = list(responses[target_model].keys())
        keys = all_keys[:num_samples]
    elif load_summaries:
        raise Exception("Model not found!", target_model)
    else:
        keys = list(articles.keys())
    return responses, articles, keys


def load_cnn_dailymail_data():
    """
    Load the CNN/DailyMail dataset splits.
    Returns (train_data, test_data, validation_data).
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]
    return train_data, test_data, validation_data


def load_xsum_data():
    """
    Load the XSum dataset splits.
    Returns (train_data, test_data, validation_data).
    """
    dataset = load_dataset("EdinburghNLP/xsum")
    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]
    return train_data, test_data, validation_data


def write_to_jsonl_for_finetuning(
    questions, answers, system_prompt, file_name="finetuningdata.jsonl"
):
    """
    Write question/answer pairs and a system prompt to a JSONL file for finetuning.
    """
    formatted_data = ""
    for question, answer in zip(questions, answers):
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
        formatted_data += json.dumps(entry) + "\n"
    with open(file_name, "w") as file:
        file.write(formatted_data)
