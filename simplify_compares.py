from tqdm import tqdm
import json
import pandas as pd
import os


def simplify_comparative_scores(results, model_name=None):
    detect = {}; prefer = {}
    for result in results:
        model = result['model']
        if model not in detect:
            detect[model] = []
        if model not in prefer:
            prefer[model] = []
        
        detect[model].append(result['detection_score'])
        prefer[model].append(result['self_preference'])
        assert result['detection_score'] != result['self_preference']
        print(result['detection_score'], result['self_preference'])
    detect_df, prefer_df = pd.DataFrame(detect), pd.DataFrame(prefer)
    print(detect_df, prefer_df)
    new_col_names = list(detect_df.columns)[:-1]
    new_col_names.append(model_name)
    detect_df.columns = new_col_names
    prefer_df.columns = new_col_names
    return detect_df.mean(axis=0), prefer_df.mean(axis=0)

for dataset in ["cnn", "xsum"]:
    for result_file in os.listdir(f"individual_setting/score_results/{dataset}"):
        if "comparison_results" in result_file and "simple" not in result_file:
            print(result_file.split("_")[0])
            data = json.load(open(f"individual_setting/score_results/{dataset}/{result_file}", "r"))
            detect, prefer = simplify_comparative_scores(data, model_name=result_file.split("_")[0])
            detect.to_csv(f"individual_setting/score_results/{dataset}/{result_file.split('_')[0]}_comparison_results_detect_simple.csv")
            prefer.to_csv(f"individual_setting/score_results/{dataset}/{result_file.split('_')[0]}_comparison_results_prefer_simple.csv")