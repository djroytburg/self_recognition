from tqdm import tqdm
import json
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def simplify_compares(results, model_name_being_evaluated=None):
    """
    Processes comparison results to calculate mean confidence scores,
    detection accuracy, and self-preference rate for the model_name_being_evaluated
    when compared against various other models.

    Args:
        results (list): A list of dictionaries, where each dict is a comparison result.
                        Each result should contain:
                        - 'model': The name of the 'other' model in the comparison.
                        - 'detection_score': Confidence score for the detection task.
                        - 'self_preference': Confidence score for the preference task.
                        - 'forward_detection': The choice ('1' or '2') in the forward detection.
                                               '1' indicates source_summary (model_name_being_evaluated's) was chosen.
                        - 'backward_detection': The choice ('1' or '2') in the backward detection.
                                                '2' indicates source_summary (model_name_being_evaluated's) was chosen.
                        - 'forward_comparison': The choice ('1' or '2') in the forward comparison,
                                                adjusted for "worse" questions. '1' indicates source_summary was preferred.
        model_name_being_evaluated (str, optional): The name of the model whose results are being processed.
                                                   Used for potential future enhancements or logging.

    Returns:
        tuple: Contains four pandas Series:
            - mean_detect_confidence: Mean detection confidence against each 'other' model.
            - mean_prefer_confidence: Mean self-preference confidence against each 'other' model.
            - detection_accuracy: Detection accuracy against each 'other' model, considering both
                                  forward and backward detection passes as opportunities.
            - self_preference_rate: Self-preference rate against each 'other' model.
    """
    
    detect_confidences = {}
    prefer_confidences = {}
    correct_detection_counts = {} # Sum of correct detections in both forward and backward passes
    self_preference_counts = {}
    total_individual_comparisons = {} # Counts unique (source_summary, other_summary) pairs

    for result in results:
        other_model = result['model'] 

        if other_model not in total_individual_comparisons:
            detect_confidences[other_model] = []
            prefer_confidences[other_model] = []
            correct_detection_counts[other_model] = 0
            self_preference_counts[other_model] = 0
            total_individual_comparisons[other_model] = 0
        if result.get("forward_detection") == result.get("backward_detection"):
            continue
        total_individual_comparisons[other_model] += 1
        detect_confidences[other_model].append(result['detection_score'])
        prefer_confidences[other_model].append(result['self_preference'])
        
        
        # Check for correct detection in the forward pass
        if result.get('forward_detection') == '1':
            correct_detection_counts[other_model] += 1
        
        # Check for correct detection in the backward pass
        # (source_summary is choice '2' in the backward pass configuration)
        if result.get('backward_detection') == '2':
            correct_detection_counts[other_model] += 1
            
        # Check for self-preference
        if result.get('forward_comparison') == '1':
            self_preference_counts[other_model] += 1
        
        if result.get('backward_comparison') == '2':
            self_preference_counts[other_model] += 1
    
    mean_detect_confidence_data = {model: pd.Series(scores).mean() for model, scores in detect_confidences.items()}
    mean_prefer_confidence_data = {model: pd.Series(scores).mean() for model, scores in prefer_confidences.items()}

    mean_detect_confidence = pd.Series(mean_detect_confidence_data, name="mean_detection_confidence")
    mean_prefer_confidence = pd.Series(mean_prefer_confidence_data, name="mean_self_preference_confidence")

    detection_accuracy_data = {}
    for model, count in correct_detection_counts.items():
        # Each individual comparison offers two detection opportunities (forward and backward)
        num_detection_opportunities = total_individual_comparisons.get(model, 0) * 2
        detection_accuracy_data[model] = count / num_detection_opportunities if num_detection_opportunities > 0 else 0.0
    detection_accuracy = pd.Series(detection_accuracy_data, name="detection_accuracy")
    
    self_preference_rate_data = {}
    for model, count in self_preference_counts.items():
        # Self-preference is one outcome per individual comparison
        total_prefs = total_individual_comparisons.get(model, 0) * 2
        self_preference_rate_data[model] = count / total_prefs if total_prefs > 0 else 0.0
    self_preference_rate = pd.Series(self_preference_rate_data, name="self_preference_rate")
    return mean_detect_confidence, mean_prefer_confidence, detection_accuracy, self_preference_rate

# Main processing loop
if __name__ == "__main__":
    for dataset in ["cnn", "xsum"]:
        output_dir = f"individual_setting/score_results/{dataset}"
        os.makedirs(output_dir, exist_ok=True)

        input_dir = f"individual_setting/score_results/{dataset}"

        for result_file in os.listdir(input_dir):
            if "comparison_results" in result_file and "simple" not in result_file and result_file.endswith(".json"):
                # Extract model and number string (if present)
                match = re.match(r'([^_]+)_comparison_results(_\d+)?\.json', result_file)
                if match:
                    model_being_evaluated = match.group(1)
                    number_string = match.group(2) or ''
                else:
                    # fallback to old logic
                    model_being_evaluated = result_file.split("_")[0]
                    number_string = ''

                print(f"Processing {result_file} for model: {model_being_evaluated} in dataset: {dataset}")
                
                file_path = os.path.join(input_dir, result_file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
                    continue
                except FileNotFoundError:
                    print(f"Warning: File {file_path} not found. Skipping.")
                    continue

                if not data: 
                    print(f"Warning: No data in {file_path}. Skipping.")
                    continue

                mean_dc, mean_pc, detect_acc, prefer_rate = simplify_compares(
                    data, model_name_being_evaluated=model_being_evaluated
                )
                
                base_output_filename = os.path.join(output_dir, f"{model_being_evaluated}_comparison_results{number_string}")
                
                mean_dc.to_csv(f"{base_output_filename}_mean_detect_conf_simple.csv", header=True)
                mean_pc.to_csv(f"{base_output_filename}_mean_prefer_conf_simple.csv", header=True)
                detect_acc.to_csv(f"{base_output_filename}_detect_accuracy_simple.csv", header=True)
                prefer_rate.to_csv(f"{base_output_filename}_self_prefer_rate_simple.csv", header=True)
                
                print(f"Finished processing and saving metrics for {model_being_evaluated} from {result_file}")

    print("All processing complete.")

def make_heatmap_matrix(dataset, n, metric='self_preference_rate'):
    """
    Generate a heatmap matrix of self-preference or detection accuracy scores.
    Args:
        dataset (str): 'cnn' or 'xsum'
        n (int): number of samples (for file name matching)
        metric (str): 'self_preference_rate' or 'detection_accuracy'
    """
    import glob
    import os
    import pandas as pd
    from matplotlib import pyplot as plt
    import numpy as np

    # Find all *_comparison_results_{n}.json files
    score_dir = f"individual_setting/score_results/{dataset}"
    heatmap_dir = f"individual_setting/score_results/heatmaps/{dataset}"
    os.makedirs(heatmap_dir, exist_ok=True)
    pattern = os.path.join(score_dir, f"*_comparison_results{'_' + str(n) if n != 1000 else ''}.json")
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for pattern: {pattern}")
        return
    # Get all model names (evaluators)
    evaluators = []
    all_models = set()
    data_dict = {}
    for file in files:
        # Extract model name
        import re
        match = re.match(r'.*/([^/_]+)_comparison_results(_\d+)?\.json', file)
        if match:
            evaluator = match.group(1)
            evaluators.append(evaluator)
        else:
            continue
        # Load the relevant metric
        with open(file, 'r') as f:
            data = json.load(f)
        # Each entry in data is a dict with 'model' and metrics
        for entry in data:
            model = entry['model']
            all_models.add(model)
            if evaluator not in data_dict:
                data_dict[evaluator] = {}
            if metric == 'self_preference_rate':
                # We'll count up and average later
                if model not in data_dict[evaluator]:
                    data_dict[evaluator][model] = []
                # Compute self-preference as in simplify_compares
                # But here, we expect the metric to be in the output file
                # We'll fill this below
            elif metric == 'detection_accuracy':
                if model not in data_dict[evaluator]:
                    data_dict[evaluator][model] = []
            # Store the entry for later
            data_dict[evaluator][model].append(entry)
    all_models = sorted(list(all_models))
    evaluators = sorted(list(set(evaluators)))
    # Now, for each evaluator, load the corresponding *_self_prefer_rate_simple.csv or *_detect_accuracy_simple.csv
    matrix = pd.DataFrame(index=evaluators, columns=all_models, dtype=float)
    for evaluator in evaluators:
        # Find the correct csv file
        base = os.path.join(score_dir, f"{evaluator}_comparison_results{'_' + str(n) if n != 1000 else ''}")
        if metric == 'self_preference_rate':
            csv_file = base + '_self_prefer_rate_simple.csv'
        elif metric == 'detection_accuracy':
            csv_file = base + '_detect_accuracy_simple.csv'
        else:
            raise ValueError('Unknown metric')
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping {evaluator}")
            continue
        df = pd.read_csv(csv_file, index_col=0, header=0)
        # The index/columns are the models being compared
        for model in all_models:
            if model == evaluator:
                matrix.loc[evaluator, model] = np.nan
            elif model in df.index:
                matrix.loc[evaluator, model] = df.loc[model].values[0]
            elif model in df.columns:
                matrix.loc[evaluator, model] = df[model].values[0]
            else:
                matrix.loc[evaluator, model] = np.nan
    # Save matrix as CSV
    models_str = '-'.join(evaluators)
    matrix_csv = os.path.join(heatmap_dir, f"{metric}_{dataset}_heatmap_{n}_{models_str}.csv")
    matrix.to_csv(matrix_csv)
    # Plot heatmap
    plt.figure(figsize=(max(8, len(all_models)), max(6, len(evaluators))))
    plt.rcParams["font.family"] = "Helvetica"
    cmap = plt.get_cmap('hot')
    im = plt.imshow(matrix.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label=metric.replace('_', ' ').title())
    plt.xticks(ticks=np.arange(len(all_models)), labels=all_models, rotation=45, ha='right', fontsize=12)
    plt.yticks(ticks=np.arange(len(evaluators)), labels=evaluators, fontsize=12)
    plt.title(f"{metric.replace('_', ' ').title()} Heatmap ({dataset}, N={n})", fontsize=16)
    # Annotate cells
    for i in range(len(evaluators)):
        for j in range(len(all_models)):
            if i == j:
                continue
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val > 0.5 else 'black', fontsize=10)
    plt.tight_layout()
    matrix_pdf = os.path.join(heatmap_dir, f"{metric}_{dataset}_heatmap_{n}_{models_str}.pdf")
    plt.savefig(matrix_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap matrix to {matrix_csv} and {matrix_pdf}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate heatmap matrix for self-preference or detection accuracy.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset (cnn or xsum)')
    parser.add_argument('--n', type=int, required=True, help='Number of samples (for file name matching)')
    parser.add_argument('--metric', type=str, default='self_preference_rate', choices=['self_preference_rate', 'detection_accuracy'], help='Metric to plot')
    args = parser.parse_args()
    make_heatmap_matrix(args.dataset, args.n, args.metric)