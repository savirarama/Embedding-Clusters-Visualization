import json
from get_modified_files import get_modified_files_from_commit
from tqdm import tqdm
from typing import List
import argparse
import glob
import os


def load_data_from_patterns(patterns: List[str]) -> List[dict]:
    all_data = []
    for pattern in patterns:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                all_data.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return all_data

def map_inducing_commits_to_target(data, repo):
    mapping = {}
    for entry in data:
        expexted_file_path = ""
        for induce_hash in entry.get("induceCommitHashList", []):
            expected_file_paths = glob.glob(f"../GitCF/experiment_data/ishida/{repo}/*/{entry.get("fixIssueID")}/_expected.json")
            for file_path in expected_file_paths:
                if os.path.exists(file_path):
                    expected_file_path = file_path
                    break
            expected_files = json.load(open(expected_file_path, 'r'))
            mapping[induce_hash] = expected_files
    return mapping


def compute_metrics(recommended, actual):
    recommended_set = set(recommended)
    actual_set = set(actual)
    true_positives = len(recommended_set & actual_set)
    precision = true_positives / len(recommended_set) if recommended_set else 0.0
    recall = true_positives / len(actual_set) if actual_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def compute_average_precision(recommended, actual):
    ap = 0.0
    hit_count = 0
    for idx, file in enumerate(recommended):
        if file in actual:
            hit_count += 1
            ap += hit_count / (idx + 1)
    ap = ap / len(actual) if actual else 0.0
    return ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find similar commits based on embeddings.')
    # parser.add_argument('--input-path', type=str, required=True,
    #                   help='Path of the evaluated recommendation')
    parser.add_argument('--recommendation-file-name', type=str, required=True,
                      help='File name of recommendation list')
    parser.add_argument('--output-path', type=str, required=True,
                      help='Path to evaluation result')
    parser.add_argument('--repo-name', type=str, required=True,
                      help='Name of the repository')
    parser.add_argument('--n', type=int, required=True,
                      help='Number of recommended files')

    args = parser.parse_args()
    input_patterns = [f'data/{args.repo_name}/sid.json', f'data/{args.repo_name}/mid_single.json']
    data = load_data_from_patterns(input_patterns)
    print(f"Loaded {len(data)} query data from input patterns.")

    # with open(args.input_path, 'r') as f:
    #     eval_data = json.load(f)

    commit_to_target = map_inducing_commits_to_target(data, args.repo_name)

    precisions, recalls, f1s = [], [], []
    aps = []
    for entry in data:
        query_commits = entry.get('induceCommitHashList')
        for query_commit in query_commits:
            recommended_files_info = json.load(open(f"data/{args.repo_name}/recommendation/{query_commit}_{args.recommendation_file_name}",'r'))
            recommended_files = [entry['file'] for entry in recommended_files_info]
            actual_files = commit_to_target[query_commit]
            precision, recall, f1 = compute_metrics(recommended_files[:args.n], actual_files)
            mean_ap = compute_average_precision(recommended_files, actual_files)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            aps.append(mean_ap)
        #print(f"QueryCommit: {query_commit}\n  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AP: {ap:.3f}")


    if precisions:
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
        map_score = sum(aps) / len(aps)
        print(f"\nAverage Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1: {avg_f1:.3f}")
        print(f"Mean Average Precision (MAP): {map_score:.3f}")

        eval_result = {
            "k": args.n,
            "Average Precision": round(avg_precision, 3),
            "Average Recall": round(avg_recall, 3),
            "Average F1": round(avg_f1, 3),
            "Mean Average Precision (MAP)": round(map_score, 3)
        }


        with open(args.output_path, 'w') as f:
            json.dump(eval_result, f, indent=4)
    else:
        print("No matching ground truth found for any queryCommit.")


