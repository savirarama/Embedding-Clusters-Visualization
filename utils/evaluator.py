import json
from get_modified_files import get_modified_files_from_commit
from tqdm import tqdm

# Path to the sid.json file
SID_JSON_PATH = 'data/bic_bfc_pairs/hive/sid.json'
OUTPUT_JSON_PATH = 'eval_results/hive/hive_compressed_eval_results.json'
REPO_PATH = '.'  # Change if your repo is elsewhere
REMOTE_URL = 'https://github.com/apache/hive.git'  # Set if you want to fetch from a remote
REMOTE_NAME = 'external_source'
EVAL_JSON_PATH = 'recommendation_results/hive_compressed_file_recommendation_sid.json'  # Set the correct path

def get_ground_truth(data):

    for entry in tqdm(data, desc="Obtaining actual target files"):
        # Get queryFiles from induceCommitHashList
        query_files = set()
        for commit_hash in entry.get('induceCommitHashList', []):
            files = get_modified_files_from_commit(commit_hash, repo_path=REPO_PATH, remote_url=REMOTE_URL, remote_name=REMOTE_NAME)
            if files:
                query_files.update(files)
        entry['queryFiles'] = sorted(list(query_files))

        # Get targetFiles from fixCommitHashList, excluding queryFiles
        target_files = set()
        for commit_hash in entry.get('fixCommitHashList', []):
            files = get_modified_files_from_commit(commit_hash, repo_path=REPO_PATH, remote_url=REMOTE_URL, remote_name=REMOTE_NAME)
            if files:
                target_files.update(files)
        # Remove files already in queryFiles
        target_files = target_files - query_files
        entry['targetFiles'] = sorted(list(target_files))
    return data

def compute_metrics(recommended, actual):
    recommended_set = set(recommended)
    actual_set = set(actual)
    true_positives = len(recommended_set & actual_set)
    precision = true_positives / len(recommended_set) if recommended_set else 0.0
    recall = true_positives / len(actual_set) if actual_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Average Precision (AP) calculation
    ap = 0.0
    hit_count = 0
    for idx, file in enumerate(recommended):
        if file in actual_set:
            hit_count += 1
            ap += hit_count / (idx + 1)
    ap = ap / len(actual_set) if actual_set else 0.0
    return precision, recall, f1, ap

if __name__ == '__main__':
    with open(SID_JSON_PATH, 'r') as f:
        data = json.load(f)

    with open(EVAL_JSON_PATH, 'r') as f:
        eval_data = json.load(f)

    ground_truth = get_ground_truth(data)
    # Build a mapping from induceCommitHashList (as tuple) to targetFiles
    commit_to_target = {}
    for entry in tqdm(ground_truth, desc='Build a mapping from induceCommitHashList to targetFiles'):
        # Use tuple of induceCommitHashList as key
        key = tuple(entry.get('induceCommitHashList', []))
        commit_to_target[key] = set(entry.get('targetFiles', []))

    precisions, recalls, f1s = [], [], []
    aps = []
    for entry in eval_data:
        query_commit = entry.get('queryCommit')
        recommended_files = [f['file_name'] for f in entry.get('recommendedFiles', []) if 'file_name' in f]
        # Find the ground truth entry with matching induceCommitHashList
        found = False
        for induce_commits, target_files in commit_to_target.items():
            if query_commit in induce_commits:
                actual_files = target_files
                found = True
                break
        if not found:
            # No ground truth for this query_commit
            continue
        precision, recall, f1, ap = compute_metrics(recommended_files, actual_files)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aps.append(ap)
        print(f"QueryCommit: {query_commit}\n  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AP: {ap:.3f}")

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(ground_truth, f, indent=4)

    if precisions:
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
        map_score = sum(aps) / len(aps)
        print(f"\nAverage Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1: {avg_f1:.3f}")
        print(f"Mean Average Precision (MAP): {map_score:.3f}")
    else:
        print("No matching ground truth found for any queryCommit.")


