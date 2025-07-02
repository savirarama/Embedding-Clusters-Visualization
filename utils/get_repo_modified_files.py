import argparse
from get_modified_files import get_modified_files_from_commit
import json
from tqdm import tqdm

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Create triplets from JSON data files.")
    parser.add_argument("--repo-name", type=str, required=True, help="Repository name")
    parser.add_argument("--user", type=str, default="apache")

    args = parser.parse_args()
    commit_hashes_path = f"data/{args.repo_name}/commit_hashes.json"

    total_entries = []


    with open(commit_hashes_path, 'r') as f:
        commit_hashes = json.load(f)
    

    for commit in tqdm(commit_hashes, desc="Getting modified files for each commit"):
        modified_files = get_modified_files_from_commit(
            commit,
            remote_url=f"https://github.com/{args.user}/{args.repo_name}.git"
        )
        total_entries.append(modified_files)  

    with open(f"data/{args.repo_name}/modified_files.json", 'w') as f:
        json.dump(total_entries, f, indent=2)
