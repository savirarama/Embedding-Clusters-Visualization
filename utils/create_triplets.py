import json
import random
from typing import List, Tuple, Set
import pandas as pd
import glob
import os
import argparse
from sklearn.model_selection import train_test_split 
from get_modified_files import get_modified_files_from_matrix
from tqdm import tqdm

def load_data_from_patterns(patterns: List[str]) -> List[dict]:
    all_data = []
    for pattern in patterns:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            try:
                data = json.load(open(file_path, 'r'))
                all_data.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return all_data

def get_all_commits(data: List[dict]) -> set:
    all_commits = set()
    for entry in data:
        all_commits.update(entry['induceCommitHashList'])
        all_commits.update(entry['fixCommitHashList'])
    return all_commits

def create_triplets_from_subset(
    data: List[dict],
    allowed_commit_hashes: Set[str], # This parameter must be passed
    commit_hashes: List[str],
    repo: str,
    used_negatives: Set[str] = None 
) -> List[Tuple[str, str, str]]:
    triplets = []
    if used_negatives is None:
        used_negatives = set()  
    
    # Filter out used negatives from commit_hashes and shuffle for randomness
    available_negatives = [c for c in commit_hashes if c not in used_negatives]
    random.shuffle(available_negatives) 
    
    anchor_to_positives_map = {}
    anchor_to_target_files_map = {} 

    
    for entry in data:
        for induce_commit in entry['induceCommitHashList']:
            if induce_commit in allowed_commit_hashes:
                if induce_commit not in anchor_to_positives_map:
                    anchor_to_positives_map[induce_commit] = set()
                    anchor_to_target_files_map[induce_commit] = set()
                anchor_to_positives_map[induce_commit].update(
                    [fc for fc in entry['fixCommitHashList'] if fc in allowed_commit_hashes]
                )
            
                recommendedFilePaths = glob.glob(f'../GitCF/experiment_data/ishida/{repo}/*/{entry["fixIssueID"]}/_expected.json')
                if recommendedFilePaths:
                    for path in recommendedFilePaths:
                        if os.path.exists(path):
                            try:
                                recommendedFiles = json.load(open(path,'r'))
                                anchor_to_target_files_map[induce_commit].update(recommendedFiles)
                            except Exception as e:
                                print(f"Error loading recommended files for {entry['fixIssueID']}: {e}")
                else:
                    # If no specific recommended files are found, use empty set
                    anchor_to_target_files_map[induce_commit].update(set())

    # Iterate through the allowed anchors and generate triplets
    for anchor in tqdm(allowed_commit_hashes, desc="Generating triplets"):
        if anchor in anchor_to_positives_map: # Check if this anchor actually has positive pairs
            positive_candidates_for_anchor = anchor_to_positives_map[anchor]
            target_files = anchor_to_target_files_map.get(anchor, set()) 

            for positive in positive_candidates_for_anchor:
                # Get files modified by the positive commit
                positive_files = get_modified_files_from_matrix(positive, repo) or []
                
                # Try to find a valid negative candidate
                negative = None
                # Iterate over a copy of available_negatives, remove from original
                for neg_candidate in list(available_negatives): 
                    # Skip if the candidate is the anchor, positive, or any known positive
                    if neg_candidate in {anchor} | positive_candidates_for_anchor:
                        continue
                        
                    # Get files modified by the negative candidate
                    neg_files = get_modified_files_from_matrix(neg_candidate, repo) or []
                    
                    # Check if any of the negative files overlap with target files or positive files
                    if not any(f in target_files or f in positive_files for f in neg_files):
                        negative = neg_candidate
                        used_negatives.add(neg_candidate)  # Mark this commit as used
                        available_negatives.remove(neg_candidate)  # Remove from available candidates
                        break

                if negative:
                    triplets.append((anchor, positive, negative))

    return triplets

def save_triplets(triplets: List[Tuple[str, str, str]], output_file: str):
    """Save triplets to a JSON file in the format:
    [
      {"anchor": ..., "positive": ..., "negative": ...},
      ...
    ]
    """
    triplet_dicts = [
        {"anchor": anchor, "positive": positive, "negative": negative}
        for anchor, positive, negative in triplets
    ]
    with open(output_file, 'w') as f:
        json.dump(triplet_dicts, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Create triplets from JSON data files.")
    parser.add_argument('--test-size', type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--val-size', type=float, default=0.2,
                        help="Proportion of the dataset to include in the validation split.")
    parser.add_argument('--repo-name', type=str, required=True,
                        help="Repository name")
    parser.add_argument('--random-seed', type=int, default=42,
                    help="Random seed for reproducibility.")
    args = parser.parse_args()


    train_output_file_path = f"data/{args.repo_name}/triplets/train_triplets.json"
    val_output_file_path = f"data/{args.repo_name}/triplets/val_triplets.json"
    test_output_file_path = f"data/{args.repo_name}/triplets/test_triplets.json"

    # Load the commit hashes for negative sampling
    with open(f"data/{args.repo_name}/commit_hashes.json", 'r') as f:
        commit_hashes = json.load(f)
    
    input_patterns = [f'../GitCF/experiment_data/ishida/{args.repo_name}/sid.json', f'../GitCF/experiment_data/ishida/{args.repo_name}/mid_single.json']

    bic_bfc_data = load_data_from_patterns(input_patterns)
    print(f"Loaded {len(bic_bfc_data)} BIC-BFC entries from input patterns.")

    bic_bfc_unique_commits = get_all_commits(bic_bfc_data)
    print(f"Found {len(bic_bfc_unique_commits)} unique BIC-BFC commit hashes in total.")

    train_val_bic_bfc_data, test_bic_bfc_data = train_test_split(
        bic_bfc_data,
        test_size=args.test_size,
        random_state=args.random_seed
    )

    if args.test_size >= 1.0: 
        remaining_ratio = 0.0
    else:
        remaining_ratio = 1.0 - args.test_size

    if remaining_ratio > 0:
        val_relative_size = args.val_size / remaining_ratio
    else:
        val_relative_size = 0.0 

    train_bic_bfc_data, val_bic_bfc_data = train_test_split(
        train_val_bic_bfc_data,
        test_size=val_relative_size,
        random_state=args.random_seed
    )
    print(f"\nAfter second split:")
    print(f"  Training set size: {len(train_bic_bfc_data)}")
    print(f"  Validation set size: {len(val_bic_bfc_data)}")

    train_commit_hashes = list(set(get_all_commits(train_bic_bfc_data)))
    val_commit_hashes = list(set(get_all_commits(val_bic_bfc_data)))
    test_commit_hashes = list(set(get_all_commits(test_bic_bfc_data)))

    print(f"\nTraining set will use {len(train_commit_hashes)} unique commit hashes (from its pairs).")
    print(f"Validation set will use {len(val_commit_hashes)} unique commit hashes (from its pairs).")
    print(f"Test set will use {len(test_commit_hashes)} unique commit hashes (from its pairs).")

    print("\nGenerating training triplets...")
    train_triplets = create_triplets_from_subset(
        data=train_bic_bfc_data,                
        allowed_commit_hashes=train_commit_hashes,     
        commit_hashes=commit_hashes,          
        repo=args.repo_name,
        used_negatives=set()   
    )
    save_triplets(train_triplets, train_output_file_path)
    print(f"Created {len(train_triplets)} training triplets and saved to {train_output_file_path}")

    train_used_negatives = {t[2] for t in train_triplets}
    print(f"Number of unique negatives used in training: {len(train_used_negatives)}")

    val_triplets = create_triplets_from_subset(
        data=val_bic_bfc_data,                # Pass the validation data subset
        allowed_commit_hashes=val_commit_hashes,     # Negative candidates from entire pool
        commit_hashes=commit_hashes,            # Additional repository-wide negative candidates
        repo=args.repo_name,
        used_negatives=train_used_negatives     # Pass used negatives from training
    )

    save_triplets(val_triplets, val_output_file_path)
    print(f"Created {len(val_triplets)} test triplets and saved to {val_output_file_path}")

    val_used_negatives = train_used_negatives | {t[2] for t in val_triplets}
    print(f"Number of unique negatives used in training and validation: {len(val_used_negatives)}")
    # --- Generate Triplet Data for Test Set ---
    print("\nGenerating test triplets...")
    test_triplets = create_triplets_from_subset(
        data=test_bic_bfc_data,     
        allowed_commit_hashes=test_commit_hashes,        
        commit_hashes=commit_hashes,         
        repo=args.repo_name,
        used_negatives=val_used_negatives,   
    )
    save_triplets(test_triplets, test_output_file_path)
    print(f"Created {len(test_triplets)} test triplets and saved to {test_output_file_path}")


    train_used_negatives = {t[2] for t in train_triplets}
    test_used_negatives = {t[2] for t in test_triplets}
    print(f"\nDebug info:")
    print(f"Number of unique negatives in training: {len(train_used_negatives)}")
    print(f"Number of unique negatives in test: {len(test_used_negatives)}")
    overlap = train_used_negatives.intersection(test_used_negatives)
    print(f"\nOverlap check - Negatives used in both train and test: {len(overlap)} (should be 0)")
    print("Overlapping negatives:", overlap)

    print("\nSample Training Triplet:")
    if train_triplets:
        sample_idx = random.randrange(len(train_triplets))
        print(f"Anchor: {train_triplets[sample_idx][0]}")
        print(f"Positive: {train_triplets[sample_idx][1]}")
        print(f"Negative: {train_triplets[sample_idx][2]}")
    else:
        print("No training triplets generated.")

    print("\nSample Test Triplet:")
    if test_triplets:
        sample_idx = random.randrange(len(test_triplets))
        print(f"Anchor: {test_triplets[sample_idx][0]}")
        print(f"Positive: {test_triplets[sample_idx][1]}")
        print(f"Negative: {test_triplets[sample_idx][2]}")
    else:
        print("No test triplets generated.")


if __name__ == "__main__":
    main()