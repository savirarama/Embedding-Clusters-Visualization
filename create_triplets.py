import json
import random
from typing import List, Tuple, Set
import pandas as pd
import glob
import argparse
from sklearn.model_selection import train_test_split 
from get_modified_files import get_modified_files_from_commit
from tqdm import tqdm

# --- Your existing helper functions (no changes needed) ---
def load_data_from_file(file_path: str) -> List[dict]:
    """Load data from a single JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_data_from_patterns(patterns: List[str]) -> List[dict]:
    """Load data from multiple JSON files matching the given patterns."""
    all_data = []
    for pattern in patterns:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            try:
                data = load_data_from_file(file_path)
                all_data.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return all_data

def get_all_commits(data: List[dict]) -> set:
    """Get all unique commit hashes from the dataset."""
    all_commits = set()
    for entry in data:
        all_commits.update(entry['induceCommitHashList'])
        all_commits.update(entry['fixCommitHashList'])
    return all_commits

# --- Modified create_triplets to take a subset of commits and data ---
def create_triplets_from_subset(
    data: List[dict],
    allowed_commit_hashes: Set[str],
    all_possible_negatives: Set[str], # All unique commits from the full dataset for global negatives
    commit_hashes: List[str], # New parameter for additional negative candidates
    repo_path: str = '.', # New parameter for get_modified_files_from_commit
    remote_url: str = None, # New parameter for get_modified_files_from_commit
    used_negatives: Set[str] = None # New parameter to track used negatives
) -> List[Tuple[str, str, str]]:
    """
    Create triplet pairs (anchor, positive, negative) from a subset of data,
    ensuring all hashes used are within the allowed_commit_hashes.
    Negative candidates can be from commit_hashes list, but must not overlap with positive/anchor
    and must modify different files than the target files.
    Each negative commit can only be used once across all triplets.
    """
    triplets = []
    if used_negatives is None:
        used_negatives = set()  # Track which commits have been used as negatives
    
    # Filter out used negatives from commit_hashes
    available_negatives = [c for c in commit_hashes if c not in used_negatives]
    print(f"Available negative candidates after filtering: {len(available_negatives)}")
    
    # First, build a map from induceCommitHashList to fixCommitHashList within the allowed set
    anchor_to_positives_map = {}
    anchor_to_target_files_map = {}  # New map to store target files for each anchor
    
    for entry in data:
        for induce_commit in entry['induceCommitHashList']:
            if induce_commit in allowed_commit_hashes:
                if induce_commit not in anchor_to_positives_map:
                    anchor_to_positives_map[induce_commit] = set()
                    anchor_to_target_files_map[induce_commit] = set()
                # Ensure positive commits are also within the allowed set for *this split*
                anchor_to_positives_map[induce_commit].update(
                    [fc for fc in entry['fixCommitHashList'] if fc in allowed_commit_hashes]
                )
                # Store the recommended files as target files
                if 'recommendedFiles' in entry:
                    anchor_to_target_files_map[induce_commit].update(entry['recommendedFiles'])

    # Now, iterate through the allowed anchors and generate triplets
    for anchor in tqdm(allowed_commit_hashes, desc="Generating triplets"):
        if anchor in anchor_to_positives_map: # Check if this anchor actually has positive pairs
            positive_candidates_for_anchor = anchor_to_positives_map[anchor]
            target_files = anchor_to_target_files_map[anchor]

            for positive in positive_candidates_for_anchor:
                # Get files modified by the positive commit
                positive_files = get_modified_files_from_commit(positive, repo_path, remote_url) or []
                
                # Try to find a valid negative candidate
                negative = None
                for neg_candidate in available_negatives:
                    # Skip if the candidate is the anchor, positive, or any known positive
                    if neg_candidate in {anchor} | positive_candidates_for_anchor:
                        continue
                        
                    # Get files modified by the negative candidate
                    neg_files = get_modified_files_from_commit(neg_candidate, repo_path, remote_url) or []
                    
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

### Main Execution Logic

def main():
    parser = argparse.ArgumentParser(description="Create triplets from JSON data files.")
    parser.add_argument('--input-patterns', nargs='+', default=['data/bic_bfc_pairs/*/sid.json',
                                                            'data/bic_bfc_pairs/*/mid_single.json'],
                        help="Patterns to match input JSON files.")
    parser.add_argument('--train-output-file', default='triplets_train.json',
                        help="Output file for saving training triplets.")
    parser.add_argument('--test-output-file', default='triplets_test.json',
                        help="Output file for saving test triplets.")
    parser.add_argument('--test-size', type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--commit-hashes-file', type=str, required=True,
                        help="JSON file containing list of commit hashes for negative sampling.")
    parser.add_argument('--repo-path', type=str, default='.',
                        help="Path to the git repository.")
    parser.add_argument('--remote-url', type=str,
                        help="URL of the git repository remote.")
    args = parser.parse_args()

    input_patterns = args.input_patterns
    train_output_file_path = args.train_output_file
    test_output_file_path = args.test_output_file
    test_size = args.test_size
    repo_path = args.repo_path
    remote_url = args.remote_url

    # Load the commit hashes for negative sampling
    with open(args.commit_hashes_file, 'r') as f:
        commit_hashes = json.load(f)

    # Load the bic_bfc data
    bic_bfc_data = load_data_from_patterns(input_patterns)
    print(f"Loaded {len(bic_bfc_data)} BIC-BFC entries from input patterns.")

    # Get all unique commit hashes from the ENTIRE dataset
    bic_bfc_unique_commits = get_all_commits(bic_bfc_data)
    print(f"Found {len(bic_bfc_unique_commits)} unique BIC-BFC commit hashes in total.")

    # --- Crucial Step: Split the UNIQUE COMMIT HASHS into train and test sets ---
    # Convert set to list for train_test_split
    all_unique_commits_list = list(bic_bfc_unique_commits)
    train_commit_hashes, test_commit_hashes = train_test_split(
        all_unique_commits_list,
        test_size=test_size,
        train_size=1-test_size
    )

    # Convert back to sets for faster lookups in create_triplets_from_subset
    train_commit_hashes = set(train_commit_hashes)
    test_commit_hashes = set(test_commit_hashes)

    print(f"Training set will use {len(train_commit_hashes)} unique commit hashes.")
    print(f"Test set will use {len(test_commit_hashes)} unique commit hashes.")
    print(f"Overlap check: {len(train_commit_hashes.intersection(test_commit_hashes))} common hashes (should be 0).")

    # --- Generate Triplet Data for Training Set ---
    print("\nGenerating training triplets...")
    train_triplets = create_triplets_from_subset(
        bic_bfc_data,              # Pass the full data to allow searching for pairs
        train_commit_hashes,    # Only allow anchors/positives from this set
        bic_bfc_unique_commits,     # Negative candidates can be from the entire pool
        commit_hashes,          # Additional negative candidates
        repo_path,              # Path to git repository
        remote_url,             # Remote URL for git repository
        used_negatives=set()    # Start with empty set of used negatives
    )
    save_triplets(train_triplets, train_output_file_path)
    print(f"Created {len(train_triplets)} training triplets and saved to {train_output_file_path}")

    # Get the set of negatives used in training
    train_used_negatives = {t[2] for t in train_triplets}
    print(f"Number of unique negatives used in training: {len(train_used_negatives)}")

    # --- Generate Triplet Data for Test Set ---
    print("\nGenerating test triplets...")
    test_triplets = create_triplets_from_subset(
        bic_bfc_data,             # Pass the full data
        test_commit_hashes,    # Only allow anchors/positives from this set
        bic_bfc_unique_commits,    # Negative candidates can be from the entire pool
        commit_hashes,         # Additional negative candidates
        repo_path,             # Path to git repository
        remote_url,            # Remote URL for git repository
        used_negatives=train_used_negatives   # Pass the set of negatives used in training
    )
    save_triplets(test_triplets, test_output_file_path)
    print(f"Created {len(test_triplets)} test triplets and saved to {test_output_file_path}")

    # Verify no overlap in negatives between train and test
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