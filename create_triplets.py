import json
import random
from typing import List, Tuple, Set
import pandas as pd
import glob
import argparse
from sklearn.model_selection import train_test_split # New import for splitting

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
    all_possible_negatives: Set[str] # All unique commits from the full dataset for global negatives
) -> List[Tuple[str, str, str]]:
    """
    Create triplet pairs (anchor, positive, negative) from a subset of data,
    ensuring all hashes used are within the allowed_commit_hashes.
    Negative candidates can be global, but must not overlap with positive/anchor.
    """
    triplets = []

    # Filter data to only include entries where anchor or positive commits are in the allowed set
    # This might need refinement based on how your 'entry' maps to 'commits'
    # A safer way might be to iterate through allowed_commit_hashes and then find relevant entries/positives/negatives
    
    # Let's refine this to explicitly use allowed_commit_hashes for anchors and positives
    # and all_possible_negatives for negatives
    
    # First, build a map from induceCommitHashList to fixCommitHashList within the allowed set
    anchor_to_positives_map = {}
    for entry in data:
        for induce_commit in entry['induceCommitHashList']:
            if induce_commit in allowed_commit_hashes:
                if induce_commit not in anchor_to_positives_map:
                    anchor_to_positives_map[induce_commit] = set()
                # Ensure positive commits are also within the allowed set for *this split*
                anchor_to_positives_map[induce_commit].update(
                    [fc for fc in entry['fixCommitHashList'] if fc in allowed_commit_hashes]
                )

    # Now, iterate through the allowed anchors and generate triplets
    for anchor in allowed_commit_hashes:
        if anchor in anchor_to_positives_map: # Check if this anchor actually has positive pairs
            positive_candidates_for_anchor = anchor_to_positives_map[anchor]

            for positive in positive_candidates_for_anchor:
                # Negative candidates must be from the *entire* pool of commits (all_possible_negatives)
                # but must NOT be the current anchor, positive, or any known positive for this anchor
                negative_candidates = all_possible_negatives - {anchor} - positive_candidates_for_anchor

                if negative_candidates:
                    # Randomly select a negative commit
                    negative = random.choice(list(negative_candidates))
                    triplets.append((anchor, positive, negative))
                # else:
                #     # Optionally, handle cases where no valid negative can be found.
                #     # This might indicate very small dataset or aggressive splitting.
                #     print(f"Warning: No suitable negative found for anchor {anchor}, positive {positive}")

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
    parser.add_argument('--random-seed', type=int, default=42,
                        help="Random seed for reproducibility of data splitting.")
    args = parser.parse_args()

    input_patterns = args.input_patterns
    train_output_file_path = args.train_output_file
    test_output_file_path = args.test_output_file
    test_size = args.test_size
    random_seed = args.random_seed

    # Load the entire dataset
    full_data = load_data_from_patterns(input_patterns)
    print(f"Loaded {len(full_data)} entries from input patterns.")

    # Get all unique commit hashes from the ENTIRE dataset
    all_unique_commits = get_all_commits(full_data)
    print(f"Found {len(all_unique_commits)} unique commit hashes in total.")

    # --- Crucial Step: Split the UNIQUE COMMIT HASHS into train and test sets ---
    # Convert set to list for train_test_split
    all_unique_commits_list = list(all_unique_commits)
    train_commit_hashes, test_commit_hashes = train_test_split(
        all_unique_commits_list,
        test_size=test_size,
        train_size=1-test_size,
        random_state=random_seed
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
        full_data,              # Pass the full data to allow searching for pairs
        train_commit_hashes,    # Only allow anchors/positives from this set
        all_unique_commits      # Negative candidates can be from the entire pool
    )
    save_triplets(train_triplets, train_output_file_path)
    print(f"Created {len(train_triplets)} training triplets and saved to {train_output_file_path}")


    # --- Generate Triplet Data for Test Set ---
    print("\nGenerating test triplets...")
    test_triplets = create_triplets_from_subset(
        full_data,             # Pass the full data
        test_commit_hashes,    # Only allow anchors/positives from this set
        all_unique_commits     # Negative candidates can be from the entire pool
    )
    save_triplets(test_triplets, test_output_file_path)
    print(f"Created {len(test_triplets)} test triplets and saved to {test_output_file_path}")

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