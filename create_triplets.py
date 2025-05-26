import json
import random
from typing import List, Tuple
import pandas as pd
import glob

import argparse

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

def create_triplets(data: List[dict]) -> List[Tuple[str, str, str]]:
    """Create triplet pairs (anchor, positive, negative) for training."""
    all_commits = get_all_commits(data)
    triplets = []
    
    for entry in data:
        # Get anchor commits (induce commits)
        anchor_commits = entry['induceCommitHashList']
        # Get positive commits (fix commits)
        positive_commits = entry['fixCommitHashList']
        
        # For each anchor commit
        for anchor in anchor_commits:
            # For each positive commit
            for positive in positive_commits:
                # Find negative commits (commits from other entries)
                negative_candidates = all_commits - set(positive_commits) - set(anchor_commits)
                if negative_candidates:
                    # Randomly select a negative commit
                    negative = random.choice(list(negative_candidates))
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
    parser.add_argument('--input-patterns', nargs='+', default=['data/bic_bfc_pairs/*/sid.json',
                                                            'data/bic_bfc_pairs/*/mid.json',
                                                            'data/bic_bfc_pairs/*/mid_single.json'],
                        help="Patterns to match input JSON files.")
    parser.add_argument('--output-file', help="Output file for saving triplets.")
    args = parser.parse_args()  

    output_file_path = args.output_file

    # Load the data from multiple files
    data = load_data_from_patterns(['data/bic_bfc_pairs/*/sid.json',
                                    'data/bic_bfc_pairs/*/mid.json',
                                    'data/bic_bfc_pairs/*/mid_single.json'])
    
    # Create triplets
    triplets = create_triplets(data)
    
    # Save triplets
    save_triplets(triplets, output_file_path)
    
    print(f"Created {len(triplets)} triplets")
    print("Sample triplets:")
    for i in range(min(5, len(triplets))):
        print(f"Anchor: {triplets[i][0]}")
        print(f"Positive: {triplets[i][1]}")
        print(f"Negative: {triplets[i][2]}")
        print("---")

if __name__ == "__main__":
    main() 