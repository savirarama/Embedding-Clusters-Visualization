import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
import argparse
from get_modified_files import get_modified_files_from_commit

def load_data(json_file_path, npy_file_path):
    """
    Loads commit data from a JSON file and embeddings from a .npy file.

    Args:
        json_file_path (str): Path to the JSON file containing commit hashes.
        npy_file_path (str): Path to the .npy file containing commit embeddings.
                             The order of embeddings must match the order of commits in the JSON.

    Returns:
        tuple: A list of commit hashes and a NumPy array with embeddings.
               Returns (None, None) if loading fails or files don't exist.
    """
    # Check if files exist
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return None, None
    if not os.path.exists(npy_file_path):
        print(f"Error: NPY file not found at {npy_file_path}")
        return None, None

    try:
        # Load commit hashes from JSON
        with open(json_file_path, 'r') as f:
            commit_hashes = json.load(f)

        # Load commit embeddings
        embeddings = np.load(npy_file_path)

        # Validate that the number of commits matches the number of embeddings
        if len(commit_hashes) != len(embeddings):
            print(f"Error: Number of commits in JSON ({len(commit_hashes)}) "
                  f"does not match number of embeddings in NPY file ({len(embeddings)}).")
            print("Please ensure the .npy file contains embeddings in the same order as the JSON.")
            return None, None
            
        return commit_hashes, embeddings
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None, None

def load_query_commits(json_path):
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            query_info_list = []
            for entry in data:
                query_hashes = entry.get("induceCommitHashList", [])
                target_hashes = entry.get("fixCommitHashList", [])
                for query_hash in query_hashes:
                    query_info_list.append({
                        "query": query_hash,
                        "targets": target_hashes
                    })
            return query_info_list
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return []

def find_similar_commits(commit_input_hash, n, commit_hashes, all_embeddings, target_commit_to_rank=None):
    """
    Finds the n most similar commits to a given commit hash, grouping commits
    with identical similarity scores.

    Args:
        commit_input_hash (str): The hash of the input commit.
        n (int): The number of top distinct similarity ranks to consider.
                 If n=1, it returns all commits with the highest similarity.
                 If n=2, it returns all commits from the top two distinct similarity ranks.
        commit_hashes (list): A list of all commit hashes, corresponding to the order of embeddings.
        all_embeddings (np.ndarray): NumPy array of all commit embeddings.
        target_commit_to_rank (str, optional): An optional commit hash to find its rank in the grouped list.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries containing commit hashes and their similarity scores
            - int or None: The rank of target_commit_to_rank if found (based on distinct ranks), otherwise None.
            - float or None: The similarity score of target_commit_to_rank if found, otherwise None.
    """
    # 1. Find the index and embedding of the input commit
    try:
        # Check if the input commit hash exists in the provided list
        if commit_input_hash not in commit_hashes:
            print(f"Error: Commit hash '{commit_input_hash}' not found in list of commit hashes.")
            return [], None, None # Return empty list and None for ranks on error

        # Get the positional index of the input commit hash
        positional_target_idx = commit_hashes.index(commit_input_hash)
        
        # Retrieve the embedding for the target commit using its index
        target_embedding = all_embeddings[positional_target_idx]

    except Exception as e:
        # Catch any exceptions during the retrieval of the input commit's embedding
        print(f"An error occurred while finding the input commit embedding: {e}")
        return [], None, None # Return empty list and None for ranks on error

    # 2. Calculate cosine similarity
    # Reshape target_embedding to be a 2D array (1, num_features) as required by cosine_similarity
    target_embedding_reshaped = target_embedding.reshape(1, -1)
    
    # Calculate similarities between the target embedding and all other embeddings
    # cosine_similarity returns a 2D array, which we then flatten to a 1D array of scores
    similarity_matrix = cosine_similarity(target_embedding_reshaped, all_embeddings)
    similarity_scores = similarity_matrix[0] # Extract the 1D array of similarity scores

    # 3. Store similarities with commit hashes, excluding the input commit itself
    commit_similarity_pairs = []
    for i, score in enumerate(similarity_scores):
        # The input commit will have a similarity score very close to 1.0 with itself,
        # so we exclude it from the list of similar commits.
        if i == positional_target_idx:
            continue
        
        # Get the commit hash using its positional index from the commit_hashes list
        commit_hash = commit_hashes[i]
        commit_similarity_pairs.append({'hash': commit_hash, 'similarity': float(score)})

    # 4. Sort by similarity in descending order
    # Python's sort is stable, meaning that if two items have the same similarity score,
    # their relative order in the original list is preserved.
    commit_similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    # 4.5 Group commits by identical similarity scores
    grouped_commit_similarity_pairs = []
    if commit_similarity_pairs: # Only process if there are pairs
        current_similarity = None
        current_group = []
        
        for pair in commit_similarity_pairs:
            if current_similarity is None or pair['similarity'] == current_similarity:
                # If first element or same similarity, add to current group
                current_similarity = pair['similarity']
                current_group.append(pair)
            else:
                # If new similarity, save previous group and start a new one
                grouped_commit_similarity_pairs.append(current_group)
                current_similarity = pair['similarity']
                current_group = [pair]
        
        # Add the last group after the loop finishes
        grouped_commit_similarity_pairs.append(current_group)

    # 5. Return top N commit hashes based on distinct ranks
    top_n_commits = []
    ranks_processed = 0
    for group in grouped_commit_similarity_pairs:
        ranks_processed += 1
        top_n_commits.extend(group)
        # Stop once we have accumulated 'n' distinct ranks
        if ranks_processed >= n:
            break

    # 6. Find rank of target_commit_to_rank (if specified)
    target_rank = None
    target_similarity_score = None
    if target_commit_to_rank is not None:
        # Iterate through the grouped pairs to find the rank of the specified target commit
        for rank, group in enumerate(grouped_commit_similarity_pairs, start=1):
            if any(pair['hash'] == target_commit_to_rank for pair in group):
                target_rank = rank
                target_similarity_score = next(pair['similarity'] for pair in group if pair['hash'] == target_commit_to_rank)
                break
    
    return top_n_commits, target_rank, target_similarity_score

# --- Main execution ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find similar commits based on embeddings.')
    parser.add_argument('--hashes', type=str,
                      help='Path to the JSON file containing commit hashes')
    parser.add_argument('--npy', type=str, default='embedding/compressed_embeddings.npy',
                      help='Path to the NPY file containing commit embeddings')
    parser.add_argument('--query', type=str, default='data/sid.json',
                      help='Path to the JSON file containing query commits')
    parser.add_argument('--output-recommendations', type=str, 
                      help='Path to save recommendation results')
    parser.add_argument('--output-ranks', type=str,
                      help='Path to save ranking results')
    parser.add_argument('--num-similar', type=int, default=10,
                      help='Number of top similar commits to retrieve')
    parser.add_argument('--repo-name', type=str, help='Name of the repository')
    parser.add_argument('--user', type=str, help='Name of the user')

    args = parser.parse_args()

    # --- Configuration ---
    hash_file_path = args.hashes
    npy_file_path = args.npy
    query_json_path = args.query
    save_results_path = args.output_recommendations
    save_ranks_path = args.output_ranks
    num_similar_commits = args.num_similar

    print("Loading commit data...")
    all_commit_hashes, all_commit_embeddings = load_data(hash_file_path, npy_file_path)

    print(f"Loading query commits from '{query_json_path}'...")
    query_entries = load_query_commits(query_json_path)

    if all_commit_hashes is not None and all_commit_embeddings is not None and query_entries:
        # Initialize results
        rank_results = []
        recommendation_results = []

        all_true_labels = []
        all_pred_labels = []
        all_correct_predictions = 0
        total_queries = len(query_entries)

        # Process each query entry
        for entry in query_entries:
            query_hash = entry["query"]
            target_hashes = entry["targets"]

            print(f"\nProcessing query commit: {query_hash}")
            
            # --- Top-N similar commits for recommendation output ---
            similar_commits_with_scores, _, _ = find_similar_commits(
                commit_input_hash=query_hash,
                n=num_similar_commits,
                commit_hashes=all_commit_hashes,
                all_embeddings=all_commit_embeddings
            )
            query_modified_files = get_modified_files_from_commit(query_hash,
                                                                  repo_path='.',
                                                                  remote_url=f"https://github.com/{args.user}/{args.repo_name}.git",
                                                                  )
            target_modified_files = []
            for entry in similar_commits_with_scores:
               entry_modified_files = get_modified_files_from_commit(entry['hash'],
                                                                  repo_path='.',
                                                                  remote_url=f"https://github.com/{args.user}/{args.repo_name}.git",
                                                                  )
               if entry_modified_files:  # Only append if we got valid files
                   target_modified_files.extend(entry_modified_files)
            

            target_modified_files = list(set(target_modified_files))
            query_modified_files = query_modified_files or []  # Handle None case
            recommended_files = list(set(target_modified_files) - set(query_modified_files))

            recommendation_results.append({
                "queryCommit": query_hash,
                "queryModifiedFiles": query_modified_files,
                "recommendedCommitSimilarityPairs": similar_commits_with_scores,
                "recommendedFiles": recommended_files
            })

            # --- Accuracy Calculation (run once per query) ---
            similar_commit_hashes = [pair['hash'] for pair in similar_commits_with_scores]
            if any(target_hash in similar_commit_hashes for target_hash in target_hashes):
                all_correct_predictions += 1

            # --- Target commit ranking output ---
            for target_hash in target_hashes:
                _, rank, target_similarity_score = find_similar_commits(
                    commit_input_hash=query_hash,
                    n=num_similar_commits,
                    commit_hashes=all_commit_hashes,
                    all_embeddings=all_commit_embeddings,
                    target_commit_to_rank=target_hash
                )

                rank_results.append({
                    "query_commit": query_hash,
                    "target_commit": target_hash,
                    "rank": rank,
                    "similarity_score": float(target_similarity_score) if target_similarity_score is not None else None
                })

                # Binary ground truth: which top-N recommendations are actually relevant
                true_labels = [1 if pair['hash'] in target_hashes else 0 for pair in similar_commits_with_scores]
                pred_labels = [1] * len(similar_commits_with_scores)  # predicted all as relevant (top-N)

                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)

                print(f"→ Target commit '{target_hash}' ranked #{rank} for query '{query_hash}'.")

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='micro'
        )
        accuracy = all_correct_predictions / total_queries

        # Print metrics
        print("\nMetrics for all queries:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Save both outputs
        os.makedirs("recommendation_results", exist_ok=True)

        with open(save_results_path, "w") as f:
            json.dump(recommendation_results, f, indent=4)
            print(f"\n✔️ Recommendation results written to {save_results_path}")

        with open(save_ranks_path, "w") as f:
            json.dump(rank_results, f, indent=4)
            print(f"✔️ Ranking results written to {save_ranks_path}")
