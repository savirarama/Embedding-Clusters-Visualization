import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from collections import defaultdict
import argparse
import logging
from tqdm import tqdm
from get_modified_files import get_modified_files_from_matrix
from datetime import datetime
from typing import List
import glob



# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/performance_metrics_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will still print to console
        ]
    )
    return log_filename

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

def load_query_commits(query_data):
    query_info_list = []
    for entry in query_data:
        query_hashes = entry.get("induceCommitHashList", [])
        target_hashes = entry.get("fixCommitHashList", [])
        for query_hash in query_hashes:
            query_info_list.append({
                "query": query_hash,
                "targets": target_hashes
                    })
    return query_info_list

def find_similar_commits(commit_input_hash, n, commit_hashes, all_embeddings, similarity_threshold=0.05):
    """
    Finds the n most similar commits to a given commit hash, grouping commits
    with similarity scores within a specified threshold.
    """
    # 1. Find the index and embedding of the input commit
    try:
        # Check if the input commit hash exists in the provided list
        if commit_input_hash not in commit_hashes:
            print(f"Error: Commit hash '{commit_input_hash}' not found in list of commit hashes.")
            return [] # Return empty list on error

        # Get the positional index of the input commit hash
        positional_target_idx = commit_hashes.index(commit_input_hash)
        
        # Retrieve the embedding for the target commit using its index
        target_embedding = all_embeddings[positional_target_idx]

    except Exception as e:
        # Catch any exceptions during the retrieval of the input commit's embedding
        print(f"An error occurred while finding the input commit embedding: {e}")
        return [] # Return empty list on error

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

    # 4.5 Group commits by similarity within the threshold
    grouped_commit_similarity_pairs = []
    if commit_similarity_pairs: # Only process if there are pairs
        current_group_start_similarity = None
        current_group = []
        
        for pair in commit_similarity_pairs:
            if current_group_start_similarity is None:
                # First commit in the sorted list starts the first group
                current_group_start_similarity = pair['similarity']
                current_group.append(pair)
            elif (abs(current_group_start_similarity - pair['similarity']) )<= (similarity_threshold):
                # If the current commit's similarity is within the threshold of the group's starting similarity, add it to the group
                current_group.append(pair)
                #print("similar pair")
            else:
                # If new commit is outside the threshold, save previous group and start a new one
                grouped_commit_similarity_pairs.append(current_group)
                current_group_start_similarity = pair['similarity']
                current_group = [pair]
                #print("non-similar pair")
        
        # Add the last group after the loop finishes
        if current_group:
            grouped_commit_similarity_pairs.append(current_group)
        
        print(f"Num of similar commits group: {len(grouped_commit_similarity_pairs)}")

    # 5. Return top N commit hashes based on distinct ranks (groups)
    top_n_commits = []
    ranks_processed = 0
    for group in grouped_commit_similarity_pairs:
        ranks_processed += 1
        top_n_commits.extend(group)
        # Stop once we have accumulated 'n' distinct ranks
        if ranks_processed >= n:
            break

    return top_n_commits, grouped_commit_similarity_pairs[:n]

def get_recommended_files(query_hash, similar_commits, matrix, commit_hashes, n=10, w=1.0):
    # Get files modified in query commit
    query_modified_files = get_modified_files_from_matrix(query_hash, matrix)
    query_modified_files = set(query_modified_files)
    
    # Calculate total similarity for normalization
    similarity_sum = sum(commit['similarity'] for commit in similar_commits)
    
    # Dictionary to store accumulated scores for each file
    file_scores = defaultdict(float)
    
    # Process each similar commit
    for commit in tqdm(similar_commits, desc="Processing similar commits to get files"):
        commit_hash = commit['hash']
        commit_similarity = commit['similarity']
        
        # Get files modified in this commit
        modified_files = get_modified_files_from_matrix(commit_hash, matrix)

        
        # Score each file in this commit
        for file_path in modified_files:
            # Skip files that were modified in the query commit
            if file_path in query_modified_files:
                continue
                
            # Calculate file score: w * commit_similarity / similarity_sum
            file_score = w * commit_similarity / similarity_sum
            
            # Accumulate score for this file
            file_scores[file_path] += file_score
    
    # Convert to list of dicts and sort by score
    scored_files = [{'file': file_path, 'score': score} 
                   for file_path, score in file_scores.items()]
    scored_files.sort(key=lambda x: x['score'], reverse=True)

    top_files = scored_files[:n]
    
    for i, entry in enumerate(top_files):
        entry['rank'] = i + 1

    
    # Return top n files
    return top_files


# --- Main execution ---
if __name__ == "__main__":
    # Set up logging first
    log_file = setup_logging()
    logging.info("Starting commit similarity analysis")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find similar commits based on embeddings.')
    parser.add_argument('--npy-file', type=str, required=True,
                      help='Path to the .npy file containing commit embeddings')
    parser.add_argument('--output-recommendations', type=str, required=True,
                      help='Path to save recommendation results')
    parser.add_argument('--output-commit-recommendations', type=str, required=True,
                      help='Path to save commit recommendation results')
    parser.add_argument('--n-commits', type=int, default=80,
                      help='Number of top similar commits to retrieve')
    parser.add_argument('--n-files', type=int, default=10,
                      help='Number of top files to retrieve')
    parser.add_argument('--repo-name', type=str, required=True,
                      help='Name of the repository')

    args = parser.parse_args()


    # --- Configuration ---
    save_results_path = args.output_recommendations
    save_commit_results_path = args.output_commit_recommendations
    num_similar_commits = args.n_commits
    num_similar_files = args.n_files

    # Log configuration
    logging.info(f"Repository: {args.repo_name}")
    logging.info(f"Num of similar commits: {num_similar_commits}")
    logging.info(f"Num of similar files: {num_similar_files}")

    print("Loading commit data...")
    with open(f"data/{args.repo_name}/commit_hashes.json", 'r') as f:
        commit_hashes = json.load(f)

    print("Loading matrix...")
    with open(f"data/{args.repo_name}/base.json", 'r') as f:
        matrix = json.load(f)

    print("Loading embeddings...")
    commit_embeddings = np.load(args.npy_file)

    input_patterns = [f'data/{args.repo_name}/sid.json', f'data/{args.repo_name}/mid_single.json']

    query_data = load_data_from_patterns(input_patterns)

    print(f"Loaded {len(query_data)} query data from input patterns.")


    print(f"Loading query commits from...")
    query_entries = load_query_commits(query_data)
    print(f"{len(query_entries)} query commits loaded.")

    if commit_hashes is not None and commit_embeddings is not None and query_entries:
        # Initialize results
        recommendation_results = []
        grouped_commits_recommendation = []

        # Process each query entry
        for entry in query_entries:
            query_hash = entry["query"]
            target_hashes = entry["targets"]

            print(f"\nProcessing query commit: {query_hash}")
            
            # --- Top-N similar commits for recommendation output ---
            similar_commits_with_scores, grouped_commits = find_similar_commits(
                commit_input_hash=query_hash,
                n=num_similar_commits,
                commit_hashes=commit_hashes,
                all_embeddings=commit_embeddings,
                similarity_threshold=0.001
            )
            #print(f"First similar commit group: {similar_commits_with_scores[0:5]}")
            print(f"Num of similar commits group: {len(similar_commits_with_scores)}")
            
            # Get recommended files based on similar commits
            recommended_files = get_recommended_files(
                query_hash=query_hash,
                similar_commits=similar_commits_with_scores,
                commit_hashes=commit_hashes,
                matrix=matrix,
                n=num_similar_files
            )
            
            # Store results
            recommendation_results.append({
                "queryCommit": query_hash,
                "recommendedFiles": recommended_files
            })
            grouped_commits_recommendation.append({
                "queryCommit": query_hash,
                "recommendedCommits": grouped_commits
            })
            # Log the recommendations
            logging.info(f"\nRecommendations for commit {query_hash}:")
            # logging.info("Similar commits:")
            # for commit in similar_commits_with_scores:
            #     logging.info(f"Commit: {commit['hash']}, Similarity: {commit['similarity']:.4f}")
            logging.info("\nRecommended files:")
            for file_info in recommended_files:
                logging.info(f"File: {file_info['file']}, Score: {file_info['score']:.4f}")
            
        # Save file recommendation results
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        with open(save_results_path, "w") as f:
            json.dump(recommendation_results, f, indent=4)
            logging.info(f"\n✔️ File recommendation results written to {save_results_path}")

        # Save commit recommendation results
        os.makedirs(os.path.dirname(save_commit_results_path), exist_ok=True)
        with open(save_commit_results_path, "w") as f:
            json.dump(grouped_commits_recommendation, f, indent=4)
            logging.info(f"\n✔️ Commit recommendation results written to {save_commit_results_path}")
            

    else:
        print("No valid data found for processing.")

    print("Analysis completed.")
    