import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
import argparse
from get_modified_files import get_modified_files_from_commit
import logging
from datetime import datetime

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
        # Try different encodings for JSON file
        encodings = ['utf-8', 'latin1', 'cp1252']
        commit_hashes = None
        
        for encoding in encodings:
            try:
                with open(json_file_path, 'r', encoding=encoding) as f:
                    commit_hashes = json.load(f)
                break
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON with {encoding} encoding: {e}")
                continue
        
        if commit_hashes is None:
            print(f"Error: Could not decode JSON file with any of the attempted encodings: {encodings}")
            return None, None

        # Load commit embeddings
        try:
            embeddings = np.load(npy_file_path)
        except Exception as e:
            print(f"Error loading NPY file: {e}")
            return None, None

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
    """
    Loads query commits from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing query commits.
        
    Returns:
        list: List of query commit information dictionaries.
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return []

    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
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
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON with {encoding} encoding: {e}")
            continue
        except Exception as e:
            print(f"An error occurred while reading the JSON file: {e}")
            continue
    
    print(f"Error: Could not decode JSON file with any of the attempted encodings: {encodings}")
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
        list: A list of dictionaries containing commit hashes and their similarity scores
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

    return top_n_commits

def get_recommended_files(query_hash, similar_commits, repo_path, remote_url, n=10, w=1.0):
    """
    Recommends files based on similar commits, scoring them according to commit similarities.
    
    Args:
        query_hash (str): The hash of the query commit
        similar_commits (list): List of dicts containing commit hashes and their similarity scores
        repo_path (str): Path to the repository
        remote_url (str): URL of the remote repository
        n (int): Number of top files to recommend
        w (float): Weight factor for scoring (default: 1.0)
        
    Returns:
        list: List of dicts containing file paths and their accumulated scores
    """
    # Get files modified in query commit
    query_modified_files = get_modified_files_from_commit(query_hash, repo_path, remote_url) or []
    query_modified_files = set(query_modified_files)
    
    # Calculate total similarity for normalization
    similarity_sum = sum(commit['similarity'] for commit in similar_commits)
    
    # Dictionary to store accumulated scores for each file
    file_scores = defaultdict(float)
    
    # Process each similar commit
    for commit in similar_commits:
        commit_hash = commit['hash']
        commit_similarity = commit['similarity']
        
        # Get files modified in this commit
        modified_files = get_modified_files_from_commit(commit_hash, repo_path, remote_url) or []
        
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
    
    # Return top n files
    return scored_files[:n]

def evaluate_recommendations(recommendation_results, ground_truth_list):
    """
    Evaluates the performance of file recommendations using precision, recall, F1, and Hit@k metrics.
    
    Args:
        recommendation_results (list): List of recommendation results, each containing:
            - queryCommit: The query commit hash
            - recommendedFiles: List of recommended files with their scores
        ground_truth_list (list): List of ground truth entries, each containing:
            - induceCommitHashList: List of inducing commit hashes
            - recommendedFiles: List of ground truth recommended files
            
    Returns:
        dict: Dictionary containing evaluation metrics:
            - precision: Average precision across all queries
            - recall: Average recall across all queries
            - f1: Average F1 score across all queries
            - hit_at_k: Hit@k score where k is the number of recommended files
    """
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_hit_at_k = 0
    total_queries = 0
    
    # Create a mapping of query commits to ground truth files
    ground_truth_map = {}
    for entry in ground_truth_list:
        for query_hash in entry.get("induceCommitHashList", []):
            ground_truth_map[query_hash] = set(entry.get("recommendedFiles", []))
    
    for result in recommendation_results:
        query_hash = result["queryCommit"]
        recommended_files = [file_info["file"] for file_info in result["recommendedFiles"]]
        
        # Skip if no ground truth for this query
        if query_hash not in ground_truth_map:
            continue
            
        ground_truth_files = ground_truth_map[query_hash]
        
        # Calculate precision, recall, and F1
        if recommended_files:
            true_positives = len(set(recommended_files) & ground_truth_files)
            precision = true_positives / len(recommended_files)
            recall = true_positives / len(ground_truth_files) if ground_truth_files else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate Hit@k where k is the number of recommended files
            if set(recommended_files) & ground_truth_files:
                total_hit_at_k += 1
        else:
            precision = recall = f1 = 0
            
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_queries += 1
    
    # Calculate averages
    if total_queries > 0:
        avg_precision = total_precision / total_queries
        avg_recall = total_recall / total_queries
        avg_f1 = total_f1 / total_queries
        hit_at_k = total_hit_at_k / total_queries
    else:
        avg_precision = avg_recall = avg_f1 = hit_at_k = 0
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "hit_at_k": hit_at_k
    }

# --- Main execution ---
if __name__ == "__main__":
    # Set up logging first
    log_file = setup_logging()
    logging.info("Starting commit similarity analysis")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find similar commits based on embeddings.')
    parser.add_argument('--hashes', type=str, required=True,
                      help='Path to the JSON file containing commit hashes')
    parser.add_argument('--npy', type=str, default='embedding/compressed_embeddings.npy',
                      help='Path to the NPY file containing commit embeddings')
    parser.add_argument('--query', type=str, default='data/sid.json',
                      help='Path to the JSON file containing query commits')
    parser.add_argument('--output-recommendations', type=str, required=True,
                      help='Path to save recommendation results')
    parser.add_argument('--eval-results-path', type=str, required=True,
                      help='Path to save evaluation results')
    parser.add_argument('--n-commits', type=int, default=10,
                      help='Number of top similar commits to retrieve')
    parser.add_argument('--n-files', type=int, default=10,
                      help='Number of top files to retrieve')
    parser.add_argument('--repo-name', type=str, required=True,
                      help='Name of the repository')
    parser.add_argument('--user', type=str, required=True,
                      help='Name of the user')
    parser.add_argument('--ground-truth', type=str, required=True,
                      help='Path to the ground truth file')

    args = parser.parse_args()

    # Validate required arguments
    if not args.user or not args.repo_name:
        parser.error("Both --user and --repo-name are required arguments")

    # --- Configuration ---
    hash_file_path = args.hashes
    npy_file_path = args.npy
    query_json_path = args.query
    save_results_path = args.output_recommendations
    eval_results_path = args.eval_results_path
    num_similar_commits = args.n_commits
    num_similar_files = args.n_files

    # Log configuration
    logging.info(f"Query file: {query_json_path}")
    logging.info(f"Embedding file: {npy_file_path}")
    logging.info(f"Repository: {args.user}/{args.repo_name}")
    logging.info(f"Num of similar commits: {num_similar_commits}")
    logging.info(f"Num of similar files: {num_similar_files}")

    print("Loading commit data...")
    all_commit_hashes, all_commit_embeddings = load_data(hash_file_path, npy_file_path)

    print(f"Loading query commits from '{query_json_path}'...")
    query_entries = load_query_commits(query_json_path)

    # Validate ground truth file
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found at {args.ground_truth}")
        exit(1)

    with open(args.ground_truth, 'r') as f:
        ground_truth_list = json.load(f)

    if all_commit_hashes is not None and all_commit_embeddings is not None and query_entries:
        # Initialize results
        recommendation_results = []

        # Process each query entry
        for entry in query_entries:
            query_hash = entry["query"]
            target_hashes = entry["targets"]

            print(f"\nProcessing query commit: {query_hash}")
            
            # --- Top-N similar commits for recommendation output ---
            similar_commits_with_scores = find_similar_commits(
                commit_input_hash=query_hash,
                n=num_similar_commits,
                commit_hashes=all_commit_hashes,
                all_embeddings=all_commit_embeddings
            )
            
            # Get recommended files based on similar commits
            recommended_files = get_recommended_files(
                query_hash=query_hash,
                similar_commits=similar_commits_with_scores,
                repo_path='.',
                remote_url=f"https://github.com/{args.user}/{args.repo_name}.git",
                n=num_similar_files
            )
            
            # Store results
            recommendation_results.append({
                "queryCommit": query_hash,
                "recommendedFiles": recommended_files
            })
            
            # Log the recommendations
            logging.info(f"\nRecommendations for commit {query_hash}:")
            logging.info("Similar commits:")
            for commit in similar_commits_with_scores:
                logging.info(f"Commit: {commit['hash']}, Similarity: {commit['similarity']:.4f}")
            logging.info("\nRecommended files:")
            for file_info in recommended_files:
                logging.info(f"File: {file_info['file']}, Score: {file_info['score']:.4f}")
            
        # Save recommendation results
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        with open(save_results_path, "w") as f:
            json.dump(recommendation_results, f, indent=4)
            logging.info(f"\n✔️ Recommendation results written to {save_results_path}")
            
        # Evaluate recommendations
        evaluation_metrics = evaluate_recommendations(recommendation_results, ground_truth_list)
        
        # Log evaluation results
        logging.info("\nEvaluation Metrics:")
        logging.info(f"Precision: {evaluation_metrics['precision']:.4f}")
        logging.info(f"Recall: {evaluation_metrics['recall']:.4f}")
        logging.info(f"F1 Score: {evaluation_metrics['f1']:.4f}")
        logging.info(f"Hit@{num_similar_files}: {evaluation_metrics['hit_at_k']:.4f}")
        
        # Save evaluation results
        with open(eval_results_path, "w") as f:
            json.dump(evaluation_metrics, f, indent=4)
            logging.info(f"\n✔️ Evaluation metrics written to {eval_results_path}")

    else:
        print("No valid data found for processing.")

    print("Analysis completed.")
    