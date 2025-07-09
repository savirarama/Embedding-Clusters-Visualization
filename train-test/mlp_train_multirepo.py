import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from triplet_dataset import TripleCommitMultiRepoDataset 
from mlp import MLPEmbeddingMultiRepo 
import argparse

def get_embedding_optimized(commit_hash, commit_hash_to_idx_map, all_embeddings):
    try:
        idx = commit_hash_to_idx_map[commit_hash]
        return torch.tensor(all_embeddings[idx], dtype=torch.float32)
    except KeyError:
        return None
    except Exception as e:
        print(f"An error occurred during retrieval of commit hash {commit_hash}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Arguments required for training MLP")
    parser.add_argument('--lr', type=float, required=True, help= 'Learning rate')
    parser.add_argument('--model-save-path', type=str, required=True, help= 'Path to store the parameters of trained MLP')
    parser.add_argument('--epochs', type=int, required=True, help= 'Number of epochs for training')
    parser.add_argument('--patience', type=int, default=10, help= 'Number of epochs to wait before early stopping')
    parser.add_argument('--min-delta', type=float, default=0.001, help= 'Minimum delta for early stopping')
    parser.add_argument('--loss-graph-path', type=str, required=True, help= 'Path to store the resulting loss graph')

    args = parser.parse_args()

    epochs = args.epochs
    patience = args.patience
    min_delta = args.min_delta
    model_save_path = args.model_save_path
    repos = ["accumulo", "ambari", "camel", "calcite", "cassandra", "flink", "hadoop", "hive", "ignite", "lucene-solr", "oozie", "pig", "spark", "struts", "thrift", "wicket"]

    model = MLPEmbeddingMultiRepo(768, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    triplet_loss = nn.TripletMarginLoss(margin=0.5)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_flag = False

    print("Loading ALL validation triplets from all repositories...")
    all_val_triplets_data = []
    val_triplet_paths_all = glob.glob("data/*/triplets/val_triplets.json")
    for path in val_triplet_paths_all:
        try:
            with open(path, 'r') as file:
                all_val_triplets_data.extend(json.load(file)) # Use extend to flatten the list
        except FileNotFoundError:
            print(f"Warning: Validation triplet file not found at {path}. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {path}. Skipping.")
    print(f"Loaded a total of {len(all_val_triplets_data)} global validation triplets.")

    print("Building global commit hash to embedding map...")
    global_commit_to_embedding = {}
    for r_name in repos:
        r_commit_hashes_path = f"data/{r_name}/commit_hashes.json"
        r_embeddings_path = f"data/{r_name}/embeddings/initial_embeddings.npy"
        try:
            with open(r_commit_hashes_path, 'r') as file:
                r_commit_hashes = json.load(file)
            r_embeddings = np.load(r_embeddings_path).astype(np.float32)
            for idx, hash_val in enumerate(r_commit_hashes):
                if hash_val not in global_commit_to_embedding:
                    global_commit_to_embedding[hash_val] = r_embeddings[idx]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load data for global embedding map from {r_name}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while loading global embeddings for {r_name}: {e}")
            continue
    print(f"Global embedding map built with {len(global_commit_to_embedding)} unique commit hashes.")

    # Prepare global validation dataset (once, before epoch loop)
    all_val_anchors = []
    all_val_positives = []
    all_val_negatives = []
    if all_val_triplets_data: # Only process if there are validation triplets loaded
        for entry in all_val_triplets_data:
            anchor_hash = entry['anchor']
            positive_hash = entry['positive']
            negative_hash = entry['negative']

            anchor_emb = global_commit_to_embedding.get(anchor_hash)
            pos_emb = global_commit_to_embedding.get(positive_hash)
            neg_emb = global_commit_to_embedding.get(negative_hash)

            if anchor_emb is not None and pos_emb is not None and neg_emb is not None:
                all_val_anchors.append(torch.tensor(anchor_emb, dtype=torch.float32))
                all_val_positives.append(torch.tensor(pos_emb, dtype=torch.float32))
                all_val_negatives.append(torch.tensor(neg_emb, dtype=torch.float32))
            # else:
            #     print(f"Warning: A validation triplet with hash {anchor_hash}/{positive_hash}/{negative_hash} could not be fully resolved in global embeddings. Skipping.")

    # Create global validation DataLoader
    global_val_dataloader = None
    if all_val_anchors:
        global_val_dataset = TripleCommitMultiRepoDataset(all_val_anchors, all_val_positives, all_val_negatives)
        global_val_dataloader = DataLoader(global_val_dataset, batch_size=len(global_val_dataset), shuffle=False, drop_last=False)
        print(f"Global validation dataset prepared with {len(all_val_anchors)} triplets.")
    else:
        print("No valid global validation triplets found after filtering.")


    for epoch in range(epochs):
        if early_stop_flag:
            break

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_train_losses = []

        for repo in repos:
            print(f"Processing repository: {repo}")
            commit_hashes_path = f"data/{repo}/commit_hashes.json"
            embeddings_path = f"data/{repo}/embeddings/initial_embeddings.npy"
            train_triplets_path = f"data/{repo}/triplets/train_triplets.json"

            # 1. Load commit hashes
            try:
                with open(commit_hashes_path, 'r') as file:
                    commit_hashes = json.load(file)
                commit_hash_to_idx = {hash_val: idx for idx, hash_val in enumerate(commit_hashes)}
            except FileNotFoundError:
                print(f"Error: Commit hashes file not found at {commit_hashes_path}. Skipping repo.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {commit_hashes_path}. Skipping repo.")
                continue

            # 2. Load actual embeddings for the repo (from .npy file) (Re-added error handling)
            try:
                repo_embeddings = np.load(embeddings_path)
                if repo_embeddings.dtype != np.float32:
                    repo_embeddings = repo_embeddings.astype(np.float32)
            except FileNotFoundError:
                print(f"Error: Embeddings file not found at {embeddings_path}. Skipping repo.")
                continue
            except Exception as e:
                print(f"Error loading .npy file from {embeddings_path}: {e}. Skipping repo.")
                continue

            if len(commit_hashes) != repo_embeddings.shape[0]:
                print(f"Mismatch in lengths for {repo}: {len(commit_hashes)} commit hashes but {repo_embeddings.shape[0]} embeddings. Skipping repository.")
                continue

            # 3. Load triplet data for TRAINING (Re-added error handling)
            try:
                with open(train_triplets_path, 'r') as file:
                    triplets_train_data = json.load(file)
            except FileNotFoundError:
                print(f"Error: Train triplet file not found for {repo}. Skipping repo.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from train triplet file for {repo}. Skipping repo.")
                continue

            anchors_train = [entry['anchor'] for entry in triplets_train_data]
            positives_train = [entry['positive'] for entry in triplets_train_data]
            negatives_train = [entry['negative'] for entry in triplets_train_data]

            filtered_anchors_train = []
            filtered_positives_train = []
            filtered_negatives_train = []

            for i in range(len(anchors_train)):
                anchor_emb = get_embedding_optimized(anchors_train[i], commit_hash_to_idx, repo_embeddings)
                pos_emb = get_embedding_optimized(positives_train[i], commit_hash_to_idx, repo_embeddings)
                neg_emb = get_embedding_optimized(negatives_train[i], commit_hash_to_idx, repo_embeddings)

                if anchor_emb is not None and pos_emb is not None and neg_emb is not None:
                    filtered_anchors_train.append(anchor_emb)
                    filtered_positives_train.append(pos_emb)
                    filtered_negatives_train.append(neg_emb)

            print(f"Number of train triplets retrieved for {repo}: {len(filtered_anchors_train)}")

            # --- Training Phase ---
            if filtered_anchors_train:
                train_dataset = TripleCommitMultiRepoDataset(filtered_anchors_train, filtered_positives_train, filtered_negatives_train)
                train_dataloader = DataLoader(train_dataset, batch_size=len(filtered_anchors_train), shuffle=True, drop_last=False)

                model.train()
                repo_train_loss = 0.0

                for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
                    optimizer.zero_grad()

                    anchor_out = nn.functional.normalize(model(anchor), p=2, dim=1)
                    positive_out = nn.functional.normalize(model(positive), p=2, dim=1)
                    negative_out = nn.functional.normalize(model(negative), p=2, dim=1)

                    loss_value = triplet_loss(anchor_out, positive_out, negative_out)

                    loss_value.backward()
                    optimizer.step()

                    repo_train_loss += loss_value.item()

                if len(train_dataloader) > 0: # Ensure no division by zero if dataloader ends up empty
                    avg_repo_train_loss = repo_train_loss / len(train_dataloader)
                    epoch_train_losses.append(avg_repo_train_loss)
                    print(f"Repo {repo} - Train Loss: {avg_repo_train_loss:.4f}")
                else:
                    print(f"Train dataloader for {repo} was empty. No training loss recorded for this repo.")
            else:
                print(f"Train dataloader for {repo} was empty. No training loss recorded for this repo.")

        if epoch_train_losses:
            avg_epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_epoch_train_loss)
        else:
            train_losses.append(train_losses[-1] if train_losses else 0.0)
            print("No training data processed for this epoch across all repos.")


        # --- Global Validation Phase ---
        print("\nStarting global validation for the epoch...")
        if global_val_dataloader: 
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (anchor, positive, negative) in enumerate(global_val_dataloader):
                    anchor_out = nn.functional.normalize(model(anchor), p=2, dim=1)
                    positive_out = nn.functional.normalize(model(positive), p=2, dim=1)
                    negative_out = nn.functional.normalize(model(negative), p=2, dim=1)

                    loss_value = triplet_loss(anchor_out, positive_out, negative_out)
                    total_val_loss += loss_value.item()

            avg_epoch_val_loss = total_val_loss / len(global_val_dataloader)
            val_losses.append(avg_epoch_val_loss)
        else:
            avg_epoch_val_loss = val_losses[-1] if val_losses else 0.0 # Maintain last loss or 0
            val_losses.append(avg_epoch_val_loss)
            print("No valid global validation data was available for this epoch.")


        print(f"Epoch {epoch+1}/{epochs}, Average Train Loss: {train_losses[-1]:.4f}, Average Val Loss: {val_losses[-1]:.4f}")

        if avg_epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Saving model to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss for {patience} consecutive epochs.")
                early_stop_flag = True

    print("Training finished.")

    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss per Epoch with Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(args.loss_graph_path, dpi=300, bbox_inches='tight')
    plt.show()