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


def get_embedding(commit_hash, commit_hashes_list, all_embeddings):
    try:
        idx = commit_hashes_list.index(commit_hash)
        return torch.tensor(all_embeddings[idx], dtype=torch.float32)
    except ValueError:
        print(f"Commit hash {commit_hash} not found in commit_hashes list.")
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
    repos= ["accumulo", "ambari", "camel", "calcite", "cassandra", "flink", "hadoop", "hive", "ignite", "lucene-solr", "oozie", "pig", "spark", "struts", "thrift", "wicket"]
    #repos=["accumulo", "ambari", "camel", "calcite", "cassandra", "flink", "hadoop", "hbase", "hive", "ignite", "lucene-solr", "oozie", "pig", "spark", "struts", "thrift", "wicket"]

    model = MLPEmbeddingMultiRepo(768, 64) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    triplet_loss = nn.TripletMarginLoss(margin=0.5)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_flag = False

    for epoch in range(epochs):
        if early_stop_flag:
            break

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_train_losses = []
        epoch_val_losses = []

        avg_epoch_val_loss = float('inf') 

        for repo in repos:
            print(f"Processing repository: {repo}")
            commit_hashes_path = f"data/{repo}/commit_hashes.json"
            embeddings_path = f"data/{repo}/embeddings/initial_embeddings.npy"
            train_triplets_path = f"data/{repo}/triplets/train_triplets.json"
            test_triplets_path = f"data/{repo}/triplets/test_triplets.json"

            # 1. Load commit hashes
            print(f"Loading commit hashes from {commit_hashes_path}.")
            try:
                with open(commit_hashes_path, 'r') as file:
                    commit_hashes = json.load(file)
                print(f"Loaded {len(commit_hashes)} commit hashes.")
            except FileNotFoundError:
                print(f"Error: Commit hashes file not found at {commit_hashes_path}. Skipping repo.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {commit_hashes_path}. Skipping repo.")
                continue

            # 2. Load actual embeddings for the repo (from .npy file)
            print(f"Loading embeddings from {embeddings_path}.")
            try:
                repo_embeddings = np.load(embeddings_path)
                print(f"Loaded {repo_embeddings.shape[0]} embeddings with shape {repo_embeddings.shape} for {repo}.")
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
                continue # Skip this repository and move to the next one

            # 3. Load triplet data
            try:
                with open(train_triplets_path, 'r') as file:
                    triplets_train_data = json.load(file)
                with open(test_triplets_path, 'r') as file:
                    triplets_test_data = json.load(file)
            except FileNotFoundError:
                print(f"Error: Triplet files not found for {repo}. Skipping repo.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from triplet files for {repo}. Skipping repo.")
                continue

            anchors_train = [entry['anchor'] for entry in triplets_train_data]
            positives_train = [entry['positive'] for entry in triplets_train_data]
            negatives_train = [entry['negative'] for entry in triplets_train_data]
        
            anchors_test = [entry['anchor'] for entry in triplets_test_data]
            positives_test = [entry['positive'] for entry in triplets_test_data]
            negatives_test = [entry['negative'] for entry in triplets_test_data]


            # --- Fetch embeddings and filter for complete triplets ---
            print("Fetching and filtering triplets...")
            filtered_anchors_train = []
            filtered_positives_train = []
            filtered_negatives_train = []
            
            for i in range(len(anchors_train)):
                anchor_emb = get_embedding(anchors_train[i], commit_hashes, repo_embeddings)
                pos_emb = get_embedding(positives_train[i], commit_hashes, repo_embeddings)
                neg_emb = get_embedding(negatives_train[i], commit_hashes, repo_embeddings)
                
                if anchor_emb is not None and pos_emb is not None and neg_emb is not None:
                    filtered_anchors_train.append(anchor_emb)
                    filtered_positives_train.append(pos_emb)
                    filtered_negatives_train.append(neg_emb)

            filtered_anchors_test = []
            filtered_positives_test = []
            filtered_negatives_test = []

            for i in range(len(anchors_test)):
                anchor_emb = get_embedding(anchors_test[i], commit_hashes, repo_embeddings)
                pos_emb = get_embedding(positives_test[i], commit_hashes, repo_embeddings)
                neg_emb = get_embedding(negatives_test[i], commit_hashes, repo_embeddings)

                if anchor_emb is not None and pos_emb is not None and neg_emb is not None:
                    filtered_anchors_test.append(anchor_emb)
                    filtered_positives_test.append(pos_emb)
                    filtered_negatives_test.append(neg_emb)

            print(f"Number of train triplets retrieved for {repo}: {len(filtered_anchors_train)}")
            print(f"Number of test triplets retrieved for {repo}: {len(filtered_anchors_test)}")

            train_dataset = TripleCommitMultiRepoDataset(filtered_anchors_train, filtered_positives_train, filtered_negatives_train)
            train_dataloader = DataLoader(train_dataset, batch_size=len(filtered_anchors_train), shuffle=True, drop_last=False)

            test_dataset = TripleCommitMultiRepoDataset(filtered_anchors_test, filtered_positives_test, filtered_negatives_test)
            test_dataloader = DataLoader(test_dataset, batch_size=len(filtered_anchors_test), shuffle=False, drop_last=False)

            # --- Training Phase ---
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

            if len(train_dataloader) > 0: 
                avg_repo_train_loss = repo_train_loss / len(train_dataloader)
                epoch_train_losses.append(avg_repo_train_loss)
            else:
                print(f"Train dataloader for {repo} was empty. No training loss recorded for this repo.")

            # --- Validation Phase ---
            model.eval()
            repo_val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (anchor, positive, negative) in enumerate(test_dataloader):
                    anchor_out = nn.functional.normalize(model(anchor), p=2, dim=1)
                    positive_out = nn.functional.normalize(model(positive), p=2, dim=1)
                    negative_out = nn.functional.normalize(model(negative), p=2, dim=1)

                    loss_value = triplet_loss(anchor_out, positive_out, negative_out)
                    repo_val_loss += loss_value.item()

            if len(test_dataloader) > 0: 
                avg_repo_val_loss = repo_val_loss / len(test_dataloader)
                epoch_val_losses.append(avg_repo_val_loss)
            else:
                print(f"Test dataloader for {repo} was empty. No validation loss recorded for this repo.")

            print(f"Repo {repo} - Train Loss: {avg_repo_train_loss:.4f}, Val Loss: {avg_repo_val_loss:.4f}")

        if epoch_train_losses:
            avg_epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_epoch_train_loss)
        else:
            train_losses.append(train_losses[-1] if train_losses else 0.0) # Maintain last loss or 0
            print("No training data processed for this epoch across all repos.")

        if epoch_val_losses:
            avg_epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            val_losses.append(avg_epoch_val_loss)
        else:
            val_losses.append(val_losses[-1] if val_losses else 0.0) # Maintain last loss or 0
            print("No validation data processed for this epoch across all repos.")


        print(f"Epoch {epoch+1}/{epochs}, Average Train Loss: {train_losses[-1]:.4f}, Average Val Loss: {val_losses[-1]:.4f}")

        # --- Early Stopping Logic ---
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