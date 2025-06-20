import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from triplet_dataset import TripleCommitDataset
from mlp import MLPEmbedding
import argparse
import chromadb

def get_embedding(commit_hash, collection):
    try:
        results = collection.get(
            ids=[commit_hash],        # Filter by specific ID
            include=['embeddings'] # What to retrieve
            )
        if results and results['ids'] and len(results['ids']) > 0:
            retrieved_id = results['ids'][0]
            retrieved_embedding = results['embeddings'][0]
            return retrieved_embedding
        else:
            print(f"No commits found with commit hash = {commit_hash}.")
            return None
    except Exception as e:
        print(f"An error occurred during retrieval of commit hash {commit_hash}: {e}")
        return None
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Arguments required for training MLP")
    parser.add_argument('--lr', type=float, required=True, help= 'Learning rate')
    parser.add_argument('--commit-hashes-path', type=str, required=True, help= 'Path of JSON file that stores list of commit hashes')
    parser.add_argument('--model-save-path', type=str, required=True, help= 'Path to store the parameters of trained MLP')
    parser.add_argument('--train-triplets', type=str, required=True, help= 'Path of JSON file that stores list of triplets for training')
    parser.add_argument('--test-triplets', type=str, required=True, help= 'Path of JSON file that stores list of triplets for testing')
    parser.add_argument('--db-path', type=str, required=True, help= 'Path of database that stores commit embeddings')
    parser.add_argument('--collection-name', type=str, required=True, help= 'Name of collection that stores the commit embeddings')
    parser.add_argument('--epochs', type=int, required=True, help= 'Number of epochs for training')
    parser.add_argument('--patience', type=int, default=10, help= 'Number of epochs to wait before early stopping')
    parser.add_argument('--min-delta', type=float, default=0.001, help= 'Minimum delta for early stopping')
    parser.add_argument('--loss-graph-path', type=str, required=True, help= 'Path to store the resulting loss graph')
    args = parser.parse_args()

    epochs = args.epochs
    patience = args.patience
    min_delta = args.min_delta
    model_save_path = args.model_save_path

    print(f"Loading commit hashes from {args.commit_hashes_path}.")
    with open(args.commit_hashes_path, 'r') as file:
        commit_hashes = json.load(file)
    print(f"Loaded {len(commit_hashes)} commit hashes.")

    # Load the triplet data from the JSON file
    with open(args.train_triplets, 'r') as file:
        triplets_train_data = json.load(file)

    with open(args.test_triplets, 'r') as file:
        triplets_test_data = json.load(file)

    anchors_train = [entry['anchor'] for entry in triplets_train_data]
    positives_train = [entry['positive'] for entry in triplets_train_data]
    negatives_train = [entry['negative'] for entry in triplets_train_data]
    
    anchors_test = [entry['anchor'] for entry in triplets_test_data]
    positives_test = [entry['positive'] for entry in triplets_test_data]
    negatives_test = [entry['negative'] for entry in triplets_test_data]

    #  Initialize ChromaDB Client
    print(f"Connecting to ChromaDB persistent client at: {args.db_path}")
    client = chromadb.PersistentClient(path=args.db_path)

    # --- 2. Get the Collection ---
    print(f"Getting collection: '{args.collection_name}'")
    try:
        collection = client.get_collection(name=args.collection_name)
        print(f"Collection '{args.collection_name}' loaded. Total items: {collection.count()}")
    except Exception as e:
        print(f"Error getting collection '{args.collection_name}': {e}")
        print("Please ensure the collection exists and the path is correct.")
        exit()

    # Initialize lists to store the retrieved embeddings
    anchors_train_embeddings = []
    positives_train_embeddings = []
    negatives_train_embeddings = []

    anchors_test_embeddings = []
    positives_test_embeddings = []
    negatives_test_embeddings = []

    # Obtaining embeddings from database
    try:
    # --- Connect to the ChromaDB database ---

        # --- Fetch embeddings for each commit hash ---
        print("Fetching train anchor embeddings...")
        anchors_train_embeddings = [get_embedding(anchor, collection) for anchor in anchors_train]

        print("Fetching train positive embeddings...")
        positives_train_embeddings = [get_embedding(pos, collection) for pos in positives_train]

        print("Fetching train negative embeddings...")
        negatives_train_embeddings = [get_embedding(neg, collection) for neg in negatives_train]
        
        print("Fetching test anchor embeddings...")
        anchors_test_embeddings = [get_embedding(anchor, collection) for anchor in anchors_test]

        print("Fetching test positive embeddings...")
        positives_test_embeddings = [get_embedding(pos, collection) for pos in positives_test]

        print("Fetching test negative embeddings...")
        negatives_test_embeddings = [get_embedding(neg, collection) for neg in negatives_test]
    

        # Optional: Filter out any 'None' values if hashes were not found
        anchors_train_embeddings = [emb for emb in anchors_train_embeddings if emb is not None]
        positives_train_embeddings = [emb for emb in positives_train_embeddings if emb is not None]
        negatives_train_embeddings = [emb for emb in negatives_train_embeddings if emb is not None]

        anchors_test_embeddings = [emb for emb in anchors_test_embeddings if emb is not None]
        positives_test_embeddings = [emb for emb in positives_test_embeddings if emb is not None]
        negatives_test_embeddings = [emb for emb in negatives_test_embeddings if emb is not None]

        # --- Rebuild lists with only complete triplets ---
        print("Filtering for complete triplets...")

        # For training data
        filtered_anchors_train = []
        filtered_positives_train = []
        filtered_negatives_train = []
        for anchor, pos, neg in zip(anchors_train_embeddings, positives_train_embeddings, negatives_train_embeddings):
            if anchor is not None and pos is not None and neg is not None:
                filtered_anchors_train.append(anchor)
                filtered_positives_train.append(pos)
                filtered_negatives_train.append(neg)

        # For testing data
        filtered_anchors_test = []
        filtered_positives_test = []
        filtered_negatives_test = []
        for anchor, pos, neg in zip(anchors_test_embeddings, positives_test_embeddings, negatives_test_embeddings):
            if anchor is not None and pos is not None and neg is not None:
                filtered_anchors_test.append(anchor)
                filtered_positives_test.append(pos)
                filtered_negatives_test.append(neg)
    finally:
        # --- Close the database connection ---
        if 'conn' in locals() and conn:
            conn.close()

    print("\n\n")
    print(f"Number of train anchor embeddings retrieved: {len(filtered_anchors_train)}")
    print(f"Number of train positive embeddings retrieved: {len(filtered_positives_train)}")
    print(f"Number of train negative embeddings retrieved: {len(filtered_negatives_train)}")
    print("\n\n")
    print(f"Number of test anchor embeddings retrieved: {len(filtered_anchors_test)}")
    print(f"Number of test positive embeddings retrieved: {len(filtered_positives_test)}")
    print(f"Number of test negative embeddings retrieved: {len(filtered_negatives_test)}")

    train_dataset = TripleCommitDataset(filtered_anchors_train, filtered_positives_train, filtered_negatives_train)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)

    test_dataset = TripleCommitDataset(filtered_anchors_test, filtered_positives_test, filtered_negatives_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)

    # Instantiate MLP
    model = MLPEmbedding(768, 64)

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    triplet_loss = nn.TripletMarginLoss(margin=0.5)

    # Train MLP
    train_losses = []
    val_losses = []

    best_val_loss = float('inf') # Initialize with a very high value
    epochs_no_improve = 0 # Counter for epochs without improvement
    early_stop = False # Flag to control early stopping

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for anchor, positive, negative in train_dataloader:
            optimizer.zero_grad()
            

            #Normalize the output
            anchor_out = nn.functional.normalize(model(anchor), p=2, dim=1)
            positive_out = nn.functional.normalize(model(positive), p=2, dim=1)
            negative_out = nn.functional.normalize(model(negative), p=2, dim=1)
            
            loss_value = triplet_loss(anchor_out, positive_out, negative_out)

            loss_value.backward()
            optimizer.step()
            
            total_train_loss += loss_value.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculations for validation
            for anchor, positive, negative in test_dataloader:
                anchor_out = nn.functional.normalize(model(anchor), p=2, dim=1)
                positive_out = nn.functional.normalize(model(positive), p=2, dim=1)
                negative_out = nn.functional.normalize(model(negative), p=2, dim=1)

                loss_value = triplet_loss(anchor_out, positive_out, negative_out)
                total_val_loss += loss_value.item()

        avg_val_loss = total_val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model state
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Saving model to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss for {patience} consecutive epochs.")
                early_stop = True
                break # Exit the training loop

        if early_stop:
            break # Exit the outer epoch loop
            
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

    # Save the figure before showing it
    plt.savefig(args.loss_graph_path, dpi=300, bbox_inches='tight')  # You can change the filename and format

    plt.show()
