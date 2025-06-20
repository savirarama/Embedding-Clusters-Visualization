import torch
import torch.nn as nn
import numpy as np
from mlp import MLPEmbedding
import json
import argparse
from datetime import datetime
import chromadb 
import os

# --- ChromaDB Specific Configurations ---
# Define a batch size for processing and then upserting to ChromaDB.
# This helps manage memory usage and optimizes performance for large datasets.

parser = argparse.ArgumentParser(description="Arguments required for processing and storing embeddings with MLP and ChromaDB")
parser.add_argument("--params", type=str, required=True, help = "Path of model parameters")
parser.add_argument('--commit-hashes-path', type=str, required=True, help= 'Path of JSON file that stores list of commit hashes')
parser.add_argument("--initialdb-path", type=str, required=True,
                    help="Path to the directory of the initial ChromaDB instance (source embeddings)")
parser.add_argument("--initial-collection-name", type=str, required=True,
                    help="Name of the collection in the initial ChromaDB that stores source embeddings")
parser.add_argument("--processeddb-path", type=str, required=True,
                    help="Path to the directory of the processed ChromaDB instance (target embeddings)")
parser.add_argument("--processed-collection-name", type=str, required=True,
                    help="Name of the collection in the processed ChromaDB for storing transformed embeddings")
parser.add_argument('--batch-size', type=int, required=True,
                    help = 'Batch size for inserting data into database')

args = parser.parse_args()

print(f"Loading commit hashes from {args.commit_hashes_path}.")
with open(args.commit_hashes_path, 'r') as file:
    commit_hashes = json.load(file)
print(f"Loaded {len(commit_hashes)} commit hashes.")

# Model initialization
# Ensure MLPEmbedding is correctly defined in mlp.py and accessible
model = MLPEmbedding(768, 64) # Input dim 768, Output dim 64 (based on your original code)

# Load model parameters
state_dict = torch.load(args.params, map_location=torch.device('cpu')) 
model.load_state_dict(state_dict)
model.eval() # Set model to evaluation mode

# --- ChromaDB Client Initialization ---
initial_chroma_client = None
processed_chroma_client = None

try:
    print(f"Connecting to initial ChromaDB at '{args.initialdb_path}'.")
    # Ensure the directory exists (important if this is the first time running ChromaDB here)
    os.makedirs(args.initialdb_path, exist_ok=True)
    initial_chroma_client = chromadb.PersistentClient(path=args.initialdb_path)
    initial_collection = initial_chroma_client.get_collection(name=args.initial_collection_name)
    print(f"Connected to initial collection '{args.initial_collection_name}'. Total items: {initial_collection.count()}")

    print(f"Connecting to processed ChromaDB at '{args.processeddb_path}'.")
    os.makedirs(args.processeddb_path, exist_ok=True)
    processed_chroma_client = chromadb.PersistentClient(path=args.processeddb_path)
    # Use get_or_create_collection as the processed DB might be new or already exist
    processed_collection = processed_chroma_client.get_or_create_collection(name=args.processed_collection_name)
    print(f"Connected to processed collection '{args.processed_collection_name}'. Total items: {processed_collection.count()}")

    # --- Prepare for Batch Processing and Storing ---
    processed_ids = []
    processed_embeddings_list = []
    processed_metadatas = []
    processed_documents = [] # Optional, but good practice to include if source text exists

    print("Processing commit hashes and collecting results in batches...")
    with torch.no_grad(): # Disable gradient calculations for inference
        for i, commit_hash in enumerate(commit_hashes):
            if (i + 1) % 100 == 0:
                print(f"  Processing commit {i+1}/{len(commit_hashes)}")

            # 1. READ from initial ChromaDB
            result = initial_collection.get(
                ids=[commit_hash],
                include=['embeddings', 'metadatas', 'documents']
            )

            if result and result['ids'] and len(result['ids']) > 0:
                original_embedding = result['embeddings'][0] # List[float]
                original_metadata = result['metadatas'][0] # Dict
                original_document = result['documents'][0] if result['documents'] else None

                embedding_tensor = torch.tensor(original_embedding, dtype=torch.float32)

                output_tensor = model(embedding_tensor)

                processed_ids.append(commit_hash)
                processed_embeddings_list.append(output_tensor.tolist())

                # Reuse original metadata and add processing timestamp.
                updated_metadata = original_metadata.copy() 
                updated_metadata["ingestion_timestamp"] = datetime.now().isoformat()
                
                processed_metadatas.append(updated_metadata)
                processed_documents.append(original_document or f"Processed Commit: {commit_hash}") # Re-use or create document

                # Perform batch upsert if current batch size is reached
                if len(processed_ids) >= args.batch_size:
                    print(f"  Upserting batch of {len(processed_ids)} processed embeddings...")
                    processed_collection.upsert(
                        ids=processed_ids,
                        embeddings=processed_embeddings_list,
                        metadatas=processed_metadatas,
                        documents=processed_documents
                    )
                    # Clear lists for the next batch
                    processed_ids = []
                    processed_embeddings_list = []
                    processed_metadatas = []
                    processed_documents = []

            else:
                print(f"    -> WARNING: Commit hash {commit_hash} not found in the initial collection.")

    # --- Final Batch Upsert (for any remaining items) ---
    if len(processed_ids) > 0:
        print(f"  Upserting final batch of {len(processed_ids)} processed embeddings...")
        processed_collection.upsert(
            ids=processed_ids,
            embeddings=processed_embeddings_list,
            metadatas=processed_metadatas,
            documents=processed_documents
        )
        print("Final batch processed.")

    print(f"\nSuccessfully processed and saved all relevant embeddings to '{args.processeddb_path}'.")
    print(f"Total items in processed collection: {processed_collection.count()}")

except chromadb.API.exceptions.CollectionNotFoundError as e:
    print(f"Error: The initial collection '{args.initial_collection_name}' was not found in '{args.initialdb_path}'. "
          f"Please ensure it exists and the name is correct. Details: {e}")
except FileNotFoundError as e:
    print(f"Error: File not found. Please check your paths. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:

    if initial_chroma_client:
        del initial_chroma_client
        pass
    if processed_chroma_client:
        del processed_chroma_client
        pass
    print("All ChromaDB client resources released.")