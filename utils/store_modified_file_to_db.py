import os
import glob
import json
import numpy as np
import argparse
import chromadb
from get_modified_files import get_modified_files_from_commit

parser = argparse.ArgumentParser(description="Arguments required adding modified files of each commit hash")
parser.add_argument("--collection-name", type=str, required=True,
                    help="Name of the collection in the initial ChromaDB that stores source embeddings")
parser.add_argument("--db-path", type=str, required=True,
                    help="Path to the directory of the processed ChromaDB instance (target embeddings)")
parser.add_argument('--batch-size', type=int, default = 3000,
                    help = 'Batch size for inserting data into database')
parser.add_argument("--repo-name", type=str, required=True,
                    help="Repository name")
parser.add_argument("--user", type=str, default="apache")

args = parser.parse_args()

chroma_client = chromadb.PersistentClient(path=args.db_path)
collection = chroma_client.get_collection(name=args.collection_name)
print(f"Connected to collection '{args.collection_name}'. Total items: {collection.count()}")

results = collection.get(
        where={"repo_name" : args.repo_name},
        include=['metadatas', 'documents']
    )

processed_ids = []
processed_metadatas = []
processed_documents = []
#processed_embeddings = []

for i, commit_id in enumerate(results['ids']):
    if (i + 1) % 100 == 0:
        print(f"  Processing commit {i+1}/{len(results['ids'])}")
    original_metadata = results['metadatas'][i]
    original_document = results['documents'][i]
    #original_embedding = results['embeddings'][i]
    modified_files = get_modified_files_from_commit(commit_id, remote_url=f"https://github.com/{args.user}/{args.repo_name}.git")

    modified_files_json_string = json.dumps(modified_files)

    updated_metadata = original_metadata.copy()
    updated_metadata['modified_files'] = modified_files_json_string


    processed_ids.append(commit_id)
    processed_metadatas.append(updated_metadata)
    processed_documents.append(original_document)
    #processed_embeddings.append(original_embedding)
    if len(processed_ids) >= args.batch_size:
        print(f"  Upserting batch of {len(processed_ids)} processed embeddings...")
        collection.upsert(
                ids=processed_ids,
                metadatas=processed_metadatas,
                documents=processed_documents
                #embeddings=processed_embeddings
                )
        # Clear lists for the next batch
        processed_ids = []
        processed_metadatas = []
        processed_documents = []
        #processed_embeddings = []
    
if len(processed_ids) > 0:
    print(f"  Upserting final batch of {len(processed_ids)} processed embeddings...")
    collection.upsert(
            ids=processed_ids,
            metadatas=processed_metadatas,
            documents=processed_documents
            #embeddings=processed_embeddings
                    )
    print("Final batch processed.")




