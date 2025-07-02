import torch
import torch.nn as nn
import numpy as np
from mlp import MLPEmbedding, MLPEmbeddingMultiRepo
import json
import argparse


parser = argparse.ArgumentParser(description="Arguments required for processing and storing embeddings with MLP and ChromaDB")
parser.add_argument("--params", type=str, required=True, help = "Path of model parameters")
parser.add_argument('--input-path', type=str, required=True, help= 'Path of input npy file that stores embeddings')
parser.add_argument('--output-path', type=str, required=True, help= 'Path of output npy file')
parser.add_argument('--multi-repo', action='store_true', help='Use the multi-repo model')


args = parser.parse_args()

if args.multi_repo:
    model = MLPEmbeddingMultiRepo(768, 64)
else:
    model = MLPEmbedding(768, 64) 

# Load model parameters
state_dict = torch.load(args.params, map_location=torch.device('cpu')) 
model.load_state_dict(state_dict)
model.eval() 

print(f"Loaded model from {args.params}")

embeddings = np.load(args.input_path)
embeddings_tensor = torch.from_numpy(embeddings).float()

print(f"Input embeddings shape: {embeddings.shape}")

output_tensor = model(embeddings_tensor)
np.save(args.output_path, output_tensor.detach().numpy())

print(f"Output embeddings saved to {args.output_path}")

