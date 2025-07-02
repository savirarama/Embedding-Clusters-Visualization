import torch.nn as nn

class MLPEmbedding(nn.Module):
    def __init__(self, input_dim=768, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 128), 
                                 nn.Tanh(), 
                                 nn.Linear(128, 64))
                    
    def forward(self, x):
        return self.net(x)

class MLPEmbeddingMultiRepo(nn.Module):
    def __init__(self, input_dim=768, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 128), 
                                 nn.Tanh(), 
                                 nn.BatchNorm1d(128),
                                 nn.Linear(128, 64))
                    
    def forward(self, x):
        return self.net(x)