import torch
from torch.utils.data import Dataset

class TripleCommitDataset(Dataset):
    def __init__(self, anchors, positive, negative):
        self.anchors = torch.FloatTensor(anchors)
        self.positive = torch.FloatTensor(positive)
        self.negative = torch.FloatTensor(negative)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        pos = self.positive[idx]
        neg = self.negative[idx]
        return anchor, pos, neg