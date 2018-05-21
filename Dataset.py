import torch
from torch.utils.data.dataset import Dataset


class BVHDataSet(Dataset):
    def __init__(self, bvh):
        Dataset.__init__(self)
        self.bvh = bvh

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
