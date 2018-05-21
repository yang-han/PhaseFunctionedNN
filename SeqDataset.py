import torch
from torch.utils.data.dataset import Dataset


class SeqBVHDataset(Dataset):
    def __init__(self, bvh, num_of_frames):
        Dataset.__init__(self)
        self.bvh = bvh
        self.motions = bvh.motions
        self.num_of_frames = num_of_frames

    def __len__(self):
        return self.bvh.length - self.num_of_frames

    def __getitem__(self, idx):
        return (torch.tensor(
            self.bvh.motion_angles[idx:idx+self.num_of_frames]).
            view(-1),
            torch.tensor(
            self.bvh.motion_angles[idx+self.num_of_frames]).
            view(-1)
        )
