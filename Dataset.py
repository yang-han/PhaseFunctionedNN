import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from hyperparams import *
from BVH import BVH


class BVHDataset(Dataset):
    def __init__(self, data_path):
        Dataset.__init__(self)
        bvh = BVH()
        bvh.load(data_path)
        self.bvh = bvh
        self.phases = np.loadtxt(data_path.replace('bvh', 'phase'))
        print(self.phases.shape)
        root_motions = bvh.motions[:, :num_of_root_infos]
        self.root_deltas = root_motions[1:] - root_motions[:-1]
        print(self.root_deltas.shape)
        self.root_deltas[:, 0] *= delta_scale
        self.root_deltas[:, 2] *= delta_scale
        for i in range(6):
            items = self.root_deltas[:, i]
            print(np.max(items), np.min(items), np.mean(items))
        self.phase_deltas = self.phases[1:] - self.phases[:-1]
        self.phase_deltas *= phase_scale
        items = self.phase_deltas
        print(np.max(items), np.min(items), np.mean(items))

    def __len__(self):
        return self.bvh.length - trajectory_length - 1

    @property
    def in_features(self):
        return trajectory_length*num_of_root_infos + self.bvh.num_of_angles

    @property
    def out_features(self):
        return trajectory_length*num_of_root_infos + self.bvh.num_of_angles + 1

    def __getitem__(self, idx):
        # print(self.root_deltas[idx:idx+trajectory_length].shape)
        return (np.concatenate((self.root_deltas[idx:idx+trajectory_length]
                                .reshape((1, trajectory_length *
                                          num_of_root_infos)),
                                self.bvh.motion_angles[idx]
                                .reshape((1, self.bvh.num_of_angles))),
                               axis=1),
                self.phases[idx],
                np.concatenate((self.root_deltas[idx+1:idx+trajectory_length+1]
                                .reshape((1, trajectory_length *
                                          num_of_root_infos)),
                                self.phase_deltas[idx].reshape((1, 1)),
                                self.bvh.motion_angles[idx+1]
                                .reshape((1, self.bvh.num_of_angles))),
                               axis=1))


class BVHDataset2(Dataset):
    def __init__(self, data_path):
        Dataset.__init__(self)
        bvh = BVH()
        bvh.load(data_path)
        self.bvh = bvh
        self.phases = np.loadtxt(data_path.replace('bvh', 'phase'))
        self.root_motions = bvh.motions[:, :num_of_root_infos]
        self.phase_deltas = self.phases[1:] - self.phases[:-1]
        self.phase_deltas *= phase_scale
        print(self.phase_deltas[200:300])
        # items = self.phase_deltas
        # print(np.max(items), np.min(items), np.mean(items))

    def __len__(self):
        return self.bvh.length - trajectory_length - 1 - 300

    @property
    def in_features(self):
        return trajectory_length*num_of_root_infos + self.bvh.num_of_angles

    @property
    def out_features(self):
        return trajectory_length*num_of_root_infos + self.bvh.num_of_angles + 1

    def __getitem__(self, idx):
        idx += start_index
        # print(self.root_deltas[idx:idx+trajectory_length].shape)
        return (np.concatenate((self.root_motions[idx:idx+trajectory_length]
                                .reshape((1, trajectory_length *
                                          num_of_root_infos)),
                                self.bvh.motion_angles[idx]
                                .reshape((1, self.bvh.num_of_angles))),
                               axis=1),
                self.phases[idx],
                np.concatenate(
                    (self.root_motions[idx+1:idx+trajectory_length+1]
                     .reshape((1, trajectory_length *
                               num_of_root_infos)),
                     self.phase_deltas[idx].reshape((1, 1)),
                     self.bvh.motion_angles[idx+1]
                     .reshape((1, self.bvh.num_of_angles))),
            axis=1))


if __name__ == '__main__':
    dataset = BVHDataSet(base_dir + bvh_path)
    print(dataset[0])
    print(len(dataset))
