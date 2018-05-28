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


class BVHDataset3(Dataset):
    def __init__(self, data_path):
        Dataset.__init__(self)
        bvh = BVH()
        bvh.load(data_path)
        self.bvh = bvh
        self.phases = np.loadtxt(data_path.replace('bvh', 'phase'))[
            start_index:]
        self.phase_deltas = (self.phases[1:] - self.phases[:-1])

        self.phase_deltas_mean = np.mean(self.phase_deltas, axis=0)
        self.phase_deltas_std = np.std(self.phase_deltas, axis=0)
        self.phase_deltas = (self.phase_deltas -
                             self.phase_deltas_mean)/self.phase_deltas_std
        # self.phase_deltas *= phase_scale

        self.root_motions = bvh.motions[start_index:, :num_of_root_infos]
        self.trajectories = self.root_motions[:, [0, 2, 4]]
        self.trajectories = self.trajectories[1:] - self.trajectories[:-1]

        print(self.root_motions[:-1, [4]].shape)
        self.trajectories = np.concatenate(
            [self.trajectories, self.root_motions[:-1, [4]]], axis=1)
        self.trajectory_mean = np.mean(self.trajectories, axis=0)
        self.trajectory_std = np.std(self.trajectories, axis=0)
        self.trajectories = (self.trajectories -
                             self.trajectory_mean)/self.trajectory_std

        self.angles = self.bvh.motion_angles[start_index:]
        self.angles_mean = np.mean(self.angles, axis=0)
        self.angles_std = np.std(self.angles, axis=0)
        self.angles = (self.angles-self.angles_mean) / \
            (self.angles_std+(self.angles_std == 0))
        print(self.angles_mean.shape)
        # print(self.angles_std)
        # print(self.angles)
        print(self.angles.shape)
        self.angles_delta = self.angles[1:] - self.angles[:-1]
        self.angles_delta_mean = np.mean(self.angles_delta, axis=0)
        self.angles_delta_std = np.std(self.angles_delta, axis=0)
        self.angles_delta = (self.angles_delta-self.angles_delta_mean) / \
            (self.angles_delta_std+(self.angles_delta_std == 0))

    def __len__(self):
        return self.bvh.length - trajectory_length - 1 - start_index

    @property
    def in_features(self):
        return trajectory_length*num_of_trajectory_infos + \
            self.bvh.num_of_angles*2

    @property
    def out_features(self):
        return trajectory_length*num_of_trajectory_infos + \
            self.bvh.num_of_angles*2 + 1

    def __getitem__(self, idx):
        X = np.concatenate(
            (self.trajectories[idx:idx+trajectory_length]
             .reshape((1, trajectory_length *
                       num_of_trajectory_infos)),
             self.angles[idx]
             .reshape((1, self.bvh.num_of_angles)),
             self.angles_delta[idx].reshape((1, self.bvh.num_of_angles))
             ),
            axis=1)
        Y = np.concatenate(
            (self.trajectories[idx+1:idx+trajectory_length+1]
             .reshape((1, trajectory_length *
                       num_of_trajectory_infos)),
             self.angles[idx+1]
             .reshape((1, self.bvh.num_of_angles)),
             self.angles_delta[idx+1].reshape((1, self.bvh.num_of_angles)),
             self.phase_deltas[idx].reshape((1, 1))),
            axis=1)
        return (X, self.phases[idx], Y)


if __name__ == '__main__':
    # dataset = BVHDataset(base_dir + bvh_path)
    dataset = BVHDataset3(base_dir+bvh_path)
    # print(dataset[0])
    print(len(dataset))
    print(dataset.trajectory_mean.shape)
    print(dataset.angles_std.shape)
