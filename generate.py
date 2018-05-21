import torch
import numpy as np
from PFNN import BaseNet, PFNN
from BVH import BVH
from hyperparams import *
from Dataset import BVHDataset

frames = 1000


def base_net_inference():
    bvh = BVH()
    bvh.load(bvh_path)
    net = BaseNet(num_of_frames*bvh.num_of_angles, bvh.num_of_angles).double()
    net.load_state_dict(torch.load('models/pfnn_params29.pkl'))
    init_x = torch.tensor(bvh.motion_angles[:num_of_frames]).view(-1)
    x = init_x
    motions = torch.zeros((frames, bvh.num_of_angles), requires_grad=False)
    # state = torch.tensor(bvh.motions[num_of_frames+1, 3:].reshape(-1))
    y = torch.tensor(bvh.motion_angles[num_of_frames].reshape(-1))
    # print(list(net.parameters()))
    # for p in net.parameters():
    #     print(p)
    # for k in net.state_dict():
    #     print(k)
    #     print(net.state_dict()[k])
    #     # xx.copy_(params)
    for i in range(frames):
        motions[i] = y
        x = torch.cat((x[bvh.num_of_angles:], y), 0)
        print(x.shape)
        y = net(torch.tensor(x).view(-1))

        # print(state)
    # print(motions.detach().numpy().shape)
    all_states = np.concatenate((np.zeros((frames, 3)),
                                 np.ones((frames, 3))*100,
                                 motions.detach().numpy()), axis=1)
    print(all_states.shape)
    bvh.save(output_path, all_states)


def pfnn_inference():
    dataset = BVHDataset(base_dir + bvh_path)
    pfnn = PFNN(dataset.in_features, dataset.out_features).float().cuda()
    pfnn.load_state_dict(torch.load('models/pfnn_params19.pkl'))
    bvh = dataset.bvh
    init_state = dataset.bvh.motions[0, :num_of_root_infos]
    init_angles = bvh.motion_angles[0]
    phase = 0
    trajectory = dataset[0][:num_of_root_infos*trajectory_length]
    motions = torch.zeros(
        (frames, bvh.num_of_angles+num_of_root_infos), requires_grad=False)
    angles = init_angles
    state = init_state
    print(angles.shape)
    print(trajectory)
    for i in range(frames):
        x = torch.cat(
            (torch.tensor(trajectory, dtype=torch.float32)
             .view(1, num_of_root_infos*trajectory_length),
             torch.tensor(angles, dtype=torch.float32).view(1, 90)), dim=1)
        y = net(x, phase)
        trajectory = y[:trajectory_length*num_of_root_infos]
        phase += y[trajectory_length*num_of_frames]/phase_scale
        angles = y[trajectory_length*num_of_frames+1:]
        delta_state = trajectory[:num_of_root_infos]
        delta_state[0] /= delta_scale
        delta_state[2] /= delta_scale
        state += delta_state
        motions[i] = torch.cat((state, angles), dim=1)
    bvh.save(motions.numpy())


if __name__ == '__main__':
    pfnn_inference()
