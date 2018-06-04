import torch
import numpy as np
from PFNN import BaseNet, PFNN
from BVH import BVH
from hyperparams import *
from Dataset import *

frames = 500


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
    print(dataset.in_features, dataset.out_features)
    pfnn = PFNN(dataset.in_features, dataset.out_features).float()
    pfnn.load_state_dict(torch.load('models/pfnn_params99.pkl'))
    bvh = dataset.bvh
    init_state = dataset.bvh.motions[0, :num_of_root_infos]
    init_angles = bvh.motion_angles[0]
    phase = 0
    # print(len(dataset[0]))
    trajectory = dataset[0][0][0][:num_of_root_infos *
                                  trajectory_length]
    # print(len(trajectory))
    motions = np.zeros((frames, bvh.num_of_angles+num_of_root_infos))
    angles = init_angles
    state = init_state
    # print(angles.shape)
    # print(trajectory)
    fake_trajectory = np.zeros((trajectory_length, num_of_root_infos))
    fake_trajectory[:, 1] = 0.2*delta_scale
    for i in range(frames):
        print('i:  ', i)
        x = torch.cat(
            (torch.tensor(trajectory, dtype=torch.float32)
             .view(1, num_of_root_infos*trajectory_length),
             torch.tensor(angles, dtype=torch.float32).view(1, 90)), dim=1)
        y = pfnn(x, phase)
        # print(y.shape)
        trajectory = y[:, :trajectory_length*num_of_root_infos]
        phase += y[0, trajectory_length*num_of_root_infos]/phase_scale
        phase = phase.detach()
        print(phase)
        angles = y[:, trajectory_length*num_of_root_infos+1:]
        # print(y.shape)
        # print(angles.shape)
        delta_state = trajectory[0, :num_of_root_infos]
        delta_state[0] /= delta_scale
        delta_state[2] /= delta_scale
        state += delta_state.detach()
        # print(state.reshape(1, num_of_root_infos).shape)
        # print(angles.detach().numpy().shape)
        motions[i] = np.concatenate(
            (state.reshape(1, num_of_root_infos), angles.detach().numpy()),
            axis=1)
    bvh.save(output_path, motions)
    smoothed_motions = np.concatenate(
        (np.zeros((frames, num_of_root_infos)),
         motions[:, num_of_root_infos:]),
        axis=1)
    bvh.save("smooth_"+output_path, smoothed_motions)


def pfnn_inference_2():
    dataset = BVHDataset2(base_dir + bvh_path)
    print(dataset.in_features, dataset.out_features)
    pfnn = PFNN(dataset.in_features, dataset.out_features).float()
    pfnn.load_state_dict(torch.load('models/2_pfnn_params199.pkl'))
    bvh = dataset.bvh
    init_state = dataset.bvh.motions[0, :num_of_root_infos]
    init_angles = bvh.motion_angles[0]
    phase = 0.2
    # print(len(dataset[0]))
    trajectory = dataset[0][0][0][:num_of_root_infos *
                                  trajectory_length]
    # print(len(trajectory))
    motions = np.zeros((frames, bvh.num_of_angles+num_of_root_infos))
    angles = init_angles
    state = init_state
    # print(angles.shape)
    # print(trajectory)
    # fake_trajectory = np.zeros((trajectory_length, num_of_root_infos))
    # fake_trajectory = np.array(
    #     [[0, 0, 0.1, 0, 0, 0]
    #      for i in range(trajectory_length)])

    fake_trajectory = np.concatenate(
        [np.array([0, 0, 0.1, 0, 0, 0])*(x+1)
         for x in range(trajectory_length)], axis=0).\
        reshape(trajectory_length, num_of_root_infos)
    # print(fake_trajectory)
    # fake_trajectory[:, 1] = 0.2*delta_scale

    left_trajectory = np.concatenate(
        [np.array([0.1, 0, 0, 0, 0, 0])*(x+1)
         for x in range(trajectory_length)], axis=0).\
        reshape(trajectory_length, num_of_root_infos)
    left_trajectory[:, 0] -= 50
    left_trajectory[:, 2] += 10
    fake_trajectory = left_trajectory

    for i in range(frames):
        print('i:  ', i)
        x = torch.cat(
            (torch.tensor(fake_trajectory, dtype=torch.float32)
             .view(1, num_of_root_infos*trajectory_length),
             torch.tensor(angles, dtype=torch.float32).view(1, 90)), dim=1)
        y = pfnn(x, phase)
        # print(y.shape)
        # trajectory = y[:, :trajectory_length*num_of_root_infos]
        # trajectory = trajectory.view((trajectory_length, num_of_root_infos))
        # fake_trajectory[:, 2] += 0.1
        fake_trajectory[:, 0] += 0.1
        phase += y[0, trajectory_length*num_of_root_infos]/phase_scale
        phase = phase.detach()
        print(phase)
        angles = y[:, trajectory_length*num_of_root_infos+1:]
        # print(y.shape)
        # print(angles.shape)
        # print(state.reshape(1, num_of_root_infos).shape)
        # print(angles.detach().numpy().shape)
        state = y[:, :num_of_root_infos].detach().numpy()
        motions[i] = np.concatenate(
            (fake_trajectory[0, :num_of_root_infos].reshape(
                1, num_of_root_infos), angles.detach().numpy()),
            axis=1)
    bvh.save("2_left"+output_path, motions)
    smoothed_motions = np.concatenate(
        (np.zeros((frames, num_of_root_infos)),
         motions[:, num_of_root_infos:]),
        axis=1)
    bvh.save("2_smooth_left"+output_path, smoothed_motions)


# trajectory_scale = 1
# phase_scale = 1
frames = 200


def pfnn_inference_3():
    dataset = BVHDataset3(base_dir + bvh_path)
    bvh = dataset.bvh
    print(dataset.in_features, dataset.out_features)
    pfnn = PFNN(dataset.in_features, dataset.out_features).float().cuda()
    pfnn.load_state_dict(torch.load('models_7/pfnn_params199.pkl'))  # 13
    x, phase, y = dataset[198]
    print(x.shape, phase.shape, y.shape)
    x = torch.tensor(x).float()
    all_angles = x[:, trajectory_length*num_of_trajectory_infos:]
    fake_trajectory = np.zeros((trajectory_length, num_of_trajectory_infos))
    fake_trajectory[:, 1] = 0.2
    fake_trajectory = (fake_trajectory -
                       dataset.trajectory_mean) / dataset.trajectory_std
    fake_trajectory *= trajectory_scale
    print(fake_trajectory)
    motions = np.zeros((frames, bvh.num_of_angles+num_of_root_infos))
    num_of_angles = bvh.num_of_angles
    last_angles = np.zeros((1, num_of_angles))
    for i in range(frames):
        print('i:  ', i)
        if i == 0:
            angles = all_angles[:, :num_of_angles].numpy()
            angles_delta = all_angles[:, num_of_angles:]
            last_angles = (angles / angles_scale) * \
                dataset.angles_std + dataset.angles_mean
        else:
            angles_delta = output_angles - last_angles
            angles_delta = (angles_delta - dataset.angles_delta_mean) / \
                (dataset.angles_delta_std+(dataset.angles_delta_std == 0))
            angles_delta *= angles_scale
            last_angles = output_angles
        x = torch.cat(
            (torch.tensor(fake_trajectory, dtype=torch.float32)
             .view(1, num_of_trajectory_infos*trajectory_length),
             torch.tensor(angles, dtype=torch.float32).view(1, 90),
             torch.tensor(angles_delta, dtype=torch.float32).view(1, 90)
             ),
            dim=1)
        y = pfnn(x.cuda(), phase)
        phase += (y[0, -1].detach().cpu().double().numpy() / phase_scale) * \
            dataset.phase_deltas_std + dataset.phase_deltas_mean
        # phase = phase.detach()
        print(phase)
        angles_index = trajectory_length*num_of_trajectory_infos
        angles = y[:, angles_index:angles_index+dataset.bvh.num_of_angles]
        # all_angles = y[:, angles_index:-1]
        # x = y[:, :-1]
        # print(y.shape)
        # print(angles.shape)
        # print(state.reshape(1, num_of_root_infos).shape)
        # print(angles.detach().numpy().shape)
        # state = y[:, :num_of_root_infos].detach().numpy()
        # motions[i] = np.concatenate(
        #     (fake_trajectory[0, :num_of_root_infos].reshape(
        #         1, num_of_root_infos), angles.detach().numpy()),
        #     axis=1)
        output_angles = (angles.detach().cpu().numpy() / angles_scale) * \
            dataset.angles_std + dataset.angles_mean
        motions[i] = np.concatenate((
            np.zeros((1, num_of_root_infos)), output_angles),
            axis=1)
    print(motions.shape)
    bvh.save("7_"+output_path, motions)
    # smoothed_motions = np.concatenate(
    #     (np.zeros((frames, num_of_root_infos)),
    #      motions[:, num_of_root_infos:]),
    #     axis=1)
    # bvh.save("2_smooth_left"+output_path, smoothed_motions)


if __name__ == '__main__':
    # dataset = BVHDataset3(base_dir + bvh_path)
    # pfnn = PFNN(dataset.in_features, dataset.out_features).float().cuda()
    # pfnn.load_state_dict(torch.load('models_7/pfnn_params34.pkl'))  # 13
    # print(pfnn.state_dict())

    # pfnn_inference()
    pfnn_inference_3()
