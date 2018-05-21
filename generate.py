import torch
import numpy as np
from PFNN import BaseNet
from train import BVH
from hyperparams import *

frames = 1000


if __name__ == '__main__':
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
