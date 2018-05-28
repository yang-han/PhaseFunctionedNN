import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np


class BaseNet(nn.Module):
    def __init__(self, in_features, out_features):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, out_features)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)
        return x


class PFNN(nn.Module):
    def __init__(self, in_features, out_features,
                 control_nets=4, hidden_units=512):
        nn.Module.__init__(self)
        # self.fc10 = nn.Linear(in_features, 512)
        # self.fc11 = nn.Linear(in_features, 512)
        # self.fc12 = nn.Linear(in_features, 512)
        # self.fc13 = nn.Linear(in_features, 512)
        # self.fc20 = nn.Linear(512, 512)
        # self.fc21 = nn.Linear(512, 512)
        # self.fc22 = nn.Linear(512, 512)
        # self.fc23 = nn.Linear(512, 512)
        # self.fc30 = nn.Linear(512, out_features)
        # self.fc31 = nn.Linear(512, out_features)
        # self.fc32 = nn.Linear(512, out_features)
        # self.fc33 = nn.Linear(512, out_features)
        # self.fc1s = [self.fc10, self.fc11, self.fc12, self.fc13]
        # self.fc2s = [self.fc20, self.fc21, self.fc22, self.fc23]
        # self.fc3s = [self.fc30, self.fc31, self.fc32, self.fc33]
        self.in_features = in_features
        self.out_features = out_features
        self.fc1s = nn.ModuleList(
            [nn.Linear(in_features, hidden_units) for i in range(4)])
        self.fc2s = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for i in range(4)])
        self.fc3s = nn.ModuleList(
            [nn.Linear(hidden_units, out_features) for i in range(4)])
        self.dropout = nn.Dropout(0.3)
        self.elu = nn.ELU()

    def forward(self, x, p):
        # p = p.detach()
        # phase = (4*p)/(2*np.pi)
        phase = 4*p
        # phase = phase.detach()
        w = phase % 1
        # print(phase)
        k = torch.tensor(np.floor(phase), dtype=torch.int32,
                         requires_grad=False)
        base_k = k - 1
        # output = torch.zeros((x.shape[0], self.out_features),
        #                      dtype=torch.float32)
        fc10 = self.fc1s[base_k % 4].cuda()
        fc11 = self.fc1s[(base_k+1) % 4].cuda()
        fc12 = self.fc1s[(base_k+2) % 4].cuda()
        fc13 = self.fc1s[(base_k+3) % 4].cuda()
        fc20 = self.fc2s[base_k % 4].cuda()
        fc21 = self.fc2s[(base_k+1) % 4].cuda()
        fc22 = self.fc2s[(base_k+2) % 4].cuda()
        fc23 = self.fc2s[(base_k+3) % 4].cuda()
        fc30 = self.fc3s[base_k % 4].cuda()
        fc31 = self.fc3s[(base_k+1) % 4].cuda()
        fc32 = self.fc3s[(base_k+2) % 4].cuda()
        fc33 = self.fc3s[(base_k+3) % 4].cuda()

        y1 = self.elu(self.cubic(fc10(x),
                                 fc11(x),
                                 fc12(x),
                                 fc13(x),
                                 w))
        y1 = self.dropout(y1)
        y2 = self.elu(self.cubic(fc20(y1),
                                 fc21(y1),
                                 fc22(y1),
                                 fc23(y1),
                                 w))
        y2 = self.dropout(y2)
        return self.cubic(fc30(y2),
                          fc31(y2),
                          fc32(y2),
                          fc33(y2),
                          w)

    def cubic(self, y0, y1, y2, y3, w):
        w = torch.tensor(w).float().cuda()
        return (
            (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*w*w*w +
            (y0-2.5*y1+2.0*y2-0.5*y3)*w*w +
            (-0.5*y0+0.5*y2)*w +
            (y1))

    # def compute_params(self, p):
    #     nets
    #     phase = (4*p)/2*math.pi
    #     w = phase % 1
    #     k = math.floor(phase)
    #     base_k = k - 1
    #     a0 = nets[(base_k) % 4].state_dict()
    #     a1 = nets[(base_k+1) % 4].state_dict()
    #     a2 = nets[(base_k+2) % 4].state_dict()
    #     a3 = nets[(base_k+3) % 4].state_dict()
    #     for layer in net.state_dict():
    #         weight = a1[layer] + \
    #             w * (0.5*a2[layer] - 0.5*a0[layer]) + \
    #             w*w*(a0[layer] - 2.5*a1[layer] + 2*a2[layer]-0.5*a3[layer])+\
    #             math.pow(w, 3)*(1.5*a1[layer]-1.5 *
    #                             a2[layer] + 0.5*a3[layer] - 0.5*a0[layer])
    #         self.state_dict()[layer] = (torch.ones_like(
    #             weight, requires_grad=True))


class PFNN2(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        super(PFNN2, self).__init__()
        self.w1s = nn.Parameter(torch.zeros(
            (4, in_features, hidden_units), requires_grad=True))
        self.b1s = nn.Parameter(torch.zeros(
            (4, 1, hidden_units), requires_grad=True))
        self.w2s = nn.Parameter(torch.zeros(
            (4, hidden_units, hidden_units), requires_grad=True))
        self.b2s = nn.Parameter(torch.zeros(
            (4, 1, hidden_units), requires_grad=True))
        self.w3s = nn.Parameter(torch.zeros(
            (4, hidden_units, out_features), requires_grad=True))
        self.b3s = nn.Parameter(torch.zeros(
            (4, 1, out_features), requires_grad=True))

    def forward(self, x, p):
        phase = 4*p
        # phase = phase.detach()
        w = phase % 1
        # print(phase)
        # k = torch.tensor(np.floor(phase), dtype=torch.int32,
        #                  requires_grad=False)
        k = np.floor(phase).astype(np.int)
        base_k = k - 1
        # base_k = base_k.tolist()
        print(base_k)
        print(base_k.shape)
        print(self.w1s.shape)
        print(self.w1s[(base_k % 4).astype(np.int)].shape)
        # print(x.shape)
        # print((torch.mm(x, self.w1s[base_k % 4])).shape)
        return (torch.mm(x, self.w1s[(base_k % 4).astype(np.int)])) +\
            self.b1s[(base_k % 4).astype(np.int)]

    def cubic(self, y0, y1, y2, y3, w):
        return (
            (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*w*w*w +
            (y0-2.5*y1+2.0*y2-0.5*y3)*w*w +
            (-0.5*y0+0.5*y2)*w +
            (y1))


if __name__ == "__main__":
    # net = BaseNet(96, 96).cuda()
    # print(net)
    # pfnn = PFNN(93, 93).cuda()
    # print(pfnn)
    x = torch.ones((4, 93)).cuda()
    x2 = [torch.ones((1, 93)).cuda() for x in range(4)]
    # pred = pfnn(x, 0.1)
    # print(pred)
    # print(pred.shape)
    pfnn = PFNN2(93, 93, 512).cuda()

    pred = pfnn(x, np.array([0.1, 0.1, 0.1, 0.1]))
    # pred = pfnn(x2, 0.1)
    print(pred)
    print(pred.shape)
    print(pfnn)
    print(pfnn.state_dict().keys())
    # print(list(pfnn.parameters()))
    # print(pfnn.L2())
    # print(list(pfnn.parameters()))
    # print(pfnn.state_dict())
    # nets = [BaseNet(96, 96) for i in range(4)]
    # # print(net.state_dict())
    # print(nets)
    # pfnn = PFNN(nets)
    # pfnn.compute_params(0)
    # print(pfnn)
    # print(pfnn.state_dict())
