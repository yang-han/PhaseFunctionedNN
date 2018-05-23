from PFNN import BaseNet, PFNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter
from hyperparams import *
from BVH import BVH
from SeqDataset import SeqBVHDataset
from Dataset import BVHDataset


def train_base_net():
    writer = SummaryWriter()
    bvh = BVH()
    bvh.load(bvh_path)
    print(bvh.motions[:, 6:].shape)
    bvh_dataset = SeqBVHDataset(bvh, num_of_frames)
    dataloader = DataLoader(bvh_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    # train()
    net = BaseNet(num_of_frames*bvh.num_of_angles,
                  bvh.num_of_angles).cuda()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    # net.load_state_dict(torch.load('models/params199.pkl'))
    for epoch in range(30):
        running_loss = 0
        total_loss = 0
        for i, samples in enumerate(dataloader, 0):
            x, y = samples
            x = x.cuda()
            y = y.cuda()
            pred = net(x)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            total_loss = loss.item()

            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            writer.add_scalar('data/loss', loss, epoch)

        torch.save(net.state_dict(),
                   'models/singleF_params{}.pkl'.format(epoch))
        # net.load_state_dict(torch.load('params.pkl'))

    writer.export_scalars_to_json("./test.json")
    writer.close()


def train_pfnn():
    writer = SummaryWriter()
    dataset = BVHDataset(base_dir + bvh_path)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    net = PFNN(dataset.in_features, dataset.out_features).float().cuda()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    net.load_state_dict(torch.load('models/pfnn_params29.pkl'))
    for epoch in range(30, 100):
        running_loss = 0
        total_loss = 0
        for i, samples in enumerate(dataloader, 0):
            x, p, y = samples
            x = x.float().cuda()
            y = y.float().cuda()
            p = p.float().cuda()
            preds = torch.zeros_like(y, dtype=torch.float32)
            for _i, _p in enumerate(p):
                preds[_i] = net(x[_i].cuda(), _p.cuda())
                # print(preds[_i])
            optimizer.zero_grad()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_loss = loss.item()
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            writer.add_scalar('data/loss', loss, epoch)
            torch.save(net.state_dict(),
                       'models/pfnn_params{}.pkl'.format(epoch))

    writer.export_scalars_to_json("./test.json")
    writer.close()


if __name__ == "__main__":
    # train_base_net()
    train_pfnn()
