import os
import argparse
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from models.vgg_16_bn import vgg_16_bn
import json
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder
from torch.autograd import Variable

import pdb

def transfer_value(model,new_model,masks):
    layer_id_in_cfg = 0
    linear_idx = 0
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.Conv2d) or isinstance(m0, nn.Linear):
            print(m0)
            print(m1)
            if m1.weight.shape == m0.weight.shape:
                if isinstance(m0, nn.BatchNorm2d):
                    m1.weight.data = m0.weight.data.clone()
                    m1.bias.data = m0.bias.data.clone()
                    m1.running_mean = m0.running_mean.clone()
                    m1.running_var = m0.running_var.clone()
                if isinstance(m0, nn.Conv2d):
                    m1.weight.data = m0.weight.data.clone()
                if isinstance(m0, nn.Linear):
                    m1.weight.data = m0.weight.data.clone()
                    m1.bias.data = m0.bias.data.clone()
                continue

        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(masks.cpu().numpy())))   #np.asarray(end_mask.cpu().numpy()) == 1时的索引
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
        elif isinstance(m0, nn.Conv2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(masks.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data.clone()   #m.weight.data.size()   (out,in,size[0],size[1])
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(masks.cpu().numpy())))
            # pdb.set_trace()
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
            linear_idx += 1
    return new_model


def train(model, epoch):
    pdb.set_trace()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.10f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, optimizer=None):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        # test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    if optimizer != None:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%), learning rate: {}'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), optimizer.param_groups[0]['lr']))
    else:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))            
    return test_loss, correct.item() / float(len(test_loader.dataset))

if __name__ == '__main__':
    checkpoint = torch.load('/home2/pengyifan/pyf/hypergraph_cluster/log/pretrained_model/vgg_16_bn.pt.pt', map_location='cuda:0')
    model_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',512, 512, 512]
    compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    model = vgg_16_bn(compress_rate=compress_rate, cfg=model_cfg)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model = nn.DataParallel(model) 

    new_model_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',512, 512, 222]
    new_model = vgg_16_bn(compress_rate=compress_rate, cfg=new_model_cfg)
    new_model = new_model.cuda()
    new_model = nn.DataParallel(new_model) 

    # '/home2/pengyifan/pyf/hypergraph_cluster/log/graduate/vgg/max_90_idx.json'
    with open('/home2/pengyifan/pyf/hypergraph_cluster/log/graduate/vgg/mutemax_290_idx.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
    idx = list(map(int, json_data['290']))
    mask = torch.ones(512)
    for i in range(512):
        if i in idx:
            mask[i] = 0
    new_model = transfer_value(model, new_model, mask)

    epochs = 30
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs*0.5, epochs*0.75], gamma=0.1)
    kwargs = {'num_workers': 16, 'pin_memory': True} 
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("/repository/linhang/data/cifar-10/", train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("/repository/linhang/data/cifar-10/", train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=128, shuffle=True, **kwargs)

    for epoch in range(epochs):
        train(new_model, epoch)
        loss, acc = test(new_model, test_loader, optimizer=optimizer)