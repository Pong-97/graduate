import json
import math
import torch.nn as nn
from collections import OrderedDict
import torch
from thop import profile
import pdb
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
import copy
from tqdm import tqdm

MUTE_NUM = range(0,512,10)
    
norm_mean, norm_var = 0.0, 1.0

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]


class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, cfg=None, compress_rate=None):
        super(VGG, self).__init__()
        self.features = nn.Sequential()
        if cfg is None:
            cfg = defaultcfg
        if len(cfg)==17 :
            cfg.append(cfg[-1])
        if compress_rate is None:
            compress_rate = [0,0,0,0,0,0,0,0,0,0,0,0,0]

        self.relucfg = relucfg
        self.covcfg = convcfg
        self.compress_rate = compress_rate
        self.features = self.make_layers(cfg[:-1], True, compress_rate)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-1], 512)),
            ('norm1', nn.BatchNorm1d(512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(512, num_classes)),
        ]))


        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=True, compress_rate=None):
        layers = nn.Sequential()
        in_channels = 3
        cnt = 0
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                conv2d.cp_rate = compress_rate[cnt]
                cnt += 1

                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = v
        return layers

    def forward(self, x):
        # conv0输出
        x = self.features[0](x)
        # conv1输出
        x = self.features[1:4](x)
        # conv3输出
        x = self.features[4:8](x)
        # conv4输出
        x = self.features[8:11](x)
        # conv6输出
        x = self.features[11:15](x)
        # conv7输出
        x = self.features[15:18](x)
        arr = freq_cal(x)   
        start = int(arr.shape[0]*mute_idx/100)
        end = int(arr.shape[0]*(mute_idx+FREQ_GAP)/100)
        idx = np.argsort(arr)[start:end]
        if idx.shape[0] != 0:
            # x[:, idx, :, :] = x[:, idx, :, :] * 0
            x[:, start:end, :, :] = x[:, start:end, :, :] * 0
        # conv8输出
        x = self.features[18:21](x)
        # conv10输出
        x = self.features[21:25](x)
        # conv11输出
        x = self.features[25:28](x)
        # conv12输出
        x = self.features[28:31](x)
        # conv14输出
        x = self.features[31:35](x)
        # conv15输出
        x = self.features[35:38](x)
        # conv16输出
        x = self.features[38:41](x)
        x = self.features[41:](x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)


        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def lowPassCal(img,size=None):#传递参数为傅里叶变换后的频谱图和滤波尺寸
    h, w = img.shape[0:2]#获取图像属性
    if size == None:
        size = h/2
    sum_energy = 0
    low_freq_energy = 0
    size = min(min(h, size),w)
    h1,w1 = int(h/2), int(w/2)#找到傅里叶频谱图的中心点
    mask = np.zeros((h,w))
    mask[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
    img_low = img * mask#中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为0
    for ii in range(img.shape[0]):
        for jj in range(img.shape[1]):
            low_freq_energy += abs(img_low[ii][jj])
            sum_energy += abs(img[ii][jj])
    low_freq_rate = low_freq_energy / sum_energy
    return low_freq_rate


def freq_cal(feature):
    freq_rate_list = np.zeros([feature.shape[1]])
    for j in range(feature.shape[1]):
        for i in range(feature.shape[0]):
            gray = feature[i, j, :, :].detach().cpu().clone()
            # 傅里叶变换
            img_dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(img_dft)  # 将频域从左上角移动到中间
        
            # 计算低频分量
            low_freq_rate = lowPassCal(dft_shift)
        freq_rate_list[j] +=  low_freq_rate
    return freq_rate_list


def vgg_16_bn_dump(compress_rate=None, cfg=None):
    return VGG(compress_rate=compress_rate, cfg=cfg)


def test(model, test_loader, optimizer=None, file_path=None):
    model.eval()
    test_loss = 0
    correct = 0
    early_break = 0
    test_data_num = 0
    for data, target in tqdm(test_loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        # test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        # print("loss: ", F.cross_entropy(output, target, reduction='sum').item())
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        early_break += 1
        test_data_num += target.data.shape[0]
        if early_break == 2:
            break
    test_loss /= test_data_num

    if optimizer != None:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%), learning rate: {}'.format(
            test_loss, correct, test_data_num,
            100. * correct / test_data_num, optimizer.param_groups[0]['lr']))
    else:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
            test_loss, correct, test_data_num,
            100. * correct / test_data_num))            
    return test_loss, correct.item() / float(test_data_num)

def main():
    kwargs = {'num_workers': 16, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/home/max.peng/graduate/dataset', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ]),download=False),
            batch_size=1024, shuffle=False, **kwargs)
    teacher_model = vgg_16_bn_dump()
    teacher_model_path = '/home/max.peng/graduate/vgg_16_bn.pt.pt'
    print("=> loading teacher model checkpoint '{}'".format(teacher_model_path))
    checkpoint = torch.load(teacher_model_path, map_location='cpu')
    teacher_model.load_state_dict(checkpoint['state_dict'], strict=True)
    teacher_model.cuda()
    teacher_model = torch.nn.DataParallel(teacher_model)
    loss, acc = test(teacher_model, test_loader)


if __name__ == '__main__':
    FREQ_GAP = 25
    MUTE_IDX = range(0,100,FREQ_GAP)

    for mute_idx in MUTE_IDX:
        print('mute: {}%-{}%'.format(mute_idx, (mute_idx+FREQ_GAP)))
        main()