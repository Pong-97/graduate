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
        # conv0??????
        x = self.features[0](x)
        # conv1??????
        x = self.features[1:4](x)
        # conv3??????
        x = self.features[4:8](x)
        # conv4??????
        x = self.features[8:11](x)
        # conv6??????
        x = self.features[11:15](x)
        # conv7??????
        x = self.features[15:18](x)
        # conv8??????
        x = self.features[18:21](x)
        # conv10??????
        x = self.features[21:25](x)
        # conv11??????
        x = self.features[25:28](x)
        # conv12??????
        x = self.features[28:31](x)
        # x = freq_filtering(x)
        # conv14??????
        x = self.features[31:35](x)
        # conv15??????
        x = self.features[35:38](x)
        # conv16??????
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


def highPassFiltering(img,size):#????????????????????????????????????????????????????????????
    h, w = img.shape[0:2]#??????????????????
    size = min(min(h, size),w)
    h1,w1 = int(h/2), int(w/2)#????????????????????????????????????
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0#??????????????????????????????????????????????????????????????????????????????????????????????????????0
    return img

def lowPassFiltering(img,size):#????????????????????????????????????????????????????????????
    h, w = img.shape[0:2]#??????????????????
    size = min(min(h, size),w)
    h1,w1 = int(h/2), int(w/2)#????????????????????????????????????
    mask = np.zeros((h,w))
    mask[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
    img = img * mask#??????????????????????????????????????????????????????????????????????????????????????????????????????0
    return img


def freq_filtering(feature):
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            gray = feature[i, j, :, :].detach().cpu().clone()
            # ???????????????
            img_dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(img_dft)  # ????????????????????????????????????
        
            dft_shift=lowPassFiltering(dft_shift, SCALE)
            # res = np.log(np.abs(dft_shift))
            # ??????????????????
            idft_shift = np.fft.ifftshift(dft_shift)  #????????????????????????????????????
            ifimg = np.fft.ifft2(idft_shift)  # ????????????????????????
            ifimg = np.abs(ifimg)
            sign = np.array(copy.deepcopy(gray))
            sign[sign<0]=-1
            sign[sign>0]=1
            ifimg = ifimg * sign
            feature[i, j, :, :] = torch.from_numpy(ifimg).cuda()
    return feature


def vgg_16_bn_dump(compress_rate=None, cfg=None):
    return VGG(compress_rate=compress_rate, cfg=cfg)


def test(model, test_loader, optimizer=None, file_path=None):
    model.eval()
    test_loss = 0
    test_loss_mute = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        # test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        print("loss: ", F.cross_entropy(output, target, reduction='sum').item())
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_loss_mute /= len(test_loader.dataset)

    if optimizer != None:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%), learning rate: {}'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), optimizer.param_groups[0]['lr']))
    else:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))            
    return test_loss, correct.item() / float(len(test_loader.dataset))

def main():
    kwargs = {'num_workers': 16, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/repository/linhang/data/cifar-10/', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=1024, shuffle=False, **kwargs)
    teacher_model = vgg_16_bn_dump()
    teacher_model_path = '/home2/pengyifan/pyf/hypergraph_cluster/log/pretrained_model/vgg_16_bn.pt.pt'
    print("=> loading teacher model checkpoint '{}'".format(teacher_model_path))
    checkpoint = torch.load(teacher_model_path, map_location='cpu')
    teacher_model.load_state_dict(checkpoint['state_dict'], strict=True)
    teacher_model.cuda()
    teacher_model = torch.nn.DataParallel(teacher_model)
    loss, acc = test(teacher_model, test_loader)


if __name__ == '__main__':
    SCALE = 100
    main()