import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from thop import profile

norm_mean, norm_var = 0.0, 1.0
defaultresnet56cfg =[16] + [16]*18 + [32]*18 + [64]*18
defaultresnet110cfg = [16] + [16]*36 + [32]*36 + [64]*36

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, compress_rate=[0.], first_stage=0):
        super(ResBasicBlock, self).__init__()
        # pdb.set_trace()
        # print(compress_rate)
        self.inplanes = inplanes
        self.planes = planes
        # self.conv1.cp_rate = compress_rate[0]
        keep_rate0 = 1-compress_rate[0]
        self.conv1 = conv3x3(inplanes, int(planes*keep_rate0), stride)
        self.bn1 = nn.BatchNorm2d(int(planes*keep_rate0))
        self.relu1 = nn.ReLU(inplace=True)

        # self.conv2.cp_rate = compress_rate[1]
        keep_rate1 = 1-compress_rate[1]
        self.conv2 = conv3x3(int(planes*keep_rate0), int(planes*keep_rate1))
        self.bn2 = nn.BatchNorm2d(int(planes*keep_rate1))
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or first_stage:
            if inplanes <= int(planes*keep_rate1):
                gap = int(planes*keep_rate1) - inplanes
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                    )
            elif inplanes >= int(planes*keep_rate1):
                gap_scale = inplanes // int(planes*keep_rate1)
                # gap = inplanes - gap_scale*int(planes*keep_rate1)
                after_slice = int(np.ceil(inplanes / (gap_scale+1)))
                gap = int(planes*keep_rate1) - after_slice
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, ::(gap_scale+1), ::2, ::2], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                    )            
        elif inplanes != int(planes*keep_rate1):
            if inplanes <= int(planes*keep_rate1):
                gap = int(planes*keep_rate1) - inplanes
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                    )
            elif inplanes >= int(planes*keep_rate1):
                # pdb.set_trace()
                gap_scale = inplanes // int(planes*keep_rate1)
                # gap = inplanes - gap_scale*int(planes*keep_rate1)
                after_slice = int(np.ceil(inplanes / (gap_scale+1)))
                gap = int(planes*keep_rate1) - after_slice
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, ::(gap_scale+1), :, :], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                    )              

    def forward(self, x):
        # pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, covcfg, cfg, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        self.covcfg = covcfg
        if cfg ==None:
            if num_layers == 56:
                cfg =defaultresnet56cfg
            if num_layers == 110:
                cfg =defaultresnet110cfg
        if num_layers == 56:
            compress_rate = [0.0]*55
            for i in range(len(compress_rate)):
                compress_rate[i] = (defaultresnet56cfg[i]-cfg[i])/defaultresnet56cfg[i]
        if num_layers == 110:
            compress_rate = [0.0]*109      
            for i in range(len(compress_rate)):
                compress_rate[i] = (defaultresnet110cfg[i]-cfg[i])/defaultresnet110cfg[i]   

        
        self.compress_rate = compress_rate
        self.num_layers = num_layers
        # pdb.set_trace()
        self.inplanes = 16
        keep_rate = 1-compress_rate[0]
        self.conv1 = nn.Conv2d(3, int(self.inplanes*keep_rate), kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes =  int(self.inplanes*keep_rate)
        self.conv1.cp_rate = keep_rate

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1,
                                       compress_rate=compress_rate[1:2 * n + 1], first_stage=0)
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2,
                                       compress_rate=compress_rate[2 * n + 1:4 * n + 1])
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2,
                                       compress_rate=compress_rate[4 * n + 1:6 * n + 1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if num_layers == 110:
            self.linear = nn.Linear(int(64 * block.expansion * (1-compress_rate[-1])), num_classes)
        else:
            self.fc = nn.Linear(int(64 * block.expansion * (1-compress_rate[-1])), num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, compress_rate, first_stage=0):
        # pdb.set_trace()
        layers = []
        layers.append(block(self.inplanes, planes, stride, compress_rate=compress_rate[0:2], first_stage=first_stage))
        # pdb.set_trace()
        self.inplanes = int(planes * block.expansion * (1-compress_rate[1]))
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, compress_rate=compress_rate[2 * i:2 * i + 2]))
            self.inplanes = int(planes * block.expansion * (1-compress_rate[2*i+1]))
        # print(self.inplanes)
        # pdb.set_trace()

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layers == 110:
            x = self.linear(x)
        else:
            x = self.fc(x)

        return x


def resnet_56(cfg=None):
    cov_cfg = [(3 * i + 2) for i in range(9 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 56, cov_cfg, cfg=cfg, num_classes=100)


def resnet_110(cfg=None):
    cov_cfg = [(3 * i + 2) for i in range(18 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 110, cov_cfg, cfg=cfg)

# resnet56#######################################################################################################################################################################################################
def main():
    # model = resnet_56(compress_rate=[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4])
    # [16, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 32, 64, 25, 25, 25, 25, 25, 25, 38, 38, 38, 57, 38, 57, 38, 57, 38, 57, 38]
    cfg = [16] + [10]*18 + [22]*18 + [48]*18
# FLOPS:  50521016.0 
# PARAMS: 335134.0
    # cfg = [12] + [12]*18 + [23]*9 + [25]*9 + [45]*9 + [50]*9
# FLOPS:  71047398.0 
# PARAMS: 474114.0
    # cfg = [16] + [10]*18 + [20]*18 + [30]*9 + [40]*9
# FLOPS:  46938296.0 
# PARAMS: 279154.0
    model = resnet_56(cfg=cfg)
    print(model)
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
    print("FLOPS: ", flops, "\nPARAMS:", params)
    # print(model)
    # inputs = torch.rand((1, 3, 32, 32)).cuda()
    # model = model.cuda().train()
    # output = model(inputs)
    # print(output.shape)
    net_ori = resnet_56()
    flops_ori, params_ori = profile(net_ori, inputs=(input, ) )  #  profile（模型，输入数据）
    print("FLOPS ori: ", flops_ori, "\nPARAMS_ori:", params_ori)
    print("FLOPS: ", flops, "\nPARAMS:", params)
    print("FLOPS compress rate: ", 1-flops/flops_ori, "\nPARAMS compress rate:", 1-params/params_ori)
if __name__ == '__main__':
    main()
# resnet56#######################################################################################################################################################################################################


# # resnet110#######################################################################################################################################################################################################
# def main():
#     # cfg = [16] + [16]*36 + [32]*36 + [64]*36
#     # model = resnet_110(cfg=cfg)
#     # checkpoint = torch.load('/home2/pengyifan/pyf/hypergraph_cluster/log/pretrained_model/cifar_resnet56/model_best.pth.tar', map_location='cuda:0')
#     # model = model.cuda()
#     # model = nn.DataParallel(model)
#     # model.load_state_dict(checkpoint['state_dict'],strict=False)

#     # checkpoint = torch.load('/home2/pengyifan/pyf/hypergraph_cluster/log/resnet110/0.4_0.5_0.6/finetuned/42/001/model_best.pth.tar', map_location='cuda:0')
#     # prec = checkpoint['best_prec1']
#     # print(prec)
#     # cfg = checkpoint['cfg']
#     # print(cfg)
#     # model = resnet_56(cfg=cfg)
#     model = resnet_110(cfg=[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 64, 64, 64, 64, 64, 64, 64, 64])
#     input = torch.randn(1, 3, 32, 32)
#     flops, params = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
#     print("FLOPS: ", flops, "\nPARAMS:", params)
#     # print(prec)
#     # print(model)
#     # inputs = torch.rand((1, 3, 32, 32)).cuda()
#     # model = model.cuda().train()
#     # output = model(inputs)
#     # print(output.shape)
# if __name__ == '__main__':
#     main()
# # resnet110#######################################################################################################################################################################################################