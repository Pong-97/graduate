import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import pdb
from thop import profile



__all__ = ['ResNet', 'resnet34', 'resnet50', 'resnet101']


model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        cfg,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
        remain_rate = None,
        first_block=False,
        first_stage=False,

    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if remain_rate and not first_block:
            self.conv1 = conv1x1(int(inplanes*remain_rate), cfg[0])
            self.bn1 = norm_layer(cfg[0])
            self.conv2 = conv3x3(cfg[0], cfg[1], stride, groups, dilation)
            self.bn2 = norm_layer(cfg[1])
            self.conv3 = conv1x1(cfg[1], int(planes * self.expansion*remain_rate))
            self.bn3 = norm_layer(int(planes * self.expansion*remain_rate))
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        elif remain_rate and first_block and first_stage:
            self.conv1 = conv1x1(inplanes, cfg[0])
            self.bn1 = norm_layer(cfg[0])
            self.conv2 = conv3x3(cfg[0], cfg[1], stride, groups, dilation)
            self.bn2 = norm_layer(cfg[1])
            self.conv3 = conv1x1(cfg[1], int(planes * self.expansion*remain_rate))
            self.bn3 = norm_layer(int(planes * self.expansion*remain_rate))
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        elif remain_rate and first_block:
            self.conv1 = conv1x1(int(inplanes*remain_rate), cfg[0])
            self.bn1 = norm_layer(cfg[0])
            self.conv2 = conv3x3(cfg[0], cfg[1], stride, groups, dilation)
            self.bn2 = norm_layer(cfg[1])
            self.conv3 = conv1x1(cfg[1], int(planes * self.expansion*remain_rate))
            self.bn3 = norm_layer(int(planes * self.expansion*remain_rate))
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers = [3, 4, 6, 3],
        num_classes = 1000,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None, 
        cfg=None,
        remain_rate=0.5,
    ):
        super(ResNet, self).__init__()

        if cfg is None:
            # Construct config variable.
            cfg = [[64, 64, 64], [256, 64, 64]*2, [256, 128, 128], [512, 128, 128]*3, [512, 256, 256], [1024, 256, 256]*5, [1024, 512, 512], [2048, 512, 512]*2]
            cfg = [item for sub_list in cfg for item in sub_list]   
           

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.remain_rate = remain_rate

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[ : 3*layers[0]], remain_rate=self.remain_rate ,first_stage=True)
        self.layer2 = self._make_layer(block, 128, layers[1], cfg=cfg[3*layers[0] : (3*layers[0]+3*layers[1])], stride=2, dilate=replace_stride_with_dilation[0], remain_rate=self.remain_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg=cfg[(3*layers[0]+3*layers[1]) : (3*layers[0]+3*layers[1]+3*layers[2])], stride=2, dilate=replace_stride_with_dilation[1], remain_rate=self.remain_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[(3*layers[0]+3*layers[1]+3*layers[2]) : ], stride=2, dilate=replace_stride_with_dilation[2], remain_rate=self.remain_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * block.expansion * self.remain_rate), 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, cfg, remain_rate, first_stage = False,
                    stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if (stride != 1 or self.inplanes != planes * block.expansion) and first_stage:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, int(planes * block.expansion * remain_rate), stride),
                norm_layer(int(planes * block.expansion * remain_rate)),
            )
        else:
            downsample = nn.Sequential(
                conv1x1(int(self.inplanes * remain_rate), int(planes * block.expansion * remain_rate), stride),
                norm_layer(int(planes * block.expansion * remain_rate)),
            )            

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample, remain_rate=remain_rate, first_block=True, first_stage = first_stage,
                            groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = int(planes * block.expansion)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)], remain_rate=remain_rate, first_stage = first_stage,
                                groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(
    arch,
    block,
    layers,
    pretrained,
    progress,
    cfg=None,
    remain_rate=0.5,
    **kwargs):
    model = ResNet(block, layers, cfg=cfg, remain_rate=remain_rate, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet34(pretrained = False, progress = True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained = False, progress = True, cfg=None, remain_rate=0.5, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print("model is ResNet50")
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, cfg, remain_rate,
                   **kwargs)


def resnet101(pretrained = False, progress = True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


class CifarBasicBlock(nn.Module):
    expansion = 1
    # pdb.set_trace()
    def __init__(self, in_planes, planes, cfg, remain_rate, first_stage=False, first_block=0, stride=1, option='A'):
        super(CifarBasicBlock, self).__init__()
        # pdb.set_trace()
        # if first_stage and first_block :
        #     self.conv1 = nn.Conv2d(int(in_planes), cfg[0], kernel_size=3, stride=stride, padding=1, bias=False)
        #     self.bn1 = nn.BatchNorm2d(cfg[0])
        #     self.conv2 = nn.Conv2d(cfg[0], int(planes*remain_rate), kernel_size=3, stride=1, padding=1, bias=False)
        #     self.bn2 = nn.BatchNorm2d(int(planes*remain_rate))
        # else:
        self.conv1 = nn.Conv2d(int(in_planes*remain_rate), cfg[0], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], int(planes*remain_rate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(planes*remain_rate))            

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # pdb.set_trace()
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, int(np.ceil((planes*remain_rate)//4)), int(np.floor((planes*remain_rate)//4))), "constant", 0)) #(batchsize, channel, height, width)
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, int(planes*remain_rate)//4, int(planes*remain_rate)//4), "constant", 0)) #(batchsize, channel, height, width)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(int(in_planes*remain_rate), int(planes*remain_rate), kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(int(planes*remain_rate))
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # pdb.set_trace()
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=[], remain_rate=0.5):
        super(CifarResNet, self).__init__()
        self.in_planes = 16
        self.remain_rate = remain_rate
        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16]*9, [32, 32]*9, [64, 64]*9]
            cfg = [item for sub_list in cfg for item in sub_list]   
        # pdb.set_trace()

        self.conv1 = nn.Conv2d(3, int(16*remain_rate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(16*remain_rate))
        self.cfg1 = cfg[ : 2*num_blocks[0]]
        self.cfg2 = cfg[2*num_blocks[0] : 2*num_blocks[0]+2*num_blocks[1]]
        self.cfg3 = cfg[2*num_blocks[0]+2*num_blocks[1] : 2*num_blocks[0]+2*num_blocks[1]+2*num_blocks[2]]

        self.layer1 = self._make_cifar_layer(block, 16, num_blocks[0], cfg=self.cfg1, stride=1, remain_rate=self.remain_rate, first_stage=True)
        self.layer2 = self._make_cifar_layer(block, 32, num_blocks[1], cfg=self.cfg2, stride=2, remain_rate=self.remain_rate)
        self.layer3 = self._make_cifar_layer(block, 64, num_blocks[2], cfg=self.cfg3, stride=2, remain_rate=self.remain_rate)
        self.linear = nn.Linear(int(64*self.remain_rate), num_classes)

        self.apply(_weights_init)

    def _make_cifar_layer(self, block, planes, num_blocks, stride, remain_rate, cfg, first_stage=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i=0
        first_block = 1
        for stride in strides:
            layers.append(block(self.in_planes, planes, cfg[2*i: 2*(i+1)], remain_rate, first_stage, first_block, stride))
            first_block = 0
            self.in_planes = planes * block.expansion
            i = i+1

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cifar_resnet56(cfg = None, remain_rate=1, **kwargs):
    print("model is resnet56")
    return CifarResNet(CifarBasicBlock, [9, 9, 9], cfg=cfg, remain_rate=remain_rate)

def main():
    # net = resnet50(cfg=[28, 26, 51,  25, 16, 51,  24, 29, 51,  44, 36,  102, 11, 19, 102, 38, 42,  102, 45,  42,  102, 66,  61,  204, 40,  68,  204, 73,  57,  204, 67,  62,  204, 68,  49,  204, 62,  54,  204, 97,  107, 409,  109, 85,  409,  102, 90,  409])
    # net = resnet50(cfg=[23, 32, 128, 30, 44, 128, 29, 29, 128, 87, 103, 256, 37, 58, 256, 87, 107, 256, 119, 112, 256, 153, 177, 512, 140, 152, 512, 127, 149, 512, 109, 125, 512, 119, 130, 512, 134, 110, 512, 172, 184, 1024, 213, 231, 1024, 215, 238, 1024])
    net = cifar_resnet56(cfg = None, remain_rate=1)
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, inputs=(input, ) )  #  profile（模型，输入数据）
    print("FLOPS: ", flops, "\nPARAMS:", params)
    return
    print(net)
    inputs = torch.rand((1, 3, 32, 32)).cuda()
    # inputs = torch.rand((1, 3, 224, 224)).cuda()
    model = net.cuda().train()
    output = model(inputs)
    print(output.shape)
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
    print("FLOPS: ", flops, "\nPARAMS:", params)

if __name__ == '__main__':
    main()



