import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from thop import profile

norm_mean, norm_var = 0.0, 1.0
# cfg = [[64], [64, 64, 64], [256, 64, 64]*2, [256, 128, 128], [512, 128, 128]*3, [512, 256, 256], [1024, 256, 256]*5, [1024, 512, 512], [2048, 512, 512]*2]
# defaultresnet50cfg = [item for sub_list in cfg for item in sub_list] 
defaultresnet50cfg = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]  

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, compress_rate=[0.], tmp_name=None, not_first_block=0):
        super(ResBottleneck, self).__init__()

        keep_rate1 = 1-compress_rate[0]
        self.conv1 = nn.Conv2d(inplanes, int(planes*keep_rate1), kernel_size=1, bias=False)
        self.conv1.compress_rate = compress_rate[0]
        self.conv1.tmp_name = tmp_name
        self.bn1 = nn.BatchNorm2d(int(planes*keep_rate1))
        self.relu1 = nn.ReLU(inplace=True)

        keep_rate2 = 1-compress_rate[1]
        self.conv2 = nn.Conv2d(int(planes*keep_rate1), int(planes*keep_rate2), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(planes*keep_rate2))
        self.conv2.compress_rate = compress_rate[1]
        self.conv2.tmp_name = tmp_name
        self.relu2 = nn.ReLU(inplace=True)

        keep_rate3 = 1-compress_rate[2]
        self.conv3 = nn.Conv2d(int(planes*keep_rate2), int(planes*keep_rate3 * self.expansion), kernel_size=1, bias=False)
        self.conv3.compress_rate = compress_rate[2]
        self.conv3.tmp_name = tmp_name
        self.bn3 = nn.BatchNorm2d(int(planes*keep_rate3 * self.expansion))
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        if not_first_block == 1:
            self.downsample = nn.Sequential()
            if inplanes != int(planes*keep_rate3 * self.expansion):
                if inplanes < int(planes*keep_rate3 * self.expansion):
                    gap = int(planes*keep_rate3 * self.expansion) - inplanes
                    self.downsample = LambdaLayer(
                        lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                        )

                elif inplanes > int(planes*keep_rate3 * self.expansion):
                    gap_scale = inplanes // int(planes*keep_rate3 * self.expansion)
                    after_slice = int(np.ceil(inplanes / (gap_scale+1)))
                    gap = int(planes*keep_rate3 * self.expansion) - after_slice
                    self.downsample = LambdaLayer(
                        lambda x: F.pad(x[:, ::(gap_scale+1), :, :], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                        )                      

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, num_blocks, covcfg, cfg, num_classes=1000):
        super(ResNet, self).__init__()

        self.covcfg = covcfg
        if cfg ==None:
            if num_layers == 50:
                cfg =defaultresnet50cfg
        if num_layers == 50:
            compress_rate = [0.0]*49
            for i in range(len(compress_rate)):
                compress_rate[i] = (defaultresnet50cfg[i]-cfg[i])/defaultresnet50cfg[i]

        self.compress_rate = compress_rate
        self.num_layers = num_layers
        # pdb.set_trace()
        self.inplanes = 64
        keep_rate1 = 1-compress_rate[0]
        self.conv1 = nn.Conv2d(3, int(self.inplanes*keep_rate1), kernel_size=7, stride=2, padding=3, bias=False)
        self.inplanes =  int(self.inplanes*keep_rate1)
        self.conv1.compress_rate = keep_rate1
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, compress_rate=compress_rate[1 : 3*num_blocks[0]+1], tmp_name='layer1')
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, compress_rate=compress_rate[3*num_blocks[0]+1 : 3*num_blocks[0]+3*num_blocks[1]+1], tmp_name='layer2')
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, compress_rate=compress_rate[3*num_blocks[0]+3*num_blocks[1]+1:3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+1], tmp_name='layer3')
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, compress_rate=compress_rate[3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+1:], tmp_name='layer4')
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(int(512 * block.expansion * (1-compress_rate[-1])), num_classes)

        self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride, compress_rate, tmp_name):
        downsample = None
        keep_rate_shortcut = 1-compress_rate[2]
        # pdb.set_trace()
        if stride != 1 or self.inplanes != int(planes * block.expansion * keep_rate_shortcut):
            conv_short = nn.Conv2d(self.inplanes, int(planes * block.expansion * keep_rate_shortcut),
                                   kernel_size=1, stride=stride, bias=False)
            conv_short.compress_rate = compress_rate[2]
            conv_short.tmp_name = tmp_name + '_shortcut'
            downsample = nn.Sequential(
                conv_short,
                nn.BatchNorm2d(int(planes * block.expansion * keep_rate_shortcut)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, compress_rate=compress_rate[0:3],
                            tmp_name=tmp_name + '_block' + str(1)))
        keep_rate_block1 = 1 - compress_rate[2]
        self.inplanes = int(planes * block.expansion * keep_rate_block1)
        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes, compress_rate=compress_rate[3*i : 3*i+3],
                                tmp_name=tmp_name + '_block' + str(i + 1), not_first_block=1))
            keep_rate_blockn = 1 - compress_rate[3*i+2]
            self.inplanes = int(planes * block.expansion * keep_rate_blockn)
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # 256 x 56 x 56
        x = self.layer2(x)

        # 512 x 28 x 28
        x = self.layer3(x)

        # 1024 x 14 x 14
        x = self.layer4(x)

        # 2048 x 7 x 7
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_50(cfg=None):
    cov_cfg = [(3*i + 3) for i in range(3*3 + 1 + 4*3 + 1 + 6*3 + 1 + 3*3 + 1 + 1)]
    model = ResNet(ResBottleneck, 50, [3, 4, 6, 3], covcfg=cov_cfg, cfg=cfg)
    return model



shortcut_conv = [18, 54, 101, 170]
shortcut_bn = [19, 55, 102, 171]
# 0.2 [51, 51, 51, 204, 51, 51, 204, 51, 51, 204, 102, 102, 409, 102, 102, 409, 102, 102, 409, 102, 102, 409, 204, 204, 819, 204, 204, 819, 204, 204, 819, 204, 204, 819, 204, 204, 819, 204, 204, 819, 409, 409, 1638, 409, 409, 1638, 409, 409, 1638]
FLOPS:  2639408275.0 
PARAMS: 16641302.0

# 0.25 [48, 48, 48, 192, 48, 48, 192, 48, 48, 192, 96, 96, 384, 96, 96, 384, 96, 96, 384, 96, 96, 384, 192, 192, 768, 192, 192, 768, 192, 192, 768, 192, 192, 768, 192, 192, 768, 192, 192, 768, 384, 384, 1536, 384, 384, 1536, 384, 384, 1536]
FLOPS:  2339350272.0 
PARAMS: 14771992.0

# 0.3 [44, 44, 44, 179, 44, 44, 179, 44, 44, 179, 89, 89, 358, 89, 89, 358, 89, 89, 358, 89, 89, 358, 179, 179, 716, 179, 179, 716, 179, 179, 716, 179, 179, 716, 179, 179, 716, 179, 179, 716, 358, 358, 1433, 358, 358, 1433, 358, 358, 1433]
FLOPS:  2026557099.0 
PARAMS: 12935549.0

# 0.35 [41, 41, 41, 166, 41, 41, 166, 41, 41, 166, 83, 83, 332, 83, 83, 332, 83, 83, 332, 83, 83, 332, 166, 166, 665, 166, 166, 665, 166, 166, 665, 166, 166, 665, 166, 166, 665, 166, 166, 665, 332, 332, 1331, 332, 332, 1331, 332, 332, 1331]
FLOPS:  1755933690.0 
PARAMS: 11239802.0

# 0.365 [40, 40, 40, 162, 40, 40, 162, 40, 40, 162, 81, 81, 325, 81, 81, 325, 81, 81, 325, 81, 81, 325, 162, 162, 650, 162, 162, 650, 162, 162, 650, 162, 162, 650, 162, 162, 650, 162, 162, 650, 325, 325, 1300, 325, 325, 1300, 325, 325, 1300]
FLOPS:  1677148781.0 
PARAMS: 10770494.0

# 0.375 [40, 40, 40, 160, 40, 40, 160, 40, 40, 160, 80, 80, 320, 80, 80, 320, 80, 80, 320, 80, 80, 320, 160, 160, 640, 160, 160, 640, 160, 160, 640, 160, 160, 640, 160, 160, 640, 160, 160, 640, 320, 320, 1280, 320, 320, 1280, 320, 320, 1280]
FLOPS:  1639370880.0 
PARAMS: 10478480.0

# 0.4 [38, 38, 38, 153, 38, 38, 153, 38, 38, 153, 76, 76, 307, 76, 76, 307, 76, 76, 307, 76, 76, 307, 153, 153, 614, 153, 153, 614, 153, 153, 614, 153, 153, 614, 153, 153, 614, 153, 153, 614, 307, 307, 1228, 307, 307, 1228, 307, 307, 1228]
FLOPS:  1500143001.0 
PARAMS: 9676689.0

# 0.425 [36, 36, 36, 147, 36, 36, 147, 36, 36, 147, 73, 73, 294, 73, 73, 294, 73, 73, 294, 73, 73, 294, 147, 147, 588, 147, 147, 588, 147, 147, 588, 147, 147, 588, 147, 147, 588, 147, 147, 588, 294, 294, 1177, 294, 294, 1177, 294, 294, 1177]
FLOPS:  1379127659.0 
PARAMS: 8940533.0

# 0.45 [35, 35, 35, 140, 35, 35, 140, 35, 35, 140, 70, 70, 281, 70, 70, 281, 70, 70, 281, 70, 70, 281, 140, 140, 563, 140, 140, 563, 140, 140, 563, 140, 140, 563, 140, 140, 563, 140, 140, 563, 281, 281, 1126, 281, 281, 1126, 281, 281, 1126]
FLOPS:  1268714643.0 
PARAMS: 8217318.0

# 0.465 [34, 34, 34, 136, 34, 34, 136, 34, 34, 136, 68, 68, 273, 68, 68, 273, 68, 68, 273, 68, 68, 273, 136, 136, 547, 136, 136, 547, 136, 136, 547, 136, 136, 547, 136, 136, 547, 136, 136, 547, 273, 273, 1095, 273, 273, 1095, 273, 273, 1095]
FLOPS:  1199614780.0 
PARAMS: 7791047.0
 
# 0.475 [33, 33, 33, 134, 33, 33, 134, 33, 33, 134, 67, 67, 268, 67, 67, 268, 67, 67, 268, 67, 67, 268, 134, 134, 537, 134, 134, 537, 134, 134, 537, 134, 134, 537, 134, 134, 537, 134, 134, 537, 268, 268, 1075, 268, 268, 1075, 268, 268, 1075]
FLOPS:  1157563834.0 
PARAMS: 7539826.0

# 0.5 [32, 32, 32, 128, 32, 32, 128, 32, 32, 128, 64, 64, 256, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024]
FLOPS:  1063426560.0 
PARAMS: 6917640.0
# 0.515 [31, 31, 31, 124, 31, 31, 124, 31, 31, 124, 62, 62, 248, 62, 62, 248, 62, 62, 248, 62, 62, 248, 124, 124, 496, 124, 124, 496, 124, 124, 496, 124, 124, 496, 124, 124, 496, 124, 124, 496, 248, 248, 993, 248, 248, 993, 248, 248, 993]
# 0.525 [30, 30, 30, 121, 30, 30, 121, 30, 30, 121, 60, 60, 243, 60, 60, 243, 60, 60, 243, 60, 60, 243, 121, 121, 486, 121, 121, 486, 121, 121, 486, 121, 121, 486, 121, 121, 486, 121, 121, 486, 243, 243, 972, 243, 243, 972, 243, 243, 972]

# 0.6 [25, 25, 25, 102, 25, 25, 102, 25, 25, 102, 51, 51, 204, 51, 51, 204, 51, 51, 204, 51, 51, 204, 102, 102, 409, 102, 102, 409, 102, 102, 409, 102, 102, 409, 102, 102, 409, 102, 102, 409, 204, 204, 819, 204, 204, 819, 204, 204, 819]
FLOPS:  683229050.0 
PARAMS: 4572522.0

# 0.7 [19, 19, 19, 76, 19, 19, 76, 19, 19, 76, 38, 38, 153, 38, 38, 153, 38, 38, 153, 38, 38, 153, 76, 76, 307, 76, 76, 307, 76, 76, 307, 76, 76, 307, 76, 76, 307, 76, 76, 307, 153, 153, 614, 153, 153, 614, 153, 153, 614]
FLOPS:  394161299.0 
PARAMS: 2724022.0
# 0.75 [16, 16, 16, 64, 16, 16, 64, 16, 16, 64, 32, 32, 128, 32, 32, 128, 32, 32, 128, 32, 32, 128, 64, 64, 256, 64, 64, 256, 64, 64, 256, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512]


def main():
    # model = resnet_50(cfg=[64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048])
    # model = resnet_50(cfg=[32, 32, 32, 180, 32, 32, 180, 32, 32, 180, 64, 64, 360, 64, 64, 360, 64, 64, 360, 64, 64, 360, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 256, 256, 2048, 256, 256, 2048, 256, 256, 2048])
    # model = resnet_50(cfg=[32, 32, 32, 180, 32, 32, 180, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048])
    model = resnet_50(cfg=[32, 32, 32, 180, 32, 32, 180, 32, 32, 180, 64, 64, 360, 64, 64, 360, 64, 64, 360, 64, 64, 360, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 256, 256, 2048, 256, 256, 2048, 256, 256, 2048])
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
    print("FLOPS: ", flops, "\nPARAMS:", params)
    # cfg1=[int(cfg[i]-(cfg[i]*compress_rate[i])) for i in range(len(cfg))]
    # inputs = torch.rand((1, 3, 224, 224)).cuda()
    # model = model.cuda().train()
    # output = model(inputs)
    # print(output.shape)
    # print(model)
    # model = model.cuda()
    # model = nn.DataParallel(model) 
    # print(model)
    # for k,m in enumerate(model.modules()):
    #     print("k: ", k, "\tm: ", m)
if __name__ == '__main__':
    main()

