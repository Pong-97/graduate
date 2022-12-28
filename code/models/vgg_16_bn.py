import math
import torch.nn as nn
from collections import OrderedDict
import torch
from thop import profile
import pdb

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
        # pdb.set_trace()
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
        x = self.features(x)

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


def vgg_16_bn(compress_rate=None, cfg=None):
    return VGG(compress_rate=compress_rate, cfg=cfg)

def main():
    net = vgg_16_bn(cfg=[45, 45, 'M', 90, 90, 'M', 160, 160, 160, 'M', 180, 180, 200, 'M', 200, 200, 200, 200], compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    # net = vgg_16_bn(cfg=[40, 40, 'M', 70, 70, 'M', 140, 140, 140, 'M', 160, 160, 160, 'M', 160, 160, 160, 160], compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    # checkpoint = torch.load("/home2/pyf/vgg_pruned_model/93.44_0.675M_69.28M_model_best_new.pth.tar", map_location='cuda:0')
    # teacher_model = vgg_16_bn()
    # teacher_model_path = '/home2/pengyifan/pyf/hypergraph_cluster/log/pretrained_model/vgg_16_bn.pt.pt'
    # print("=> loading teacher model checkpoint '{}'".format(teacher_model_path))
    # checkpoint = torch.load(teacher_model_path, map_location='cuda:0')
    # teacher_model.load_state_dict(checkpoint['state_dict'], strict=True)
    # teacher_model.cuda()
    # teacher_model = torch.nn.DataParallel(teacher_model)
    # net = vgg_16_bn(cfg=[24, 20, 'M', 38, 49, 'M', 75, 92, 72, 'M', 129, 68, 61, 'M', 22, 40, 48,48], compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    # for m0 in net.modules():
    #     if isinstance(m0, nn.Conv2d):
    #         m1 = torch.flatten(m0.weight.data)
    #         nonzero_num = torch.nonzero(m1).shape[0]
    #         all_num = m1.shape[0]
    #         zero_rate = (all_num-nonzero_num)/all_num
    #         print(zero_rate)
    #         pdb.set_trace()
    # print(net)
    # # pdb.set_trace()
    # print("\nSmall test.")
    # inputs = torch.rand((1, 3, 32, 32)).cuda()
    # model = net.cuda().eval()
    # output = model.forward(inputs)
    # print(output.shape)
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, inputs=(input, ) )  #  profile（模型，输入数据）
    print("FLOPS: ", flops, "\nPARAMS:", params)

    net_ori = vgg_16_bn(compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    flops_ori, params_ori = profile(net_ori, inputs=(input, ) )  #  profile（模型，输入数据）
    print("FLOPS ori: ", flops_ori, "\nPARAMS_ori:", params_ori)
    print("FLOPS: ", flops, "\nPARAMS:", params)
    print("FLOPS compress rate: ", 1-flops/flops_ori, "\nPARAMS compress rate:", 1-params/params_ori)

if __name__ == '__main__':
    main()