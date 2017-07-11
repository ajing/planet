import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]

def make_linear_bn_relu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_max_flat(out):
    flat = F.adaptive_max_pool2d(out,output_size=1)  ##nn.AdaptiveMaxPool2d(1)(out)
    flat = flat.view(flat.size(0), -1)
    return flat


def make_avg_flat(out):
    flat = F.adaptive_avg_pool2d(out,output_size=1)
    flat = flat.view(flat.size(0), -1)
    return flat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class ResNet(nn.Module):

    def __init__(self, block, layers, in_shape=(3,256,256), num_classes=17):
        self.inplanes = 64

        super(ResNet, self).__init__()
        in_channels, height, width = in_shape

        # self.conv0 = nn.Sequential(
        #     *make_conv_bn_relu(in_channels, 64, kernel_size=7, stride=2, padding=3, groups=1)
        # )
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.fc =  nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x) # 64, 64x64
        x = self.layer2(x) #128, 32x32
        x = self.layer3(x) #256, 16x16
        x = self.layer4(x) #512,  8x8
        x = make_avg_flat(x)
        x = self.fc(x)

        return x


def pyresnet18(**kwargs):
    model = PyResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def pyresnet34(**kwargs):
    model = PyResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


#############################################################################3


class FuseNet(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(FuseNet, self).__init__()
        in_channels, height, width = in_shape

        self.tif_preprocess = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=True)

        features_dim=256
        self.jpg_net = ResNet(BasicBlock, [2, 2, 2, 2], in_shape=(3, height, width), num_classes=features_dim)
        self.tif_net = ResNet(BasicBlock, [2, 2, 2, 2], in_shape=(3, height, width), num_classes=features_dim)

        self.fc = nn.Sequential(
            *make_linear_bn_relu(2*features_dim, 512),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        #
        jpg   = self.jpg_net(x[:,0:3,:,:])
        tif   = self.tif_net(self.tif_preprocess(x[:,3:7,:,:]))

        flat  = torch.cat([jpg, tif],1)
        logit = self.fc(flat)
        prob  = F.sigmoid(logit)
        return logit,prob




########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 17
    C,H,W = 7,256,256

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]

    if 1:
        net = FuseNet(in_shape=in_shape, num_classes=num_classes).cuda().train()

        x = Variable(inputs)
        logits, probs = net.forward(x.cuda())

        loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels.cuda()))
        loss.backward()

        print(type(net))
        print(net)

        print('probs')
        print(probs)

        #input('Press ENTER to continue.')

        #try changing keys
        pretrained_file = '/root/share/project/pytorch/data/pretrain/resnet/resnet18-5c106cde.pth'

        pretrained_dict = torch.load(pretrained_file)
        model_dict = net.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip_list }

    ## debug
    if 0:
        print('model_dict.keys()')
        print(model_dict.keys())
        print('pretrained_dict.keys()')
        print(pretrained_dict.keys())
        print('pretrained_dict1.keys()')
        print(pretrained_dict1.keys())

# d = {'x':1, 'y':2, 'z':3}
# d1 = {'x':'a', 'y':'b', 'z':'c'}
#
# In [10]: dict((d1[key], value) for (key, value) in d.items())
# Out[10]: {'a': 1, 'b': 2, 'c': 3}