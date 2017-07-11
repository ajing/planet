#inception v2, i.e. googlenet-bn
#   http://programtalk.com/vs2/python/10352/tensorpack/examples/Inception/inception-bn.py/
#   https://github.com/pertusa/InceptionBN-21K-for-Caffe
#   https://github.com/tensorflow/models/tree/master/slim
#   https://github.com/tensorflow/models/blob/master/slim/nets/inception_v2.py
#   https://github.com/lim0606/caffe-googlenet-bn
#   https://github.com/pfnet/chainer/blob/master/examples/imagenet/googlenetbn.py
#
#  caffe converter:  https://github.com/ruotianluo/pytorch-resnet
#                    https://github.com/jcjohnson/pytorch-vgg
#
#  torch to pytorch:  https://discuss.pytorch.org/t/convert-import-torch-model-to-pytorch/37/2
#
#  https://github.com/ysh329/deep-learning-model-convertor


'''
Inception-BN model on ILSVRC12.
See "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", arxiv:1502.03167
'''

import os
from torch.autograd import Variable

#--------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn   = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x



#avg pool
class InceptionA(nn.Module):

    def __init__(self, in_channels, num_1x1, num_3x3_r, num_3x3, num_d3x3_r, num_d3x3, num_pool, pool_type):
        super(InceptionA, self).__init__()

        self.branch_1x1    = BasicConv2d(in_channels, num_1x1,    kernel_size=1, stride=1, padding=0)

        self.branch_3x3_r  = BasicConv2d(in_channels, num_3x3_r,  kernel_size=1, stride=1, padding=0)
        self.branch_3x3    = BasicConv2d(num_3x3_r,   num_3x3,    kernel_size=3, stride=1, padding=1)

        self.branch_d3x3_r = BasicConv2d(in_channels, num_d3x3_r, kernel_size=1, stride=1, padding=0)
        self.branch_d3x3_0 = BasicConv2d(num_d3x3_r,  num_d3x3,   kernel_size=3, stride=1, padding=1)
        self.branch_d3x3_1 = BasicConv2d(num_d3x3,    num_d3x3,   kernel_size=3, stride=1, padding=1)

        if pool_type=='avg':
            self.pool = nn.AvgPool2d(kernel_size=3,stride=1, padding=1)
        elif pool_type=='max':
            self.pool = nn.MaxPool2d(kernel_size=3,stride=1, padding=1)
        else:
            raise Exception('unknown pool_type: %?'%pool_type)

        self.branch_pool   = BasicConv2d(in_channels, num_pool,   kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        branch_1x1  = self.branch_1x1(x)

        branch_3x3  = self.branch_3x3_r(x)
        branch_3x3  = self.branch_3x3  (branch_3x3)

        branch_d3x3 = self.branch_d3x3_r(x)
        branch_d3x3 = self.branch_d3x3_0(branch_d3x3)
        branch_d3x3 = self.branch_d3x3_1(branch_d3x3)


        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        output = torch.cat([branch_1x1, branch_3x3, branch_d3x3, branch_pool],1)
        return output



#max pool
class InceptionB(nn.Module):

    def __init__(self, in_channels, num_3x3_r, num_3x3, num_d3x3_r, num_d3x3, pool_type):
        super(InceptionB, self).__init__()

        self.branch_3x3_r  = BasicConv2d(in_channels, num_3x3_r, kernel_size=1, stride=1, padding=0)
        self.branch_3x3    = BasicConv2d(num_3x3_r,   num_3x3,   kernel_size=3, stride=2, padding=1)

        self.branch_d3x3_r = BasicConv2d(in_channels, num_d3x3_r, kernel_size=1, stride=1, padding=0)
        self.branch_d3x3_0 = BasicConv2d(num_d3x3_r,  num_d3x3,   kernel_size=3, stride=1, padding=1)
        self.branch_d3x3_1 = BasicConv2d(num_d3x3,    num_d3x3,   kernel_size=3, stride=2, padding=1)

        if pool_type=='avg':
            self.pool = nn.AvgPool2d(kernel_size=3,stride=2, padding=1)
        elif pool_type=='max':
            self.pool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        else:
            raise Exception('unknown pool_type: %?'%pool_type)

    def forward(self, x):

        branch_3x3  = self.branch_3x3_r(x)
        branch_3x3  = self.branch_3x3  (branch_3x3)

        branch_d3x3 = self.branch_d3x3_r(x)
        branch_d3x3 = self.branch_d3x3_0(branch_d3x3)
        branch_d3x3 = self.branch_d3x3_1(branch_d3x3)


        branch_pool = self.pool(x)
        output = torch.cat([branch_3x3, branch_d3x3, branch_pool],1)
        return output

class Inception2(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(Inception2, self).__init__()
        in_channels, height, width = in_shape

        self.conv1 = BasicConv2d(in_channels, 64,    kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)

        self.conv2_r = BasicConv2d( 64, 64, kernel_size=1, stride=1, padding=0)
        self.conv2   = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2   = nn.MaxPool2d(kernel_size=3,stride=2, padding=0)

        self.inc_3a = InceptionA(192, num_1x1=64, num_3x3_r= 64, num_3x3= 64, num_d3x3_r=64, num_d3x3= 96, num_pool= 32, pool_type='avg')
        self.inc_3b = InceptionA(288, num_1x1=64, num_3x3_r= 64, num_3x3= 96, num_d3x3_r=64, num_d3x3= 96, num_pool= 64, pool_type='avg')
        self.inc_3c = InceptionB(320,             num_3x3_r=128, num_3x3=160, num_d3x3_r=64, num_d3x3= 96,               pool_type='max')


        self.inc_4a = InceptionA(576, num_1x1=224, num_3x3_r= 64, num_3x3= 96, num_d3x3_r= 96, num_d3x3=128, num_pool= 128, pool_type='avg')
        self.inc_4b = InceptionA(576, num_1x1=192, num_3x3_r= 96, num_3x3=128, num_d3x3_r= 96, num_d3x3=128, num_pool= 128, pool_type='avg')
        self.inc_4c = InceptionA(576, num_1x1=160, num_3x3_r=128, num_3x3=160, num_d3x3_r=128, num_d3x3=160, num_pool= 128, pool_type='avg')
        self.inc_4d = InceptionA(608, num_1x1= 96, num_3x3_r=128, num_3x3=192, num_d3x3_r=160, num_d3x3=192, num_pool= 128, pool_type='avg')
        self.inc_4e = InceptionB(608,              num_3x3_r=128, num_3x3=192, num_d3x3_r=192, num_d3x3=256,                pool_type='max')

        self.inc_5a = InceptionA(1056, num_1x1=352, num_3x3_r=192, num_3x3=320, num_d3x3_r=160, num_d3x3=224, num_pool= 128, pool_type='avg')
        self.inc_5b = InceptionA(1024, num_1x1=352, num_3x3_r=192, num_3x3=320, num_d3x3_r=192, num_d3x3=224, num_pool= 128, pool_type='max')

        self.fc = nn.Linear(1024, num_classes)

        #the auxiliary classifiers

    def forward(self, x):

                            #3x227x227    ,256
        x = self.conv1(x)   #96x114x114   ,128
        x = self.pool1(x)   #96x57x57     , 64

        x = self.conv2_r(x) #128x57x57   , 64
        x = self.conv2  (x) #228x57x57   , 64
        x = self.pool2  (x) #288x28x28   , 32

        x = self.inc_3a (x) #384x28x28   , 32
        x = self.inc_3b (x) #480x28x28   , 32
        x = self.inc_3c (x) #864x14×14   , 16


        x = self.inc_4a (x) #576x14x14   , 16
        x = self.inc_4b (x) #576x14x14   , 16
        x = self.inc_4c (x) #608x14x14   , 16
        x = self.inc_4d (x) #512x14x14   , 16
        x = self.inc_4e (x) #960x7x7     ,  8

        x = self.inc_5a (x) #1024x7x7
        x = self.inc_5b (x) #1024x7x7

        x = F.adaptive_avg_pool2d(x,output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logit = x
        prob  = F.sigmoid(logit)
        return logit,prob

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    inputs = torch.randn(1,3,256,256)
    in_shape = inputs.size()[1:]
    num_classes = 17

    if 1:
        net = Inception2(in_shape=in_shape, num_classes=num_classes).cuda().train()
        x = Variable(inputs).cuda()

        logit,prob = net.forward(x)

        #dot = make_dot(y)
        #dot.view()
        print(type(net))
        print(net)
        print(prob)