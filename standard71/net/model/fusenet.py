## two stream network

from net.util import *



#----- helper functions --------
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

#############################################################################3


class JpgNet(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(JpgNet, self).__init__()
        in_channels, height, width = in_shape

        self.preprocess = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 16, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 16, 16, kernel_size=1, stride=1, padding=0 ),
        ) # 128

        self.conv1d = nn.Sequential(
            *make_conv_bn_relu( 16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 32, 32, kernel_size=3, stride=1, padding=1 ),
        ) # 64

        self.conv2d = nn.Sequential(
            *make_conv_bn_relu( 32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 64, 64, kernel_size=3, stride=1, padding=1 ),
        ) # 32

        self.conv3d = nn.Sequential(
            *make_conv_bn_relu( 64,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128,128, kernel_size=3, stride=1, padding=1 ),
        ) # 16

        self.conv4d = nn.Sequential(
            *make_conv_bn_relu(128,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256,256, kernel_size=3, stride=1, padding=1 ),
        ) # 8



    def forward(self, x):

        out    = self.preprocess(x)
        conv1d = self.conv1d(out)                                         #128
        out    = F.max_pool2d(conv1d, kernel_size=2, stride=2)  # 64

        conv2d = self.conv2d(out)                                         # 64
        out    = F.max_pool2d(conv2d, kernel_size=2, stride=2) # 32

        conv3d = self.conv3d(out)                                         # 32
        out    = F.max_pool2d(conv3d, kernel_size=2, stride=2) # 16

        conv4d = self.conv4d(out)                                         # 16
        out    = F.max_pool2d(conv4d, kernel_size=2, stride=2) #  8

        return out



#############################################################################3

class TifNet(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(TifNet, self).__init__()
        in_channels, height, width = in_shape

        self.preprocess = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 16, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 16, 16, kernel_size=1, stride=1, padding=0 ),
        ) # 128

        self.conv1d = nn.Sequential(
            *make_conv_bn_relu( 16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 32, 32, kernel_size=3, stride=1, padding=1 ),
        ) # 64

        self.conv2d = nn.Sequential(
            *make_conv_bn_relu( 32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 64, 64, kernel_size=3, stride=1, padding=1 ),
        ) # 32

        self.conv3d = nn.Sequential(
            *make_conv_bn_relu( 64,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128,128, kernel_size=3, stride=1, padding=1 ),
        ) # 16

        self.conv4d = nn.Sequential(
            *make_conv_bn_relu(128,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256,256, kernel_size=3, stride=1, padding=1 ),
        ) # 8



    def forward(self, x):

        out    = self.preprocess(x)
        conv1d = self.conv1d(out)                                         #128
        out    = F.max_pool2d(conv1d, kernel_size=2, stride=2)  # 64

        conv2d = self.conv2d(out)                                         # 64
        out    = F.max_pool2d(conv2d, kernel_size=2, stride=2) # 32

        conv3d = self.conv3d(out)                                         # 32
        out    = F.max_pool2d(conv3d, kernel_size=2, stride=2) # 16

        conv4d = self.conv4d(out)                                         # 16
        out    = F.max_pool2d(conv4d, kernel_size=2, stride=2) #  8

        return out

#############################################################################3


class FuseNet(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(FuseNet, self).__init__()
        in_channels, height, width = in_shape

        self.jpg_net = JpgNet( (3, height, width), num_classes )
        self.tif_net = TifNet( (4, height, width), num_classes )

        self.cls = nn.Sequential(
            *make_linear_bn_relu(256+256, 512),
            ##*make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        #
        jpg   = self.jpg_net(x[:,0:3,:,:])
        jpg_flat = F.adaptive_avg_pool2d(jpg, output_size=1)
        jpg_flat = jpg_flat.view(jpg_flat.size(0), -1)

        tif   = self.tif_net(x[:,3:7,:,:])
        tif_flat = F.adaptive_avg_pool2d(tif, output_size=1)
        tif_flat = tif_flat.view(tif_flat.size(0), -1)

        flat  = torch.cat([jpg_flat, tif_flat],1)
        #flat  = tif_flat  # jpg_flat
        logit = self.cls(flat)
        prob  = F.sigmoid(logit)

        return logit,prob




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 64
    num_classes = 17
    C,H,W = 7,112,112

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