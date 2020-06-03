
"""
Implementation of CARN architecture. The implementation is inspired by implementation
from cited repository, but there were few changes made so it is usable with this module.


Original source: https://github.com/nmhkahn/CARN-pytorch
"""

from math import log2, floor
from torch import nn, cat, add, Tensor
from torch.nn import init, Upsample, Conv2d, ReLU, AvgPool2d, BatchNorm2d
from torch.nn.functional import interpolate, upsample


class Net(nn.Module):
    def __init__(self, scale_factor, num_channels=3, base_channels=64, feat_channels=256):
        super(Net, self).__init__()

        if scale_factor == 2:
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2
        else:
            kernel_size = None
            stride = None
            padding = None
            Warning("please choose the scale factor from 2, 4, 8")
            exit()

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat_channels, norm=None)
        self.feat1 = ConvBlock(feat_channels, base_channels, kernel_size=1, stride=1, padding=0, norm='batch')
        # Back-projection stages
        self.up1 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down1 = DownBlock(base_channels, kernel_size, stride, padding)
        self.up2 = UpBlock(base_channels, kernel_size, stride, padding)
        # Reconstruction
        self.output_conv = ConvBlock(base_channels * 2, num_channels)
    # 
    # def weight_init(self):
    #     for m in self._modules:
    #         class_name = m.__class__.__name__
    #         if class_name.find('Conv2d') != -1:
    #             nn.init.kaiming_normal(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif class_name.find('ConvTranspose2d') != -1:
    #             nn.init.kaiming_normal(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))

        x = self.output_conv(cat((h2, h1), 1))

        return x


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True,
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.act = nn.PReLU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        return self.act(out)


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        self.act = nn.PReLU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


