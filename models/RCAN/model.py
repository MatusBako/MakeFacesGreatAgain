"""
Implementation of RCAN architecture. The implementation is inspired by implementation
from cited repository, but there were few changes made so it is usable with this module.

Original source: https://github.com/yulunzhang/RCAN
"""

from math import log2, floor, sqrt
from torch import nn, cat, add, Tensor
from torch.nn import init, Upsample, Conv2d, ReLU, AvgPool2d, BatchNorm2d, AdaptiveAvgPool2d, Sigmoid
from torch.nn.functional import interpolate


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()

        # feature channel downscale and upscale --> channel weight
        self.layers = nn.Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            ReLU(inplace=True),
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            Sigmoid()
        )

    def forward(self, x):
        scale = self.layers(x)
        return x * scale


class ResChAttentionBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=True, res_scale=1.):
        super().__init__()

        self.layers = nn.Sequential(
            Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=1),
            ReLU(True),
            Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=1),
            CALayer(n_feat, reduction)
        )
        self.res_scale = res_scale

    def forward(self, x):
        ret = self.res_scale * self.layers(x)
        return x + ret


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, n_resblocks, reduction, res_scale=1):
        super().__init__()

        layers = [
            ResChAttentionBlock(n_feat, kernel_size, bias=True, reduction=reduction, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        layers.append(Conv2d(n_feat, n_feat, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        res = self.layers(x)
        res += x
        return res


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Net(nn.Module):
    def __init__(self, scale_factor, in_channels=3, res_groups=6, res_blocks=16, n_feats=32):
    # def __init__(self, scale_factor, in_channels=3, res_groups=6, res_blocks=12, n_feats=24):
        """

        :param scale_factor:
        :param in_channels:
        :param res_groups: default=10
        :param res_blocks: default=20
        :param n_feats: default=64
        """
        super().__init__()

        kernel_size = 3
        reduction = 8

        self.begin = nn.Sequential(Conv2d(in_channels, n_feats, kernel_size, padding=1))

        # define body module
        modules_body = [ResidualGroup(n_feats, kernel_size, res_blocks, reduction) for _ in range(res_groups)]
        modules_body.append(Conv2d(n_feats, n_feats, kernel_size, padding=1))

        self.body = nn.Sequential(*modules_body)

        self.end = nn.Sequential(
            UpsampleBLock(n_feats, scale_factor),
            Conv2d(n_feats, in_channels, kernel_size, padding=1),
        )

    def forward(self, x):
        x = self.begin(x)

        res = self.body(x)
        res += x

        x = self.end(res)
        return x
