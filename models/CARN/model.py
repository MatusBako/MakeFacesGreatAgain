"""
Implementation of CARN architecture. The implementation is inspired by implementation
from cited repository, but there were few changes made so it is usable with this module. 


Original source: https://github.com/nmhkahn/CARN-pytorch
"""

from math import log2, floor
from torch import nn, cat, add, Tensor, eye
from torch.nn import init, Upsample, Conv2d, ReLU, AvgPool2d, BatchNorm2d
from torch.nn.functional import interpolate, relu


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(Tensor([init_value]))

    def forward(self, data):
        return data * self.scale


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            #TODO: ReLU
        )

    def forward(self, x):
        out = self.body(x)
        out = relu(out + x)
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.b1 = ResidualBlock(64, 64)
        self.b2 = ResidualBlock(64, 64)
        self.b3 = ResidualBlock(64, 64)
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class UpsampleBlock(nn.Module):
    def __init__(self, num_channels, scale):
        super(UpsampleBlock, self).__init__()

        self.up = _UpsampleBlock(num_channels, scale=scale)

    def forward(self, x, scale):
        return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, num_channels, scale):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(log2(scale))):
                modules += [nn.Conv2d(num_channels, 4 * num_channels, 3, 1, 1), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(num_channels, 9 * num_channels, 3, 1, 1), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


class Net(nn.Module):
    def __init__(self, scale_factor, num_channels=3):
        super(Net, self).__init__()

        self.scale_factor = scale_factor

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 4, 64, 1, 1, 0)

        self.upsample = UpsampleBlock(64, self.scale_factor)
        self.exit = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale_factor)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
