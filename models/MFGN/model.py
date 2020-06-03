from math import log2, floor
from torch import nn, cat, add, tensor
from torch.nn import init, Upsample, Conv2d, ReLU, AvgPool2d, BatchNorm2d, Sequential, LeakyReLU, InstanceNorm2d
from torch.nn.functional import interpolate


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(tensor([init_value]), requires_grad=True)

    def forward(self, data):
        return data * self.scale

class Net(nn.Module):
    def __init__(self, scale_factor, input_channels=3):
        super().__init__()

        assert log2(scale_factor).is_integer(), "Scale factor must be power of two"
        self.scale_factor = scale_factor

        n_channels = 24
        group_size = 6

        start = [nn.Conv2d(input_channels, n_channels, kernel_size=5, stride=1, padding=2)] + \
                [ResResDenseBlock(n_channels) for _ in range(group_size)]
        self.start = nn.Sequential(*start)

        middle = [nn.AvgPool2d(2, 2)] + [ResResDenseBlock(n_channels) for _ in range(group_size)]
        self.middle = nn.Sequential(*middle)

        end = [ResResDenseBlock(n_channels) for _ in range(group_size)]
        self.end = nn.Sequential(*end)

        upsample = [UpsampleBLock(n_channels, 2)] * int(log2(scale_factor)) \
                   + [nn.Conv2d(n_channels, input_channels, kernel_size=3, padding=1)]
        self.upsample = nn.Sequential(*upsample)

        self.out_conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        preprocessed = self.start(x)

        downsampled = self.middle(preprocessed)

        c3 = preprocessed + nn.functional.interpolate(downsampled, scale_factor=2)
        c3 = self.end(c3)
        out = self.upsample(c3) + nn.functional.interpolate(x, scale_factor=self.scale_factor)
        return self.out_conv(out)


class ResResDenseBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()

        convs = [
            nn.Sequential(
                nn.Conv2d(i * n_channels, n_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )
            if i != 5
            else nn.Conv2d(i * n_channels, n_channels, kernel_size=3, padding=1)
            for i in range(1, 6)]
        self.convs = nn.Sequential(*convs)

        # self.scale = nn.Parameter(tensor(1e-3), requires_grad=True)
        self.scale = nn.Parameter(tensor([1e-3 for _ in range(n_channels)]).view(1, n_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        last = None
        inputs = x

        for idx, conv in enumerate(self.convs):
            last = conv(inputs)

            if idx < len(self.convs) - 1:
                inputs = cat((inputs, last), dim=1)
        return x + self.scale * last


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
