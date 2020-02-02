from math import log2, floor
from torch import nn, cat, add, Tensor
from torch.nn import init, Upsample, Conv2d, ReLU, AvgPool2d, BatchNorm2d, Sequential, LeakyReLU
from torch.nn.functional import interpolate


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(Tensor([init_value]), requires_grad=True)

    def forward(self, data):
        return data * self.scale


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.block = Sequential(
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            ScaleLayer()
        )

    def forward(self, x):
        output = self.block(x)
        return add(output, x)


class Net(nn.Module):
    def __init__(self, upscale_factor: int, num_channels=3):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor

        self.d1 = nn.Sequential(
            Conv2d(num_channels, 64, kernel_size=5, stride=1, padding=2),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        self.pool1 = AvgPool2d(2, 2)

        self.d2 = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        self.u1 = nn.Sequential(

            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            Conv2d(64, num_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        self.out_conv = Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        #self.scale = ScaleLayer()
        #self._initialize_weights()

        # avgpool, upsample

    def _initialize_weights(self):
        for layer in self.layers[:-1]:
            init.orthogonal_(layer.weight, init.calculate_gain('relu'))

        #init.orthogonal_(self.layer[-1].weight)

    def forward(self, x):
        c1 = self.d1(x)

        c2 = self.pool1(c1)
        c2 = self.d2(c2)

        c3 = add(c1, interpolate(c2, scale_factor=2))
        c3 = self.u1(c3)
        out = add(interpolate(x, scale_factor=self.upscale_factor), c3)

        return out
