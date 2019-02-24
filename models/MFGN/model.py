from math import log2, floor
from torch import nn, cat, add, Tensor
from torch.nn import init, Upsample, Conv2d, ReLU, AvgPool2d, BatchNorm2d
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

        self.weight = ScaleLayer()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = add(self.weight(output), x)
        return output


class Net(nn.Module):
    def __init__(self, upscale_factor, num_channels=3):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.b_norm = BatchNorm2d(64) #TODO: odstranit

        # Feature extraction
        self.relu = ReLU()
        self.conv1 = Conv2d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.pool1 = AvgPool2d(2, 2)

        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # upsample

        self.conv5 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv2d(64, num_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv7 = Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

        #self.scale = ScaleLayer()
        #self._initialize_weights()

        # avgpool, upsample

    def _initialize_weights(self):
        for layer in self.layers[:-1]:
            init.orthogonal_(layer.weight, init.calculate_gain('relu'))

        #init.orthogonal_(self.layer[-1].weight)

    def forward(self, x):
        c1 = self.conv1(x)
        #c1 = self.b_norm(c1)
        c1 = self.relu(c1)
        c2 = self.conv2(c1)
        #c2 = self.b_norm(c2)
        c2 = self.relu(c2)

        # downsampling
        c3 = self.pool1(c2)

        c3 = self.conv3(c3)
        #c3 = self.b_norm(c3)
        c3 = self.relu(c3)

        c4 = self.conv4(c3)
        #c4 = self.b_norm(c4)
        c4 = self.relu(c4)

        c5 = add(c2, interpolate(c4, scale_factor=2))
        c5 = self.conv5(c5)
        #c5 = self.b_norm(c5)
        c5 = self.relu(c5)
        c6 = self.conv6(c5)

        out = self.pixel_shuffle(c6)
        out = self.conv7(add(interpolate(x, scale_factor=self.upscale_factor), out))
        #out = add(x, self.pixel_shuffle(c6))
        #out = add(self.scale(x), out)
        return out

