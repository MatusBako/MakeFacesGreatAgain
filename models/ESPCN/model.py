from math import log2, floor
from torch import nn, cat, add, Tensor
import torch.nn.init as init


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(Tensor([init_value]))

    def forward(self, data):
        return data * self.scale


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Net(nn.Module):
    def __init__(self, upscale_factor, num_channels=3, d=64, s=12, m=4):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        #self.scale = ScaleLayer()

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.relu(c1)
        c2 = self.conv2(c1)
        c2 = self.relu(c2)
        c3 = self.conv3(c2)
        c3 = self.relu(c3)
        out = self.conv4(c3)
        out = self.pixel_shuffle(out)
        return out
