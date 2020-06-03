from math import log2, floor
from torch import nn, cat, add, Tensor
from  torch.nn import init, Upsample, Conv2d, ReLU, Sequential
from torch.nn.functional import interpolate


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(Tensor([init_value]))

    def forward(self, data):
        return data * self.scale


class Net(nn.Module):
    def __init__(self, upscale_factor, num_channels=3, base_channel=256, num_residuals=32):
        super(Net, self).__init__()
        assert log2(upscale_factor).is_integer(), "Upscale factor must be power of two"

        self.input_conv = nn.Conv2d(num_channels, base_channel, kernel_size=3, stride=1, padding=1)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_channel, kernel=3, stride=1, padding=1))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)

        upscale_layers = [PixelShuffleBlock(base_channel, base_channel, upscale_factor=2)
                          for _ in range(int(log2(upscale_factor)))]
        self.upscale_layers = Sequential(*upscale_layers)

        self.output_conv = nn.Conv2d(base_channel, num_channels, kernel_size=3, stride=1, padding=1)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.input_conv(x)
        residual = x
        x = self.residual_layers(x)
        x = self.mid_conv(x)
        x += residual
        x = self.upscale_layers(x)
        x = self.output_conv(x)
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()


class ResnetBlock(nn.Module):
    def __init__(self, num_channel, kernel=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        #self.bn = nn.BatchNorm2d(num_channel)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        #x = self.bn(self.conv1(x))
        x = self.conv1(x)
        x = self.activation(x)
        #x = self.bn(self.conv2(x))
        x = x + residual
        return x * 0.1


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor, kernel=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2, kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.ps(self.conv(x))
        return x
