from math import log2, floor,sqrt
from torch import nn, cat, add, Tensor
from torch.nn import init, Upsample, Conv2d, ReLU, AvgPool2d, BatchNorm2d
from torch.nn.functional import interpolate


class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = add(output, identity_data)
        return output 


class Net(nn.Module):
    def __init__(self, scale_factor):
        super(Net, self).__init__()
        assert log2(scale_factor) % 1 == 0, "Scale factor must be power of 2!"

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        upscale_block = [
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        upscale = [layer for _ in range(int(log2(scale_factor))) for layer in upscale_block]
        self.upscale4x = nn.Sequential(*upscale)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out