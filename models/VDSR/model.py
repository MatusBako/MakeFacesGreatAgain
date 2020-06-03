from math import log2, floor
from torch import nn, cat, add, Tensor
from  torch.nn import init, Upsample, Conv2d, ReLU
from torch.nn.functional import interpolate


class Net(nn.Module):
    def __init__(self, scale_factor, num_channels=3, base_channels=64, num_residuals=20):
        super(Net, self).__init__()

        self.upscale_factor = scale_factor

        self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.ReLU())

        conv_blocks = []
        for _ in range(num_residuals):
            conv_blocks.append(nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.ReLU()))
        self.residual_layers = nn.Sequential(*conv_blocks)

        self.output_conv = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: Tensor):
        img = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=True)

        x = self.input_conv(img)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        return img + x
