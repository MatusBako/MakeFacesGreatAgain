from math import log2, floor
from torch import nn, cat, add, Tensor
from  torch.nn import init, Upsample, Conv2d, ReLU
from torch.nn.functional import interpolate


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(Tensor([init_value]))

    def forward(self, data):
        return data * self.scale


class Net(nn.Module):
    def __init__(self, scale_factor, num_channels=3, base_channels=64, num_residuals=6):
        super(Net, self).__init__()

        self.upscale_factor = scale_factor

        self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.ReLU(inplace=True))

        conv_blocks = []
        for _ in range(num_residuals):
            conv_blocks.append(nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.ReLU(inplace=True)))
        self.residual_layers = nn.Sequential(*conv_blocks)

        self.output_conv = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=True)

        residual = x.clone()
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        x = add(x, residual)
        return x

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

