from math import log2
from torch import nn, tanh, sigmoid, tensor
from torchvision.models import vgg19


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(log2(scale_factor))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        residuals = [ResidualBlock(64)] * 16
        self.residuals = nn.Sequential(*residuals)

        self.pre_upsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        upsample = [UpsampleBLock(64, 2)] * upsample_block_num \
            + [nn.Conv2d(64, 3, kernel_size=9, padding=4)]
        self.upsample = nn.Sequential(*upsample)

    def forward(self, x):
        block1 = self.block1(x)
        residual = self.residuals(block1)
        pre_upsample = self.pre_upsample(residual)
        upsample = self.upsample(block1 + pre_upsample)
        return upsample


class Discriminator(nn.Module):
    def __init__(self, input_size=256):
        super(Discriminator, self).__init__()
        # fcc_in_params = 512 * input_size * input_size // 128

        convs = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0),
            # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0),
            # nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0),
            # nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # - 3 for 3 downsamples, -1 for output to be 512x2x2
        for i in range(int(log2(input_size)) - 3 - 1):
            convs.extend([
                # nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        self.convs = nn.Sequential(*convs)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv = self.convs(x)
        map = self.fc(conv.view(x.size(0), -1))
        return map.view(x.size(0))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.arch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return x + self.arch(x)


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


class FeatureExtractor(nn.Module):
    def __init__(self, maxpool_idx=2, no_pooling=True):
        """
        Used for computing perception loss (MSE over activations)

        :param maxpool_idx: either 1 or 4
        :param no_pooling: flag if pooling layer is also included
        """
        super(FeatureExtractor, self).__init__()

        children = vgg19(pretrained=True).features
        max_pool_indices = [index for index, layer in enumerate(children) if isinstance(layer, nn.MaxPool2d)]

        # get all layers up to chosen maxpool layer
        maxpool_idx = max_pool_indices[maxpool_idx]

        if no_pooling:
            maxpool_idx -= 1

        layers = children[:maxpool_idx]
        self.features = nn.Sequential(*layers).requires_grad_(False).eval()

        self.register_buffer("mean_val", tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)))
        self.register_buffer("std_val", tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)))

    def preprocess(self, x):
        return (x - self.mean_val) / self.std_val

    def forward(self, x):
        return self.features(self.preprocess(x))
