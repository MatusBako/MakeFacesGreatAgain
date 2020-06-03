from math import log2
from torch import nn, tanh, sigmoid, add, cat, tensor
from torchvision.models import vgg19, vgg13


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_count = int(log2(scale_factor))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        residuals = [ResResDenseBlock(64)] * 16
        upsample = [UpsampleBLock(64, 2)] * upsample_block_count

        self.start = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.residuals = nn.Sequential(*residuals)
        self.pre_upsample = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(*upsample)
        self.end = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.start(x)

        res_out = self.residuals(x)
        res_out = self.pre_upsample(res_out)

        x = self.upsample(add(x, res_out))
        return self.end(x)


# TODO: hourglass architecture
class Discriminator(nn.Module):
    def __init__(self, input_size=256):
        super(Discriminator, self).__init__()

        convs = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # - 3 for 3 downsamples, -1 for output to be 512x2x2
        for i in range(int(log2(input_size)) - 3 - 1):
            convs.extend([
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            )

        self.convs = nn.Sequential(*convs)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        conv = self.convs(x)
        map = self.fc(conv.view(x.size(0), -1))
        return map.view(x.size(0))


class ResResDenseBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()

        self.convs = [
            nn.Sequential(
                nn.Conv2d(i * n_channels, n_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )
            if i != 5
            else nn.Conv2d(i * n_channels, n_channels, kernel_size=3, padding=1)
            for i in range(1, 6)]
        self.convs = nn.Sequential(*self.convs)
        self.scale = ScaleLayer()

    def forward(self, x):
        last = None
        inputs = x

        for idx, conv in enumerate(self.convs):
            last = conv(inputs)

            if idx < len(self.convs) - 1:
                inputs = cat((inputs, last), dim=1)
        return x + self.scale(last)


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(tensor([init_value]), requires_grad=True)

    def forward(self, data):
        return data * self.scale


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
    def __init__(self, maxpool_idx=1, skip_n_last_layers=2):
        """
        Used for computing perception loss (MSE over activations)

        :param maxpool_idx: either 1 or 4
        :param skip_n_last_layers: useful for skipping maxpool and activation
        """
        super(FeatureExtractor, self).__init__()

        children = vgg19(pretrained=True).features
        max_pool_indices = [index for index, layer in enumerate(children) if isinstance(layer, nn.MaxPool2d)]

        # get all layers up to chosen maxpool layer
        maxpool_idx = max_pool_indices[maxpool_idx]

        # skip maxpooling and activation
        if skip_n_last_layers > 0:
            maxpool_idx -= skip_n_last_layers

        layers = children[:maxpool_idx]
        self.features = nn.Sequential(*layers).requires_grad_(False).eval()

        self.register_buffer("mean_val", tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)))
        self.register_buffer("std_val", tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)))

    def preprocess(self, x):
        return (x - self.mean_val) / self.std_val

    def forward(self, x):
        return self.features(self.preprocess(x))
