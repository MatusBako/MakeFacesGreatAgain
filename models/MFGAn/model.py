from math import log2, sqrt
from torch import nn, tanh, sigmoid, tensor, add, cat, Tensor, zeros, no_grad
from torch.nn.functional import interpolate
from torchvision.models import vgg19, vgg13, vgg16


class Dummy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Generator(nn.Module):
    def __init__(self, scale_factor, input_channels=3, normalization=nn.InstanceNorm2d):
        super().__init__()

        assert log2(scale_factor).is_integer(), "Scale factor must be power of two"
        self.scale_factor = scale_factor

        n_channels = 24
        group_size = 6

        # normalization = Dummy
        # normalization = nn.BatchNorm2d

        start = [nn.Conv2d(input_channels, n_channels, kernel_size=5, stride=1, padding=2)] + \
                [ResResDenseBlock(n_channels, normalization) for _ in range(group_size)]
        self.start = nn.Sequential(*start)

        # middle = [nn.AvgPool2d(2, 2)] + [ResResDenseBlock(n_channels) for _ in range(group_size)]
        middle = [ResResDenseBlock(n_channels, normalization) for _ in range(group_size)]
        self.middle = nn.Sequential(*middle)

        end = [ResResDenseBlock(n_channels, normalization) for _ in range(group_size)]
        self.end = nn.Sequential(*end)

        upsample = [UpsampleBLock(n_channels, 2)] * int(log2(scale_factor)) \
                   + [nn.Conv2d(n_channels, input_channels, kernel_size=3, padding=1)]
        self.upsample = nn.Sequential(*upsample)

        self.out_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1),
        )

    def forward(self, x):
        preprocessed = self.start(x)

        downsampled = self.middle(nn.functional.interpolate(preprocessed, scale_factor=.5))

        c3 = preprocessed + nn.functional.interpolate(downsampled, scale_factor=2)
        c3 = self.end(c3)
        out = self.upsample(c3) + nn.functional.interpolate(x, scale_factor=self.scale_factor)
        return self.out_conv(out)


class ResResDenseBlock(nn.Module):
    def __init__(self, n_channels: int, normalization):
        super().__init__()

        convs = [
            nn.Sequential(
                nn.Conv2d(i * n_channels, n_channels, kernel_size=3, padding=1),
                normalization(n_channels),
                nn.LeakyReLU(),
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


# TODO: hourglass architecture
class Discriminator(nn.Module):
    def __init__(self, input_size=256, normalization=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()

        # normalization = Dummy
        # normalization = nn.BatchNorm2d

        self.input_size = input_size

        convs = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            normalization(64),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            # nn.AvgPool2d(2, 2),
            normalization(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=2, stride=2),
            # nn.AvgPool2d(2, 2),
            normalization(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            normalization(256),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=2, stride=2),
            # nn.AvgPool2d(2, 2),
            normalization(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            normalization(512),
            nn.LeakyReLU(0.2),
        ]

        # - 3 for 3 downsamples, -1 for output to be 512x2x2
        for i in range(int(log2(input_size)) - 3 - 1):
            convs.extend([
                # nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(512, 512, kernel_size=2, stride=2),
                # nn.AvgPool2d(2, 2),
                normalization(512),
                nn.LeakyReLU(0.2),
            ])

        self.convs = nn.Sequential(*convs)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.view(x.size(0), -1))
        return x.view(x.size(0))


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)

        # TODO: DEBUG
        ircn(self.conv.weight, up_scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


def ircn(w: nn.Parameter, scale_factor, init=None):
    """
    Initialization for convolution in upsampling block to prevent artifacts.
    Paper: https://arxiv.org/abs/1707.02937
    """

    # new_shape = [size // (scale_factor ** 2) if idx == 2 else size for idx, size in enumerate(w.shape)]
    new_shape = [size // (scale_factor ** 2) if idx == 1 else size for idx, size in enumerate(w.shape)]
    x = zeros(new_shape)

    if init is not None:
        init(x)
    else:
        stdv = 1. / sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)

    # N H W C
    x = nn.functional.interpolate(x, scale_factor=scale_factor)

    # backwards PixelShuffle operation
    # source: https://github.com/pytorch/pytorch/issues/2456
    out_channel = x.shape[1] * (scale_factor ** 2)
    out_h, out_w = (x.shape[i] // scale_factor for i in [2, 3])
    x = x.contiguous().view(x.shape[0], x.shape[1], out_h, scale_factor, out_w, scale_factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(x.shape[0], out_channel, out_h, out_w)

    with no_grad():
        w.copy_(x)


class FeatureExtractor(nn.Module):
    def __init__(self, maxpool_idx=4, no_activation=True):
        """
        Used for computing perception loss (MSE over activations)

        :param maxpool_idx: either 1 or 4
        :param no_activation: flag if activation layer before pooling is also included
        """
        super(FeatureExtractor, self).__init__()

        children = list(vgg19(pretrained=True).children())[0]
        max_pool_indices = [index for index, layer in enumerate(children) if isinstance(layer, nn.MaxPool2d)]

        # get all layers up to chosen maxpool layer
        layer_idx = max_pool_indices[maxpool_idx]

        if no_activation:
            layer_idx -= 1

        layers = children[:layer_idx]
        self.features = nn.Sequential(*layers).requires_grad_(False).eval()

        self.register_buffer("mean_val", tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)))
        self.register_buffer("std_val", tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)))

    def preprocess(self, x):
        return (x - self.mean_val) / self.std_val

    def forward(self, x):
        return self.features(self.preprocess(x))
