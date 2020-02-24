from torch import nn, cat, add, Tensor, tensor
from torch.nn import init, Upsample, Conv2d, ReLU, AvgPool2d, BatchNorm2d, Sequential, LeakyReLU
from torch.nn.functional import interpolate
from torchvision.models import vgg19


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(tensor([init_value]), requires_grad=True)

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


class FeatureExtractor(nn.Module):
    def __init__(self, maxpool_idx=1, no_activation=False):
        """
        Used for computing perception loss (MSE over activations)

        :param maxpool_idx: either 1 or 4
        :param no_activation: flag if last activation is included
        """
        super(FeatureExtractor, self).__init__()

        children = list(vgg19(pretrained=True).children())[0]
        max_pool_indices = [index for index, layer in enumerate(children) if isinstance(layer, nn.MaxPool2d)]

        # get all layers up to chosen maxpool layer
        maxpool_idx = max_pool_indices[maxpool_idx]
        layers = children[:maxpool_idx - int(no_activation)]
        self.features = nn.Sequential(*layers).requires_grad_(False).eval()

        self.register_buffer("mean_val", tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)))
        self.register_buffer("std_val", tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)))

    def preprocess(self, x):
        return (x - self.mean_val) * self.std_val

    def forward(self, x):
        return self.features(self.preprocess(x))
