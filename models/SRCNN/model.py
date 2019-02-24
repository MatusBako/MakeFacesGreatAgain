from math import log2, floor
from torch import nn, cat, add, Tensor, Size
import torch.nn.init as init
from torch.nn.functional import interpolate
from torchvision.transforms import transforms, ToPILImage, ToTensor, Resize, Compose
from PIL.Image import BICUBIC

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
    def __init__(self, scale_factor, num_channels=3, d=64):
        super(Net, self).__init__()

        self.scale_factor = scale_factor

        '''self.layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=9, padding=4,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=d, out_channels=d // 2, kernel_size=5, padding=2,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=d // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=5, padding=2,
                      bias=True),
            nn.PixelShuffle(upscale_factor))'''

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=d, out_channels=d // 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=d // 2, out_channels=num_channels, kernel_size=5, padding=2)
        )

        self.transform = None
        self._initialize_weights()

    def _initialize_weights(self):
        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

    def build_transform(self, shape):
        h, w = shape[2] * self.scale_factor, shape[3] * self.scale_factor
        return Compose([
            ToPILImage(),
            Resize((h, w), interpolation=BICUBIC),
            ToTensor()
          ])

    def forward(self, x: Tensor):
        # workaround for bicubic
        #if self.transform is None:
        self.transform = self.build_transform(x.shape)

        device = x.device

        shape = x.shape[0], x.shape[1], x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor
        new = Tensor(size=shape)
        x = x.cpu()

        for i in range(x.shape[0]):
            new[i] = self.transform(x[i])
        new.requires_grad_()

        return self.layers(new.to(device=device))
