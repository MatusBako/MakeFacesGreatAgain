from torch import nn,tensor
from torchvision.models import vgg19


class VggFeatureExtractor(nn.Module):
    def __init__(self, maxpool_idx=2, no_activation=True):
        """
        Used for computing perception loss (MSE over activations)

        :param maxpool_idx: either 1 or 4
        :param no_activation: flag if last activation is included
        """
        super(VggFeatureExtractor, self).__init__()

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
