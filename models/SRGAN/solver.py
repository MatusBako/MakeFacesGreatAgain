from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path

from torch import ones, zeros, mean, tensor, cat, log, clamp, sigmoid
from torch.nn import MSELoss, BCELoss

from torchvision.transforms import Normalize

from .model import Generator, Discriminator, FeatureExtractor

try:
    from ..abstract_gan_solver import AbstractGanSolver
except ValueError:
    # in case module above isn't in pythonpath
    script = getframeinfo(currentframe()).filename
    models_dir = dirname(dirname(abspath(script)))
    path.append(models_dir)
    from abstract_gan_solver import AbstractGanSolver


class Solver(AbstractGanSolver):
    def __init__(self, cfg=None):
        super().__init__(cfg)

        nn_config = cfg['GAN']
        self.device = nn_config['Device']

        self.mse_loss: MSELoss = MSELoss().to(nn_config['Device'])
        self.bce_loss = BCELoss().to(nn_config['Device'])

        self.feature_extractor = FeatureExtractor().to(nn_config['Device'])

        self.mean = tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).to(nn_config['Device'])
        self.std = tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).to(nn_config['Device'])
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.ones_const = ones(self.batch_size, device=nn_config['Device'], requires_grad=False)
        self.zeros_const = zeros(self.batch_size, device=nn_config['Device'], requires_grad=False)

        self.eps = 1e-8

    @property
    def discriminator_name(self):
        return "SRGAN"

    def build_generator(self, *args, **kwargs) -> Generator:
        return Generator(*args, **kwargs)

    def build_discriminator(self, *args, **kwargs) -> Discriminator:
        return Discriminator()

    def post_backward_generator(self):
        pass

    def post_backward_discriminator(self):
        pass

    def compute_discriminator_loss(self, fake_img, real_img, precomputed=None, train=True, *args, **kwargs):
        fake_response = self.discriminator(fake_img)
        real_response = self.discriminator(real_img)

        return (self.bce_loss(fake_response, self.zeros_const) +
                self.bce_loss(real_response, self.ones_const)) / 2
        # return self.bce(response, self.d_loss_response)

    def compute_generator_loss(self, label, fake_img: tensor, real_img: tensor, precomputed=None, *args, **kwargs):
        fake_response = self.discriminator(fake_img)

        gen_adv_loss = self.bce_loss(fake_response, self.ones_const)

        fake_features = self.feature_extractor(fake_img)
        real_features = self.feature_extractor(real_img).detach()
        feature_content_loss = 0.1 * self.mse_loss(fake_features, real_features)

        # TODO: koeficienty
        return gen_adv_loss + feature_content_loss, \
               [0, gen_adv_loss.item(), feature_content_loss.item()], \
               None