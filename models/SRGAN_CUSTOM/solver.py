from configparser import SectionProxy
from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path

from torch import ones, zeros, mean, tensor, cat, log, clamp, sigmoid
from torch.nn import MSELoss, BCELoss, Module

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

        nn_config: SectionProxy = cfg['GAN']
        self.device = nn_config['Device']

        self.mse = MSELoss().to(nn_config['Device'])
        #self.bce = BCELoss().to(nn_config['Device'])

        self.feature_extractor: Module = FeatureExtractor().to(nn_config['Device'])
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.ones_const = ones(self.batch_size, device=nn_config['Device'])
        self.zeros_const = zeros(self.batch_size, device=nn_config['Device'])
        self.d_loss_response = cat((ones(self.batch_size, device=nn_config['Device']),
                                    zeros(self.batch_size, device=nn_config['Device'])))


        self.eps = 1e-8

    @property
    def discriminator_name(self):
        return "SRGAN_CUSTOM"

    def build_generator(self, *args, **kwargs) -> Generator:
        return Generator(*args, **kwargs)

    def build_discriminator(self, *args, **kwargs) -> Discriminator:
        return Discriminator()

    def compute_discriminator_loss(self, response: tensor, *args, **kwargs) -> tensor:
        real_response, fake_response = sigmoid(response).split(response.size(0) // 2)
        return mean(- log(real_response + self.eps) - log(1 - fake_response + self.eps))
        #return self.bce(response, self.d_loss_response)

    def compute_generator_loss(self, response: tensor, fake_img: tensor, real_img: tensor, *args, **kwargs):
        real_response, fake_response = sigmoid(response).split(response.size(0) // 2)

        # not used
        #gen_pixel_loss = self.mse(fake_img, real_img)

        # can be BCE used like this?
        #gen_adv_loss = self.bce(fake_response, self.ones_const)
        gen_adv_loss = mean(-log(fake_response + self.eps))

        # security through obscurity
        fake_features, real_features = self.feature_extractor(cat((fake_img, real_img), 0)).split(real_response.size(0))

        gen_content_loss = self.mse(real_features, fake_features)

        #TODO: koeficienty
        # 0.0000000001 * feature_content_loss,
        return 0.01 * gen_adv_loss + 0.001 * gen_content_loss, \
            [0, 0.01 * gen_adv_loss.item(), 0.001 * gen_content_loss.item()]  # 0.0000000001 * feature_content_loss.item()]
