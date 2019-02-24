from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path

from torch import ones, zeros, mean, tensor, cat, sigmoid, log
from torch.nn import MSELoss, L1Loss

from torchvision.models import vgg19

from .model import Generator, Discriminator, FeatureExtractor
from utils import ConfigWrapper

try:
    from ..abstract_gan_solver import AbstractGanSolver
except ValueError:
    # in case module above isn't in pythonpath
    script = getframeinfo(currentframe()).filename
    models_dir = dirname(dirname(abspath(script)))
    path.append(models_dir)
    from abstract_gan_solver import AbstractGanSolver


class Solver(AbstractGanSolver):
    def __init__(self, cfg: ConfigWrapper = None):
        super().__init__(cfg)
        self.mse = MSELoss().to(cfg.device)
        self.l1 = L1Loss().to(cfg.device)
        self.feature_extractor = FeatureExtractor(vgg19(pretrained=True)).to(cfg.device)

        self.ones_const = ones(cfg.batch_size, device=cfg.device)
        self.zeros_const = zeros(cfg.batch_size, device=cfg.device)
        self.d_loss_response = cat((ones(cfg.batch_size, device=cfg.device),
                                    zeros(cfg.batch_size, device=cfg.device)))

        self.epsilon = 0.000001

    @property
    def name(self):
        return "ESRGAN"

    def get_generator_instance(self, *args, **kwargs):
        return Generator(*args, **kwargs)

    def get_discriminator_instance(self, *args, **kwargs):
        return Discriminator()

    def compute_discriminator_loss(self, response: tensor, *args, **kwargs):
        real_response, fake_response = response.split(len(response))

        return -(mean(log(sigmoid(real_response - mean(fake_response, axis=0)) + self.epsilon), axis=0)
                 + mean(log(1 - sigmoid(fake_response - mean(real_response, axis=0)) + self.epsilon), axis=0))

    def compute_generator_loss(self, response: tensor, fake_img: tensor, real_img: tensor, *args, **kwargs):
        real_response, fake_response = response.split(response.size()[0] // 2)

        gen_adv_loss = -(mean(log(sigmoid(fake_response - mean(real_response, axis=0)) + self.epsilon), axis=0)
                         + mean(log(1 - sigmoid(real_response - mean(fake_response, axis=0)) + self.epsilon), axis=0))
        gen_content_loss = self.l1(fake_img, real_img)

        real_features = self.feature_extractor(real_img)
        fake_features = self.feature_extractor(fake_img)

        feature_content_loss = self.mse(real_features, fake_features)
        return gen_content_loss + 0.001 * gen_adv_loss + 0.006 * feature_content_loss
