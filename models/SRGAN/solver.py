from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path

from torch import ones, zeros, mean, tensor, cat, log, clamp
from torch.nn import MSELoss, BCELoss

from torchvision.models import vgg19
from torchvision.transforms import Normalize

from .model import Generator, Discriminator, FeatureExtractor
from utils.config import CnnConfig

try:
    from ..abstract_gan_solver import AbstractGanSolver
except ValueError:
    # in case module above isn't in pythonpath
    script = getframeinfo(currentframe()).filename
    models_dir = dirname(dirname(abspath(script)))
    path.append(models_dir)
    from abstract_gan_solver import AbstractGanSolver


class Solver(AbstractGanSolver):
    def __init__(self, cfg: CnnConfig = None):
        super().__init__(cfg)
        self.device = cfg.device

        self.mse = MSELoss().to(cfg.device)
        self.bce = BCELoss().to(cfg.device)

        self.feature_extractor = FeatureExtractor(vgg19(pretrained=True)).to(cfg.device)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.ones_const = ones(cfg.batch_size, device=cfg.device)
        self.zeros_const = zeros(cfg.batch_size, device=cfg.device)
        self.d_loss_response = cat((ones(cfg.batch_size, device=cfg.device),
                                    zeros(cfg.batch_size, device=cfg.device)))

    @property
    def discriminator_name(self):
        return "SRGAN"

    def get_generator_instance(self, *args, **kwargs):
        return Generator(*args, **kwargs)

    def get_discriminator_instance(self, *args, **kwargs):
        return Discriminator()

    def compute_discriminator_loss(self, response: tensor, *args, **kwargs):
        real_response, fake_response = response.split(response.size()[0] // 2)
        return mean(- log(real_response) - log(1 - fake_response))
        #return self.bce(response, self.d_loss_response)

    def compute_generator_loss(self, response: tensor, fake_img: tensor, real_img: tensor, *args, **kwargs):
        real_response, fake_response = response.split(response.size()[0] // 2)

        #gen_adv_loss = self.bce(fake_response, self.ones_const)
        gen_adv_loss = mean(-log(fake_response))
        gen_content_loss = self.mse(fake_img, real_img)

        fake_img_n = tensor(fake_img, requires_grad=False)
        real_img_n = tensor(real_img, requires_grad=False)

        for c in range(3):
            fake_img_n[:, c, :, :] = (fake_img_n[:, c, :, :] - self.mean[c]) / self.std[c]
            real_img_n[:, c, :, :] = (real_img_n[:, c, :, :] - self.mean[c]) / self.std[c]

        fake_features = self.feature_extractor(fake_img_n)
        real_features = self.feature_extractor(real_img_n)
        feature_content_loss = self.mse(real_features, fake_features)

        return gen_content_loss + 0.0001 * gen_adv_loss + 0.0000000001 * feature_content_loss, \
            [gen_content_loss.item(), 0.0001 * gen_adv_loss.item(), 0.0000000001 * feature_content_loss.item()]
