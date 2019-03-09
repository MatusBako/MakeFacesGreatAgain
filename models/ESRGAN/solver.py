from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path

from torch import ones, zeros, mean, tensor, cat, sigmoid, log
from torch.nn import MSELoss, L1Loss

from torchvision.models import vgg19

from .model import Discriminator, FeatureExtractor
from utils.config import GanConfig
Generator = None

try:
    from ..abstract_gan_solver import AbstractGanSolver
except ValueError:
    # in case module above isn't in pythonpath
    script = getframeinfo(currentframe()).filename
    models_dir = dirname(dirname(abspath(script)))
    path.append(models_dir)
    from abstract_gan_solver import AbstractGanSolver


class Solver(AbstractGanSolver):
    def __init__(self, cfg: GanConfig = None, batch_size=8):
        super().__init__(cfg)

        self.device = 'cpu' if cfg is None else cfg.device
        self.batch_size = batch_size if cfg is None else cfg.batch_size

        # need to import Generator dynamically
        if cfg.generator_module != cfg.discriminator_module:
            global Generator
            try:
                Generator = __import__("models." + cfg.generator_module, fromlist=['Generator']).Generator
            except AttributeError:
                Generator = __import__("models." + cfg.generator_module, fromlist=['Net']).Net

        self.mse = MSELoss().to(self.device)
        self.l1 = L1Loss().to(self.device)

        self.feature_extractor = FeatureExtractor(vgg19(pretrained=True)).to(self.device)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.ones_const = ones(self.batch_size, device=self.device)
        self.zeros_const = zeros(self.batch_size, device=self.device)
        self.d_loss_response = cat((ones(self.batch_size, device=self.device),
                                    zeros(self.batch_size, device=self.device)))

        self.epsilon = 0.000001

    @property
    def discriminator_name(self):
        return "ESRGAN"

    def get_generator_instance(self, *args, **kwargs):
        return Generator(*args, **kwargs)

    def get_discriminator_instance(self, *args, **kwargs):
        return Discriminator()

    def compute_discriminator_loss(self, response: tensor, *args, **kwargs):
        real_response, fake_response = response.split(len(response) // 2)

        return - mean(log(sigmoid(real_response - mean(fake_response, 0) + self.epsilon)), 0) \
            - mean(log(1 - sigmoid(fake_response - mean(real_response, 0) + self.epsilon)), 0)

    def compute_generator_loss(self, response: tensor, fake_img: tensor, real_img: tensor, *args, **kwargs):
        real_response, fake_response = response.split(len(response) // 2)

        gen_adv_loss = - mean(log(sigmoid(fake_response - mean(real_response, 0)) + self.epsilon), 0) \
            - mean(log(1 - sigmoid(real_response - mean(fake_response, 0)) + self.epsilon), 0)
        gen_content_loss = self.l1(fake_img, real_img)

        #fake_img_n = tensor(fake_img, requires_grad=False)
        #real_img_n = tensor(real_img, requires_grad=False)

        #for c in range(3):
        #    fake_img_n[:, c, :, :] = (fake_img_n[:, c, :, :] - self.mean[c]) / self.std[c]
        #    real_img_n[:, c, :, :] = (real_img_n[:, c, :, :] - self.mean[c]) / self.std[c]

        #fake_features = self.feature_extractor(fake_img_n)
        #real_features = self.feature_extractor(real_img_n)
        #feature_content_loss = self.mse(real_features, fake_features)

        # + 0.000000001 * feature_content_loss, \
        return 0.001 * gen_content_loss + 0.0001 * gen_adv_loss, \
            [0.001 * gen_content_loss.item(), 0.0001 * gen_adv_loss.item(), 0]  # 0.000000001 * feature_content_loss.item()]
