from configparser import SectionProxy
from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path

from torch import ones, zeros, mean, tensor, cat, sigmoid, log
from torch.nn import MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss

from .model import Discriminator, FeatureExtractor, Generator

# Generator = None

try:
    from ..abstract_gan_solver import AbstractGanSolver
except ValueError:
    # in case module above isn't in pythonpath
    script = getframeinfo(currentframe()).filename
    models_dir = dirname(dirname(abspath(script)))
    path.append(models_dir)
    from abstract_gan_solver import AbstractGanSolver


class Solver(AbstractGanSolver):
    def __init__(self, cfg=None, batch_size=8):
        super().__init__(cfg)

        nn_config: SectionProxy = cfg['GAN']
        self.device = nn_config['Device']

        # TODO: use this as dynamic gen. import (if so, define Gen on global level)
        # model = nn_config['Generator'] if cfg is not None else self.discriminator_name
        #
        # # need to import Generator dynamically
        # global Generator
        # try:
        #     Generator = __import__("models." + model, fromlist=['Generator']).Generator
        # except AttributeError:
        #     Generator = __import__("models." + model, fromlist=['Net']).Net

        self.mse = MSELoss().to(self.device)
        self.l1 = L1Loss().to(self.device)
        self.bce = BCEWithLogitsLoss().to(self.device)

        self.feature_extractor = FeatureExtractor().to(nn_config['Device'])
        # self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.ones_const = ones(self.batch_size, device=nn_config['Device'], requires_grad=False)
        self.zeros_const = zeros(self.batch_size, device=nn_config['Device'], requires_grad=False)
        self.d_loss_response = cat((ones(self.batch_size, device=nn_config['Device'], requires_grad=False),
                                    zeros(self.batch_size, device=nn_config['Device'], requires_grad=False)))

        # TODO: make readable from config
        self.alpha = 5e-3
        self.eta = 1e-2
        self.eps = 1e-8

    @property
    def discriminator_name(self):
        return "ESRGAN_CUSTOM"

    def build_generator(self, *args, **kwargs):
        return Generator(*args, **kwargs)

    def build_discriminator(self, *args, **kwargs):
        return Discriminator()

    def _relative_loss(self, real: tensor, fake: tensor):
        return sigmoid(real - mean(fake))

    def compute_discriminator_loss(self, response: tensor, real_img, fake_img, *args, **kwargs):
        real_response, fake_response = response.split(len(response) // 2)

        # return - mean(log(sigmoid(real_response - mean(fake_response, 0) + self.epsilon)), 0) \
        #    - mean(log(1 - sigmoid(fake_response - mean(real_response, 0) + self.epsilon)), 0)

        # return - mean(log(self._relative_loss(real_response, fake_response) + self.eps)) \
        #     - mean(log(1 - self._relative_loss(fake_response, real_response) + self.eps))

        # TODO: toto neni RaGAN, ale da sa skusit na SRGAN
        return (self.bce(real_response - fake_response.mean(0, keepdim=True), self.ones_const[:real_response.size(0)]) +
                    self.bce(fake_response - real_response.mean(0, keepdim=True), self.zeros_const[:real_response.size(0)])) / 2

    def compute_generator_loss(self, response: tensor, fake_img: tensor, real_img: tensor, *args, **kwargs):
        real_response, fake_response = response.split(len(response) // 2)

        pixel_loss = self.l1(fake_img, real_img)

        # gen_adv_loss = - mean(log(fake_response - mean(real_response, 0) + self.eps)) \
        #     - mean(log(1 - real_response - mean(fake_response, 0) + self.eps))

        # gen_adv_loss = mean(self.bce(real_response - fake_response.mean(0, keepdim=True), self.ones_const),
        #                     self.bce(fake_response - real_response.mean(0, keepdim=True), self.zeros_const))

        gen_adv_loss = self.bce(fake_response - real_response.mean(0, keepdim=True), self.ones_const[:real_response.size(0)])

        fake_features, real_features = self.feature_extractor(cat((fake_img, real_img), 0)).split(real_response.size(0))

        # feature_content_loss = self.mse(real_features, fake_features)
        feature_loss = self.mse(real_features, fake_features)

        # TODO: add parameters to config
        return 1e-2 * pixel_loss + 1e-3 * gen_adv_loss + 1e-3 * feature_loss, \
               [1e-2 * pixel_loss.item(), 5e-3 * gen_adv_loss.item(), 1e-3 * feature_loss]
        # 0.000000001 * feature_content_loss.item()]
