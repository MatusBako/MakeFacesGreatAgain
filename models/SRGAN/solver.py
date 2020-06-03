from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path

from torch import ones, zeros, mean, tensor, cat, log, clamp, sigmoid
from torch.nn import MSELoss, BCELoss

from torchvision.transforms import Normalize

from .model import Generator, Discriminator, FeatureExtractor
from models.abstract_gan_solver import AbstractGanSolver
Generator = None


class Solver(AbstractGanSolver):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        nn_config = cfg['GAN']
        self.device = nn_config['Device']

        model = nn_config['Generator'] if cfg is not None else self.discriminator_name

        global Generator
        try:
            Generator = __import__("models." + model, fromlist=['Generator']).Generator
        except AttributeError:
            Generator = __import__("models." + model, fromlist=['Net']).Net

        self.mse_loss: MSELoss = MSELoss().to(self.device)
        self.bce_loss = BCELoss().to(self.device)

        self.feature_extractor = FeatureExtractor().to(self.device)

        self.ones_const = ones(self.batch_size, device=self.device, requires_grad=False)
        self.zeros_const = zeros(self.batch_size, device=self.device, requires_grad=False)

        self.pixel_loss_param = nn_config.getfloat('PixelLossParam', fallback=0)
        self.adversarial_loss_param = nn_config.getfloat('AdversarialLossParam', fallback=0)
        self.feature_loss_param = nn_config.getfloat('FeatureLossParam', fallback=0)

        self.zero = zeros(1).to(self.device)

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

        return self.bce_loss(fake_response, self.zeros_const) + self.bce_loss(real_response, self.ones_const)
        # return self.bce(response, self.d_loss_response)

    def compute_generator_loss(self, label, fake_img: tensor, real_img: tensor, precomputed=None, *args, **kwargs):
        pixel_loss = self.pixel_loss_param * self.mse_loss(fake_img, real_img) if self.pixel_loss_param != 0. else self.zero

        if self.adversarial_loss_param != 0:
            fake_response = self.discriminator(fake_img)
            gen_adv_loss = self.adversarial_loss_param * self.bce_loss(fake_response, self.ones_const)
        else:
            gen_adv_loss = self.zero

        if self.feature_loss_param != 0:
            fake_features = self.feature_extractor(fake_img)
            real_features = self.feature_extractor(real_img).detach()
            feature_loss = self.feature_loss_param * self.mse_loss(fake_features, real_features)
        else:
            feature_loss = self.zero

        components = {
            "pixel_loss": pixel_loss.item(),
            "adv_loss": gen_adv_loss.item(),
            "feature_loss": feature_loss.item()
        }

        # TODO: koeficienty
        return pixel_loss + gen_adv_loss + feature_loss, components, None