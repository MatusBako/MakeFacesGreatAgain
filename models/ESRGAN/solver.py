from configparser import SectionProxy

from torch import ones, zeros, mean, tensor, cat, log, sigmoid
from torch.nn import MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss

from torchvision.models import vgg19

from models.abstract_gan_solver import AbstractGanSolver
from .model import Discriminator, FeatureExtractor, Generator

Generator = None


class Solver(AbstractGanSolver):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        nn_config: SectionProxy = cfg['GAN']
        self.device = nn_config['Device']

        # TODO: use this as dynamic gen. import (if so, define Gen on global level)
        model = nn_config['Generator'] if cfg is not None else self.discriminator_name

        # need to import Generator dynamically
        global Generator
        try:
            Generator = __import__("models." + model, fromlist=['Generator']).Generator
        except AttributeError:
            Generator = __import__("models." + model, fromlist=['Net']).Net

        self.mse = MSELoss().to(self.device)
        self.l1 = L1Loss().to(self.device)
        self.bcewl = BCEWithLogitsLoss().to(self.device)
        self.bce_loss = BCELoss().to(self.device)

        self.feature_extractor = FeatureExtractor().to(self.device)

        self.ones_const = ones(self.batch_size, device=nn_config['Device'], requires_grad=False)
        self.zeros_const = zeros(self.batch_size, device=nn_config['Device'], requires_grad=False)

        # TODO: make readable from config
        self.pixel_loss_param = nn_config.getfloat('PixelLossParam', fallback=0)
        self.adversarial_loss_param = nn_config.getfloat('AdversarialLossParam', fallback=0)
        self.feature_loss_param = nn_config.getfloat('FeatureLossParam', fallback=0)

        self.zero = zeros(1).to(self.device)

    @property
    def discriminator_name(self):
        return "ESRGAN"

    def build_generator(self, *args, **kwargs):
        return Generator(*args, **kwargs)

    def build_discriminator(self, *args, **kwargs):
        return Discriminator()

    def compute_discriminator_loss(self, fake_img, real_img, precomputed=None, train=True, *args, **kwargs):
        fake_response = self.discriminator(fake_img.detach())
        real_response = self.discriminator(real_img)

        l1 = self.bcewl(real_response - fake_response.mean(0, keepdim=True), self.ones_const[:real_response.size(0)])
        l2 = self.bcewl(fake_response - real_response.mean(0, keepdim=True), self.zeros_const[:real_response.size(0)])
        return l1 + l2

    def compute_generator_loss(self, label, fake_img: tensor, real_img: tensor, precomputed=None, *args, **kwargs):
        pixel_loss = self.pixel_loss_param * self.l1(fake_img, real_img) \
            if self.pixel_loss_param != 0. \
            else self.zero

        if self.adversarial_loss_param != 0:
            fake_response = self.discriminator(fake_img)
            real_response = self.discriminator(real_img).detach()

            gen_adv_loss = self.adversarial_loss_param * self.bcewl(fake_response - real_response.mean(0, keepdim=True),
                                                                    self.ones_const[:fake_response.size(0)])
        else:
            gen_adv_loss = self.zero

        if self.feature_loss_param != 0:
            fake_features = self.feature_extractor(fake_img)
            real_features = self.feature_extractor(real_img).detach()
            feature_loss = self.feature_loss_param * self.mse(fake_features, real_features)
        else:
            feature_loss = self.zero

        components = {
            "pixel_loss": pixel_loss.item(),
            "adv_loss": gen_adv_loss.item(),
            "feature_loss": feature_loss.item()
        }

        # TODO: add parameters to config
        return pixel_loss + gen_adv_loss + feature_loss, components, None
