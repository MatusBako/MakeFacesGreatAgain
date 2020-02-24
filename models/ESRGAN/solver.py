from configparser import SectionProxy

from torch import ones, zeros, mean, tensor, cat, sigmoid, log
from torch.nn import MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss

from torchvision.models import vgg19

from models.abstract_gan_solver import AbstractGanSolver
from .model import Discriminator, FeatureExtractor, Generator
Generator = None


class Solver(AbstractGanSolver):
    def __init__(self, cfg=None, batch_size=8):
        super().__init__(cfg)

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
        self.bcewl = BCEWithLogitsLoss().to(nn_config['Device'])

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
        return (l1 + l2) / 2

    def compute_generator_loss(self, label, fake_img: tensor, real_img: tensor, precomputed=None, *args, **kwargs):
        fake_response = self.discriminator(fake_img)
        real_response = self.discriminator(real_img).detach()

        pixel_loss = 1e-2 * self.l1(fake_img, real_img)

        gen_adv_loss = 1e-3 * self.bcewl(fake_response - real_response.mean(0, keepdim=True),
                                         self.ones_const[:fake_response.size(0)])

        fake_features = self.feature_extractor(fake_img)
        real_features = self.feature_extractor(real_img).detach()
        feature_loss = 1e-3 * self.mse(fake_features, real_features)

        # TODO: add parameters to config
        return pixel_loss + gen_adv_loss + feature_loss, \
               [pixel_loss.item(), gen_adv_loss.item(), feature_loss], \
               None