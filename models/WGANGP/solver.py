from configparser import SectionProxy
import numpy as np

from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path
from torch import ones, zeros, mean, tensor, cat, log, clamp, sigmoid, Tensor
from torch.nn import MSELoss, BCELoss, LogSigmoid, BCEWithLogitsLoss
from torch.autograd import grad, Variable
from torchvision.models import vgg19, densenet201
from torchvision.transforms import Normalize

from .model import Discriminator, FeatureExtractor, Generator
from models.abstract_gan_solver import AbstractGanSolver
Generator = None


class Solver(AbstractGanSolver):
    def __init__(self, nn_config=None):
        super().__init__(nn_config)

        nn_config: SectionProxy = nn_config['GAN']
        self.device = nn_config['Device']

        # need to import Generator dynamically
        model = nn_config['Generator'] if nn_config is not None else self.discriminator_name

        global Generator
        try:
            Generator = __import__("models." + model, fromlist=['Generator']).Generator
        except AttributeError:
            Generator = __import__("models." + model, fromlist=['Net']).Net

        self.mse_loss = MSELoss().to(self.device)
        self.bce_loss = BCEWithLogitsLoss().to(self.device)
        self.feature_extractor = FeatureExtractor().to(self.device)

        self.ones_const = ones(self.batch_size, device=self.device, requires_grad=False)
        self.zeros_const = zeros(self.batch_size, device=self.device, requires_grad=False)
        self.d_loss_response = cat((ones(self.batch_size, device=self.device, requires_grad=False),
                                    zeros(self.batch_size, device=self.device, requires_grad=False)))
        self.zero = zeros(1).to(self.device)

        self.pixel_loss_param = nn_config.getfloat('PixelLossParam', fallback=0.)
        self.adversarial_loss_param = nn_config.getfloat('AdversarialLossParam', fallback=1.)
        self.feature_loss_param = nn_config.getfloat('FeatureLossParam', fallback=1.)
        self.lambda_gp = nn_config.getfloat('GradPenaltyParam', fallback=1.)

    @property
    def discriminator_name(self):
        return "WGANGP"

    def build_generator(self, *args, **kwargs):
        return Generator(*args, **kwargs)

    def build_discriminator(self, *args, **kwargs):
        return Discriminator()

    def compute_gradient_penalty(self, real_img: tensor, fake_img: tensor):
        """Calculates the gradient penalty loss for WGAN GP"""

        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_img.size(0), 1, 1, 1))).to(self.device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_img + ((1 - alpha) * fake_img)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        #fake = Tensor(real.size(0), 1).fill_(1.).requires_grad_(False)

        # Get gradient w.r.t. interpolates
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=self.ones_const[:real_img.size(0)],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def compute_discriminator_loss(self, fake_img, real_img, precomputed=None, train=True, *args, **kwargs):
        fake_response = self.discriminator(fake_img)
        real_response = self.discriminator(real_img)

        gradient_penalty = 0. if not train else self.compute_gradient_penalty(real_img, fake_img)
        return - mean(real_response) + mean(fake_response) + self.lambda_gp * gradient_penalty
        # return self.bce(response, self.d_loss_response)

    def compute_generator_loss(self, label, fake_img: tensor, real_img: tensor, precomputed=None, *args, **kwargs):
        loss = 0.
        components = {}

        if self.pixel_loss_param != 0:
            pixel_loss = self.pixel_loss_param * self.mse(fake_img, real_img)

            loss += pixel_loss
            components["pixel_loss"] = pixel_loss.item()

        pixel_loss = self.pixel_loss_param * self.mse_loss(fake_img, real_img) if self.pixel_loss_param != 0. else self.zero

        if self.adversarial_loss_param != 0:
            fake_response = self.discriminator(fake_img)
            gen_adv_loss = self.adversarial_loss_param * self.bce_loss(fake_response, self.ones_const)

            loss += gen_adv_loss
            components["adv_loss"] = gen_adv_loss.item()

        if self.feature_loss_param != 0:
            fake_features = self.feature_extractor(fake_img)
            real_features = self.feature_extractor(real_img).detach()
            feature_loss = self.feature_loss_param * self.mse_loss(fake_features, real_features)

            loss += feature_loss
            components["feature_loss"] = feature_loss.item()

        return loss, components, None
