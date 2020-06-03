from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path

import numpy as np

from torch import ones, zeros, mean, tensor, cat, log, clamp, sigmoid, dist, Tensor
from torch.nn import MSELoss, BCELoss, BCEWithLogitsLoss, Module, L1Loss
from torch.autograd import grad
from torch.nn.functional import normalize

from torchvision.models import vgg19
from torchvision.transforms import Normalize

from .model import Discriminator, FeatureExtractor, Generator
from models.abstract_gan_solver import AbstractGanSolver
from feature_extractor import Senet50FeatureExtractor

Generator = None


class Solver(AbstractGanSolver):
    def __init__(self, cfg):
        super().__init__(cfg)

        nn_config = cfg['GAN']

        self.mse = MSELoss().to(self.device)
        self.l1 = L1Loss().to(self.device)
        self.bcewl = BCEWithLogitsLoss().to(self.device)

        # TODO: use this as dynamic gen. import (if so, define Gen on global level)
        model = nn_config['Generator'] if cfg is not None else self.discriminator_name

        global Generator
        try:
            Generator = __import__("models." + model, fromlist=['Generator']).Generator
        except AttributeError:
            Generator = __import__("models." + model, fromlist=['Net']).Net

        self.ones_const = ones(self.batch_size, device=self.device)
        self.zeros_const = zeros(self.batch_size, device=self.device)
        self.d_loss_response = cat((ones(self.batch_size, device=self.device),
                                    zeros(self.batch_size, device=self.device)))

        self.pixel_loss_param = nn_config.getfloat('PixelLossParam', fallback=0)
        self.adversarial_loss_param = nn_config.getfloat('AdversarialLossParam', fallback=0)
        self.feature_loss_param = nn_config.getfloat('FeatureLossParam', fallback=0)
        self.variance_loss_param = nn_config.getfloat('VarianceLossParam', fallback=0)
        self.identity_loss_param = nn_config.getfloat("IdentityLossParam", fallback=0)
        self.gradient_penalty_param = nn_config.getfloat("GradientPenaltyParam", fallback=0)

        if self.feature_loss_param > 0:
            self.feature_extractor = FeatureExtractor().to(self.device)

        if self.identity_loss_param > 0:
            self.identity_extractor = Senet50FeatureExtractor(
                "/run/media/hacky/DATA2/FFHQ/mtcnn_detections_ffhq.pkl",
                "/home/hacky/datasets/VGG2/senet50_ft_pytorch/senet50_ft_dims_2048.pth"
            ).to(self.device)

        self.zero = zeros(1).to(self.device)

    @property
    def discriminator_name(self):
        return "MFGAn_Basic"

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
        # fake = Tensor(real.size(0), 1).fill_(1.).requires_grad_(False)

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
        fake_response = self.discriminator(fake_img.detach())
        real_response = self.discriminator(real_img)

        gradient_penalty = 0. \
            if not train or self.gradient_penalty_param == 0 \
            else self.compute_gradient_penalty(real_img, fake_img)

        # response = cat((real_response, fake_response))
        # return self.bcewl(response, self.d_loss_response)
        # return mean(- log(sigmoid(real_response) + self.epsilon), 0) \
        #     - mean(log(1 - sigmoid(fake_response) + self.epsilon), 0)
        return self.bcewl(real_response - fake_response.mean(0, keepdim=True),
                          self.ones_const[:real_response.size(0)]) \
               + self.bcewl(fake_response - real_response.mean(0, keepdim=True),
                            self.zeros_const[:real_response.size(0)]) + gradient_penalty

    def compute_generator_loss(self, label, fake_img: tensor, real_img: tensor, precomputed=None, *args, **kwargs):
        loss = 0.
        components = {}

        if self.pixel_loss_param != 0:
            pixel_loss = self.pixel_loss_param * self.l1(fake_img, real_img)

            loss += pixel_loss
            components["pixel_loss"] = pixel_loss.item()

        if self.adversarial_loss_param != 0:
            fake_response = self.discriminator(fake_img)
            real_response = self.discriminator(real_img).detach()

            # adversarial_loss = self.adversarial_loss_param * self.bcewl(fake_response, self.ones_const)
            adversarial_loss = self.adversarial_loss_param * self.bcewl(
                fake_response - real_response.mean(0, keepdim=True),
                self.ones_const[:fake_response.size(0)])

            loss += adversarial_loss
            components["adv_loss"] = adversarial_loss.item()

        if self.feature_loss_param > 0:
            fake_features = self.feature_extractor(fake_img)
            real_features = self.feature_extractor(real_img)
            feature_loss = self.feature_loss_param * self.mse(real_features, fake_features)

            loss += feature_loss
            components["feature_loss"] = feature_loss.item()

        if self.variance_loss_param != 0:
            var_loss = self.variance_loss_param * compute_variance_loss(fake_img)

            loss += var_loss
            components["variance_loss"] = var_loss.item()

        if self.identity_loss_param > 0:
            fake_identities = self.identity_extractor(label, fake_img)
            real_identities = self.identity_extractor(label, real_img).detach()

            norm_fake = normalize(fake_identities, p=2, dim=1)
            norm_real = normalize(real_identities, p=2, dim=1)
            # compute l2 distance in hyperball metric space
            identity_loss = self.identity_loss_param * (norm_fake - norm_real).pow(2).sum(1).mean()

            loss += identity_loss
            components["identity_loss"] = identity_loss.item()

        return loss, components, None


def compute_variance_loss(x: Tensor):
    n, c, h, w = x.shape
    num_count_h = n * c * (h - 1) * w
    num_count_w = n * c * h * (w - 1)

    # h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
    # w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()

    h_tv = dist(x[:, :, 1:, :], x[:, :, :-1, :], p=1)
    w_tv = dist(x[:, :, :, 1:], x[:, :, :, :-1], p=1)

    return h_tv / num_count_h + w_tv / num_count_w
