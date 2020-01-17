from configparser import SectionProxy
import numpy as np

from inspect import getframeinfo, currentframe
from os.path import dirname, abspath
from sys import path
from torch import ones, zeros, mean, tensor, cat, log, clamp, sigmoid, Tensor
from torch.nn import MSELoss, BCELoss
from torch.autograd import grad, Variable
from torchvision.models import vgg19, densenet201
from torchvision.transforms import Normalize

from .model import Discriminator, FeatureExtractor, Generator
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

        self.mse = MSELoss().to(self.device)
        self.bce = BCELoss().to(self.device)
        self.feature_extractor = FeatureExtractor().to(self.device)

        # values from docs
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.ones_const = ones(self.batch_size, device=nn_config['Device'], requires_grad=False)
        self.zeros_const = zeros(self.batch_size, device=nn_config['Device'], requires_grad=False)
        self.d_loss_response = cat((ones(self.batch_size, device=nn_config['Device'], requires_grad=False),
                                    zeros(self.batch_size, device=nn_config['Device'], requires_grad=False)))

        # TODO: put parameter into config
        self.lambda_gp = 10

    @property
    def discriminator_name(self):
        return "SRGAN"

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

    def compute_discriminator_loss(self, response: tensor, real_img, fake_img, *args, **kwargs):
        real_response, fake_response = response.split(response.size()[0] // 2)

        gradient_penalty = self.compute_gradient_penalty(real_img, fake_img)
        return - mean(real_response) + mean(fake_response) + self.lambda_gp * gradient_penalty
        # return self.bce(response, self.d_loss_response)

    def compute_generator_loss(self, response: tensor, fake_img: tensor, real_img: tensor, *args, **kwargs):
        real_response, fake_response = response.split(response.size()[0] // 2)

        return - fake_response.mean(), [0, fake_response.mean(), 0]

        gen_content_loss = self.mse(fake_img, real_img)

        gen_adv_loss = self.bce(fake_response, self.ones_const)

        fake_features, real_features = self.feature_extractor(cat((fake_img, real_img), 0)).split(real_response.size(0))
        feature_content_loss = self.mse(real_features, fake_features)

        # 0.0000001 * feature_content_loss, \
        return 1e-3 * gen_content_loss + 1e-4 * gen_adv_loss + 1e-3 * feature_content_loss, \
               [1e-3 * gen_content_loss.item(), 1e-4 * gen_adv_loss.item(), 1e-3 * feature_content_loss.item()]#0.0000001 * feature_content_loss.item()]
