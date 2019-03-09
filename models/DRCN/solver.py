from torch.nn import MSELoss

from .model import Net
from utils.config import CnnConfig

from numpy import mean
from os.path import dirname, abspath
from inspect import getframeinfo, currentframe
from sys import path

try:
    from ..abstract_cnn_solver import AbstractCnnSolver
except ValueError:
    # in case module above isn't in pythonpath
    script = getframeinfo(currentframe()).filename
    models_dir = dirname(dirname(abspath(script)))
    path.append(models_dir)
    from abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg: CnnConfig = None):
        super().__init__(cfg)
        self.loss = MSELoss()

        # TODO: architecure specific parameters
        self.loss_alpha = 1.0
        self.loss_beta = 0.001

    def compute_loss(self, output, target):
        # loss1
        loss_1 = 0
        for d in range(8):
            loss_1 += (self.loss(output[d], target) / 8)

        # loss2
        loss_2 = self.loss(output, target)

        # regularization
        reg_term = 0
        for theta in self.net.parameters():
            reg_term += mean(sum(theta ** 2))

        # total loss
        total_loss = self.loss_alpha * loss_1 + (1 - self.loss_alpha) * loss_2 + self.loss_beta * reg_term

        return total_loss

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    @property
    def name(self):
        return "DRCN"
