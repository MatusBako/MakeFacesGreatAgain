from torch.nn import MSELoss

from .model import Net
from utils import ConfigWrapper

from os.path import dirname, abspath
from inspect import getframeinfo, currentframe
from sys import path

from torch import device, cuda

try:
    from ..abstract_cnn_solver import AbstractCnnSolver
except ValueError:
    # in case module above isn't in pythonpath
    script = getframeinfo(currentframe()).filename
    models_dir = dirname(dirname(abspath(script)))
    path.append(models_dir)
    from abstract_cnn_solver import AbstractCnnSolver

class Solver(AbstractCnnSolver):
    def __init__(self, cfg: ConfigWrapper = None):
        super().__init__(cfg)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.loss = MSELoss()

    def compute_loss(self, output, target):
        return self.loss(output, target)

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    @property
    def name(self):
        return "SRCNN"