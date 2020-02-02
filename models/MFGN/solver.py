from torch.nn import MSELoss

from .model import Net

from os.path import dirname, abspath
from inspect import getframeinfo, currentframe
from sys import path

try:
    from ..abstract_cnn_solver import AbstractCnnSolver
except ValueError:
    # in case module above isn't in pythonpath
    script = getframeinfo(currentframe()).filename
    root_dir = dirname(dirname(dirname(abspath(script))))
    path.append(root_dir)
    from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.loss = MSELoss()

    def compute_loss(self, output, target):
        return self.loss(output, target)

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    @property
    def name(self):
        return "MFGN4"
