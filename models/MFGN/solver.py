from torch.nn import MSELoss

from .model import Net
from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.loss = MSELoss()

    def compute_loss(self, label, output, target):
        return self.loss(output, target), {}

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    @property
    def name(self):
        return "MFGN"
