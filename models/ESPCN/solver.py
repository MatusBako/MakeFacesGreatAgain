from torch.nn import MSELoss

from .model import Net
from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.mse_loss = MSELoss().to(self.device)

    def compute_loss(self, label, output, target):
        pixel_loss = self.mse_loss(output, target)

        components = {
            "pixel_loss": pixel_loss.item(),
        }

        return pixel_loss, components

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    @property
    def name(self):
        return "ESPCN"
