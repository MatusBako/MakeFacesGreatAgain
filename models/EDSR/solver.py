from torch.nn import L1Loss

from .model import Net
from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.loss = L1Loss().to(self.device)

    def compute_loss(self, label, output, target):
        pixel_loss = self.loss(output, target)

        components = {
            "pixel_loss": pixel_loss.item(),
        }

        return pixel_loss, components

    def get_net_instance(self, *args, **kwargs):
        """
        Network size is limited due to training on GTX 1080
        """
        return Net(base_channel=160, num_residuals=32, *args, **kwargs)

    @property
    def name(self):
        return "EDSR"
