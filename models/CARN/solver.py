from configparser import SectionProxy
from torch.nn import MSELoss, L1Loss

from .model import Net

from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg=None, mode="train"):
        super().__init__(cfg, mode)

        if mode == "train":
            nn_config: SectionProxy = cfg['CNN']
            self.device = nn_config['Device']

            self.loss = MSELoss().to(self.device)
        elif mode == "single":
            pass
        else:
            raise Exception(f"Wrong mode \"{mode}\"!")

    def compute_loss(self, label, output, target):
        pixel_loss = self.loss(output, target)

        components = {
            "pixel_loss": pixel_loss.item(),
        }

        return pixel_loss, components

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    @property
    def name(self):
        return "CARN"
