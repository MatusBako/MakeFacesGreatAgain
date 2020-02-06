from configparser import SectionProxy
from torch.nn import MSELoss

from .model import Net

from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg=None):
        super().__init__(cfg)

        nn_config: SectionProxy = cfg['CNN']
        self.device = nn_config['Device']

        self.loss = MSELoss().to(self.device)

    def compute_loss(self, output, target):
        pixel_loss = self.loss(output, target)

        components = {
            "pixel_loss": pixel_loss.item(),
        }

        return pixel_loss, components

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    def post_backward(self):
        pass

    @property
    def name(self):
        return "CARN"
