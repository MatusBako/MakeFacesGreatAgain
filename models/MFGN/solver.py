from torch.nn import L1Loss


from .model import Net
from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg=None, mode="train"):
        super().__init__(cfg, mode)

        if mode == "train":
            self.loss = L1Loss()
            self.pixel_loss_param = cfg['CNN'].getfloat('PixelLossParam', fallback=1)

    def compute_loss(self, label, output, target):
        pixel_loss = self.pixel_loss_param * self.loss(output, target)

        components = {
            "pixel_loss": pixel_loss.item(),
        }

        return pixel_loss, components

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    @property
    def name(self):
        return "MFGN"
