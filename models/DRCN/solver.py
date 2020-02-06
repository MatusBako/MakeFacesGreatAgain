from torch import mean, tensor, sum, cat
from torch.nn import MSELoss

from .model import Net
from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.loss = MSELoss().to(self.device)

        # TODO: architecure specific parameters
        self.alpha = tensor(0.5, device=self.device)
        self.beta = tensor(0.001, device=self.device)

    def compute_loss(self, output, target):
        # loss1

        tmp = tensor(0.)

        # loss_1 = sum(tensor([self.loss(self.net.reconstructed[idx], target) for idx in range(8)]) / 8)
        loss_1 = sum(self.loss(cat(self.net.reconstructed), target.repeat(self.net.num_recursions, 1, 1, 1))) / 8

        # loss2
        loss_2 = self.loss(output, target)

        components = {
            "pixel_loss": loss_2.item(),
            "recurrent_loss": loss_1.item(),
        }

        total_loss = self.alpha * loss_1 + (1 - self.alpha) * loss_2

        return total_loss, components

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    def post_backward(self):
        pass

    @property
    def name(self):
        return "DRCN"
