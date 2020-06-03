from torch.nn import MSELoss, Parameter

from .model import Net
from models.abstract_cnn_solver import AbstractCnnSolver


class Solver(AbstractCnnSolver):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.loss = MSELoss().to(self.device)
        self.theta = 1e-2

    def compute_loss(self, label, output, target):
        pixel_loss = self.loss(output, target)

        components = {
            "pixel_loss": pixel_loss.item(),
        }

        return pixel_loss, components

    def get_net_instance(self, *args, **kwargs):
        return Net(base_channels=64, num_residuals=20, *args, **kwargs)

    def post_backward(self):
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        clip_val = self.theta / lr

        for p in self.net.parameters():
            p.grad.data.clamp_(min=-clip_val, max=clip_val)

        # alternative: torch.nn.utils.clip_grad_norm(self.net.parameters(),clip)

    @property
    def name(self):
        return "VDSR"
