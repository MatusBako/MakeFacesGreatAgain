from configparser import SectionProxy
from torch.nn import MSELoss

from .model import Net, FeatureExtractor
from models.abstract_cnn_solver import AbstractCnnSolver
from feature_extractor import Senet50FeatureExtractor


class Solver(AbstractCnnSolver):
    def __init__(self, cfg=None):
        super().__init__(cfg)

        nn_cfg: SectionProxy = cfg["CNN"]
        self.mse_loss = MSELoss().to(self.device)
        self.feature_extractor = Senet50FeatureExtractor(nn_cfg["DetectionsPath"], nn_cfg["WeightsPath"]).to(self.device)

        self.alpha = nn_cfg.getfloat("Alpha")
        self.beta = nn_cfg.getfloat("Beta")

    def compute_loss(self, label, output, target):
        pixel_loss = self.alpha * self.mse_loss(output, target)

        fake_features = self.feature_extractor(label, output)
        real_features = self.feature_extractor(label, target).detach()
        identity_loss = self.beta * self.mse_loss(fake_features, real_features)

        components = {
            "pixel_loss": pixel_loss.item(),
            "identity_loss": identity_loss.item()
        }

        return pixel_loss + identity_loss, components

    def get_net_instance(self, *args, **kwargs):
        return Net(*args, **kwargs)

    @property
    def name(self):
        return "MFGNSI"
