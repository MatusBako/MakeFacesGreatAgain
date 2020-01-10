import dlib
import numpy as np

from configparser import ConfigParser, SectionProxy
from datetime import datetime
from inspect import getsource, getframeinfo, currentframe
from numpy import log10
from os import mkdir
from os.path import dirname, abspath
from PIL import Image
from sys import exc_info, stderr, path
from time import time
from typing import Tuple, Optional

from torch import load, save, no_grad, device, optim, cuda
from torch.nn import MSELoss, Module
from torch.nn.functional import interpolate
from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop
from torch.utils.data import DataLoader

from factory import build_feature_extractor, build_scheduler, build_optimizer

try:
    from utils import Drawer, Logger
    from utils.config import CnnConfig
except ModuleNotFoundError:
    script = getframeinfo(currentframe()).filename
    root_dir = dirname((abspath(script)))
    path.append(root_dir)
    from utils import Drawer, Logger
    from utils.config import CnnConfig

from abc import ABC, abstractmethod


class AbstractCnnSolver(ABC):
    def __init__(self, config: ConfigParser = None):

        self.iteration = 0
        self.learning_rate = None
        self.net = None
        self.optimizer = None
        self.scheduler = None

        self.output_folder = None
        self.drawer = None
        self.logger = None
        self._cfg = None
        self.test_iter = None
        self.upscale_factor = None

        self.device = device("cuda" if cuda.is_available() else "cpu")

        if config is not None:
            self._cfg = config
            nn_cfg: SectionProxy = config["CNN"]

            self.device = device(nn_cfg['Device'])
            self.batch_size = nn_cfg.getint('BatchSize')
            self.upscale_factor = nn_cfg.getint('UpscaleFactor')
            self.learning_rate = nn_cfg.getfloat('LearningRate')

            self.iter_limit = nn_cfg.getint('IterationLimit')
            self.iter_per_snapshot = nn_cfg.getint('IterationsPerSnapshot')
            self.iter_per_image = nn_cfg.getint('IterationsPerImage')
            self.iter_to_eval = nn_cfg.getint('IterationsToEvaluation')
            self.test_iter = nn_cfg.getint('EvaluationIterations')

            timestamp = str(datetime.fromtimestamp(time()).strftime('%Y.%m.%d-%H:%M:%S'))
            self.output_folder = nn_cfg['OutputFolder'] + "/" + self.name + "-" + timestamp

            self.identity_extractor = build_feature_extractor(config['FeatureExtractor'])

        self.mse = MSELoss()

        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed(self.seed)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_net_instance(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, output, target):
        pass

    def build_models(self):
        self.net: Module = self.get_net_instance(self.upscale_factor).to(self.device)
        self.optimizer = build_optimizer(self._cfg['Optimizer'], self.net.parameters(), self.learning_rate)
        self.scheduler = build_scheduler(self._cfg['Scheduler'], self.optimizer)

    def save_model(self):
        checkpoint_path = self.output_folder + "/" + self.name + "-" + str(self.iteration) + ".mdl"
        dic = {
            'model_name': self.name,
            'upscale': self.upscale_factor,
            'iteration': self.iteration,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        save(dic, checkpoint_path)
        self.net.to(self.device)

    def load_model(self, model_path: str):
        state = load(model_path)

        if state['model_name'] != self.name:
            raise Exception("This snapshot is for model " + state['model_name'] + "!")

        self.iteration = state['iteration']
        self.upscale_factor = state['upscale']

        self.net: Module = self.get_net_instance(self.upscale_factor).to(self.device)
        self.net.load_state_dict(state['model'])

        self.optimizer = build_optimizer(self._cfg['Optimizer'], self.net.parameters(), self.learning_rate)
        self.optimizer.load_state_dict(state['optimizer'])

        self.scheduler = build_scheduler(self._cfg['Scheduler'], self.optimizer)
        self.scheduler.load_state_dict(state['scheduler'])

    def train_setup(self) -> None:
        assert self._cfg is not None, "Create solver with config to train!"

        # create output folder
        try:
            mkdir(self.output_folder)
        except Exception:
            print("Can't create output folder!", file=stderr)
            ex_type, ex_inst, ex_tb = exc_info()
            raise ex_type.with_traceback(ex_inst, ex_tb)

        # save model description
        with open(self.output_folder + "/arch.txt", 'w') as f:
            f.write(str(self.net) + "\n\n")
            f.write(getsource(self.net.forward))

        # save config
        with open(self.output_folder + '/config.ini', 'w') as f:
            self._cfg.write(f)

        self.drawer = Drawer(self.output_folder, scale_factor=self.upscale_factor)
        self.logger = Logger(self.output_folder)

        self.net.train()
        # TODO: create methods for moving nets and loss to device
        self.net.to(self.device)
        self.loss.to(self.device)

    def train(self, train_set: DataLoader, test_set: DataLoader) -> None:
        self.train_setup()

        while self.iteration <= self.iter_limit:
            for _, (data, target) in enumerate(train_set):
                data, target = data.to(self.device), target.to(self.device)

                result = self.net(data)
                loss = self.compute_loss(result, target)

                train_loss = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # evaluate
                if self.iteration > 0 and self.iteration % self.iter_to_eval == 0:

                    # store training collage
                    if self.drawer and self.iteration % self.iter_per_image == 0:
                        data = interpolate(data, scale_factor=self.upscale_factor) \
                            .cpu().numpy().transpose((0, 2, 3, 1))
                        target = target.cpu().numpy().transpose((0, 2, 3, 1))
                        result = result.detach().cpu().numpy().transpose((0, 2, 3, 1))

                        self.drawer.save_images(data, result, target, "Train-" + str(self.iteration))

                    (test_loss, _), (psnr, psnr_diff), distances = self.evaluate(test_set)

                    if self.logger:
                        line = " ".join([f"Iter:{self.iteration}",
                                         f"Train_loss:{round(train_loss, 5)}",
                                         f"Test_loss:{round(test_loss, 5)}",
                                         f"PSNR:{round(psnr, 5)}",
                                         f"PSNR_diff:{round(psnr_diff, 5)}",
                                         f"Identity_dist_mean:{round(distances.mean(), 5):.5f}",
                                         f"Identity_dist_var:{round(distances.var(), 5):.5f}",
                                         ])
                        self.logger.log(line)

                # snapshot
                if self.iteration % self.iter_per_snapshot == 0 and self.iteration > 0:
                    self.save_model()

                if self.iteration > self.iter_limit:
                    break

                self.iteration += 1

                old_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.scheduler.step(train_loss)
                new_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                if old_lr != new_lr:
                    self.logger.log("LearningRateAdapted")

    def evaluate(self, test_set, identity_only=False):
        assert self.net is not None, "Net model not loaded!"

        net_psnr = 0.
        bilinear_psnr = 0.
        test_loss = 0.
        iterations = 0

        identity_dists = []

        self.net.eval()

        with no_grad():
            for batch_num, (data, real) in enumerate(test_set):
                data, real = data.to(self.device), real.to(self.device)

                fake = self.net(data)

                if not identity_only:
                    test_loss = self.compute_loss(fake, real).item()
                    mse_loss = self.mse(fake, real).item()
                    net_psnr += 10 * log10(1 / mse_loss)

                resized_data = interpolate(data, scale_factor=self.upscale_factor, mode='bilinear', align_corners=True)
                bilinear_mse_loss = self.mse(resized_data, real).item()
                bilinear_psnr += 10 * log10(1 / bilinear_mse_loss)

                resized_data = resized_data.cpu().numpy()

                for res_img, tar_img in zip(fake, real):
                    target_identity, result_identity = self.identity_extractor(tar_img, res_img)

                    if target_identity is None:
                        continue

                    # TODO: verify for senet
                    # TODO: mtcnn detections
                    identity_dists.append(self.identity_extractor.identity_dist(result_identity, target_identity))

                fake = fake.cpu().numpy()
                real = real.cpu().numpy()

                iterations += 1
                if self.test_iter is not None and iterations >= self.test_iter:
                    break

        self.net.train()

        net_psnr /= iterations
        bilinear_psnr /= iterations

        if self.drawer is not None and self.iteration % self.iter_per_image == 0:
            data = resized_data.transpose((0, 2, 3, 1))
            fake = fake.transpose((0, 2, 3, 1))
            real = real.transpose((0, 2, 3, 1))

            self.drawer.save_images(data, fake, real, "Test-" + str(self.iteration))

        # self.net.train()
        return (test_loss, 0.), \
               (net_psnr, net_psnr - bilinear_psnr), \
               np.array(identity_dists)

    def test(self):
        pass

    def single_pass(self, image: Image, downscale: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert self.net is not None, "Model is not loaded!"

        self.net.to(self.device)

        factor = self.upscale_factor if downscale else 1

        # in_transform = Compose([CenterCrop((216, 176)), Resize((216 // factor, 176 // factor)), ToTensor()])
        # eval_transform = Compose([CenterCrop((216, 176)), ToTensor()])

        in_transform = Compose([ToTensor()])
        eval_transform = Compose([ToTensor()])

        # transform = Compose([Resize((256 // factor, 256 // factor)), ToTensor()])

        # add dimension so that tensor is 4d
        inp = in_transform(image).to(self.device).unsqueeze(0)

        with no_grad():
            result = self.net(inp)

        # security through obscurity
        result = result.cpu().detach().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]

        if eval_transform is not None:
            image = eval_transform(image).numpy().transpose((1, 2, 0))[:, :, ::-1]

        return image, result
