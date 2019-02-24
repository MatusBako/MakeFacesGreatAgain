from datetime import datetime
from inspect import getsource
import numpy as np
from numpy import log10
from os import mkdir
from PIL import Image
from sys import exc_info, stderr
from time import time
from typing import Tuple

from torch import load, save, no_grad, device, optim, cuda
from torch.nn import MSELoss
from torch.nn.functional import interpolate
from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop
from torch.utils.data import DataLoader

from utils import Drawer, Logger, ConfigWrapper

from abc import ABC, abstractmethod


class AbstractCnnSolver:
    def __init__(self, cfg: ConfigWrapper=None):
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

        #TODO: get cuda device from config
        self.device = device("cuda" if cuda.is_available() else "cpu")

        if cfg is not None:
            self._cfg = cfg
            self.batch_size = cfg.batch_size
            self.learning_rate = cfg.learning_rate
            self.upscale_factor = cfg.scale_factor

            self.iter_limit = cfg.iter_limit
            self.iter_per_snapshot = cfg.iter_per_snapshot
            self.iter_per_image = cfg.iter_per_image
            self.iter_to_eval = cfg.iter_per_eval
            self.test_iter = cfg.test_iter

            self.output_folder = cfg.output_folder

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

    @staticmethod
    def create_scheduler(optimizer):
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, cooldown=5000,
        #                                                      patience=200, factor=0.5)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20000, 40000, 60000, 80000], gamma=0.2)

    def build_models(self, cfg: ConfigWrapper):
        self.net = self.get_net_instance(cfg.scale_factor).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = self.create_scheduler(self.optimizer)

    def save_model(self):
        path = self.output_folder + "/" + self.name + "-" + str(self.iteration) + ".mdl"
        dic = {
            'model_name': self.name,
            'upscale': self.upscale_factor,
            'iteration': self.iteration,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        save(dic, path)
        self.net.to(self.device)

    def load_model(self, path: str):
        state = load(path)

        if state['model_name'] != self.name:
            raise Exception("This snapshot is for model " + state['model_name'] + "!")

        self.iteration = state['iteration']
        self.upscale_factor = state['upscale']

        self.net = self.get_net_instance(self.upscale_factor).to(self.device)
        self.net.load_state_dict(state['model'])

        self.optimizer = optim.Adam(self.net.parameters())
        self.scheduler = self.create_scheduler(self.optimizer)

        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])

    def train_setup(self) -> None:
        assert self._cfg is not None, "Create solver with config to train!"

        # save model description
        with open(self.output_folder + "/arch.txt", 'w') as f:
            f.write(str(self.net) + "\n\n")
            f.write(getsource(self.net.forward))

        self._cfg.save(self.output_folder)
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
                self.scheduler.step()

                # evaluate
                if self.iteration > 0 and self.iteration % self.iter_to_eval == 0:

                    # store training collage
                    if self.drawer and self.iteration % self.iter_per_image == 0:
                        data = interpolate(data, scale_factor=self.upscale_factor, mode='nearest', align_corners=True)\
                            .cpu().numpy().transpose((0, 2, 3, 1))
                        target = target.cpu().numpy().transpose((0, 2, 3, 1))
                        result = result.detach().cpu().numpy().transpose((0, 2, 3, 1))

                        self.drawer.save_images(data, result, target, "Train-" + str(self.iteration))

                    test_loss, psnr, psnr_diff = self.evaluate(test_set)

                    if self.logger:
                        line = " ".join(["Iter:" + str(self.iteration),
                                         "Train_loss:" + str(round(train_loss, 5)),
                                         "Test_loss:" + str(round(test_loss, 5)),
                                         "PSNR:" + str(round(psnr, 5)),
                                         "PSNR_diff:" + str(round(psnr_diff, 5))
                                         ])
                        self.logger.log(line)

                if self.iteration > self.iter_limit:
                    break

                self.iteration += 1

                # snapshot
                if self.iteration % self.iter_per_snapshot == 0:
                    self.save_model()

            # for ReduceLROnPlateau
            #self.scheduler.step(train_loss)

    def evaluate(self, test_set):
        assert self.net is not None, "Net model not loaded!"

        net_psnr = 0.
        bilinear_psnr = 0.
        test_loss = 0.
        iterations = 0

        #self.net.eval()

        with no_grad():
            for batch_num, (data, target) in enumerate(test_set):
                data, target = data.to(self.device), target.to(self.device)

                result = self.net(data)
                test_loss = self.compute_loss(result, target).item()
                mse_loss = self.mse(result, target).item()
                net_psnr += 10 * log10(1 / mse_loss)

                resized_data = interpolate(data, scale_factor=self.upscale_factor, mode='bilinear', align_corners=True)
                bilinear_mse_loss = self.mse(resized_data, target).item()
                bilinear_psnr += 10 * log10(1 / bilinear_mse_loss)

                resized_data = resized_data.cpu()
                result = result.cpu()
                target = target.cpu()

                iterations += 1
                if self.test_iter is not None and iterations % self.test_iter == 0:
                    break

        net_psnr /= iterations
        bilinear_psnr /= iterations

        if self.drawer is not None and self.iteration % self.iter_per_image == 0:
            data = resized_data.numpy().transpose((0, 2, 3, 1))
            result = result.numpy().transpose((0, 2, 3, 1))
            target = target.numpy().transpose((0, 2, 3, 1))

            self.drawer.save_images(data, result, target, "Test-" + str(self.iteration))

        #self.net.train()
        return test_loss, net_psnr, net_psnr - bilinear_psnr

    def test(self):
        pass

    def single_pass(self, image: Image, downscale: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert self.net is not None, "Model is not loaded!"

        self.net.to(self.device)

        factor = self.upscale_factor if downscale else 1

        #in_transform = Compose([CenterCrop((216, 176)), Resize((216 // factor, 176 // factor)), ToTensor()])
        #eval_transform = Compose([CenterCrop((216, 176)), ToTensor()])

        in_transform = Compose([ToTensor()])
        eval_transform = Compose([ToTensor()])

        #transform = Compose([Resize((256 // factor, 256 // factor)), ToTensor()])

        # add dimension so that tensor is 4d
        inp = in_transform(image).to(self.device).unsqueeze(0)

        with no_grad():
            result = self.net(inp)

        # security through obscurity
        result = result.cpu().detach().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]

        if eval_transform is not None:
            image = eval_transform(image).numpy().transpose((1, 2, 0))[:, :, ::-1]

        return image, result
