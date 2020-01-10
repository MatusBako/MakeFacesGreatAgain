import dlib
import numpy as np

from configparser import SectionProxy, ConfigParser
from datetime import datetime
from inspect import getsource, getframeinfo, currentframe
from numpy import log10
from os import mkdir
from os.path import dirname, abspath
from PIL import Image
from sys import exc_info, stderr, path
from time import time
from tqdm import tqdm
from typing import Tuple, Optional

from torch import load, save, no_grad, device, optim, cuda, cat, tensor
from torch.autograd import Variable
from torch.nn import MSELoss, Module
from torch.nn.functional import interpolate
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
from torch.utils.data import DataLoader

from factory import build_optimizer, build_scheduler, build_feature_extractor

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


class AbstractGanSolver:
    def __init__(self, config: Optional[ConfigParser] = None):
        self.iteration = 0

        self.generator = None
        self.learning_rate_gen = None
        self.optimizer_gen = None
        self.scheduler_gen = None

        self.discriminator = None
        self.learning_rate_disc = None
        self.optimizer_disc = None
        self.scheduler_disc = None

        self.output_folder = None
        self.drawer = None
        self.logger = None
        self._cfg = None
        self.test_iter = None
        self.upscale_factor = None

        self.device = device("cuda" if cuda.is_available() else "cpu")

        if config is not None:
            self._cfg = config
            nn_cfg: SectionProxy = config["GAN"]

            self.device = device(nn_cfg['Device'])
            self.batch_size = nn_cfg.getint('BatchSize')
            self.upscale_factor = nn_cfg.getint('UpscaleFactor')
            self.generator_name = nn_cfg['Generator']

            self.learning_rate_disc = nn_cfg.getfloat('DiscriminatorLearningRate')
            self.learning_rate_gen = nn_cfg.getfloat('GeneratorLearningRate')

            self.iter_limit = nn_cfg.getint('IterationLimit')
            self.iter_per_snapshot = nn_cfg.getint('IterationsPerSnapshot')
            self.iter_per_image = nn_cfg.getint('IterationsPerImage')
            self.iter_to_eval = nn_cfg.getint('IterationsToEvaluation')
            self.test_iter = nn_cfg.getint('EvaluationIterations')

            timestamp = str(datetime.fromtimestamp(time()).strftime('%Y.%m.%d-%H:%M:%S'))
            self.output_folder = nn_cfg['OutputFolder'] + "/" + self.discriminator_name + "-" + timestamp

            self.identity_extractor = build_feature_extractor(config['FeatureExtractor'])

        self.mse = MSELoss()
        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed(self.seed)

    @property
    @abstractmethod
    def discriminator_name(self):
        pass

    @abstractmethod
    def build_discriminator(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_generator(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_discriminator_loss(self, response, *args, **kwargs):
        pass

    @abstractmethod
    def compute_generator_loss(self, fake_response, fake_img, real_img, *args, **kwargs):
        pass

    def build_models(self):
        self.discriminator: Module = self.build_discriminator(self.upscale_factor).to(self.device)

        self.optimizer_disc = build_optimizer(self._cfg['DiscOptimizer'],
                                              self.discriminator.parameters(),
                                              self.learning_rate_disc)
        self.scheduler_disc = build_scheduler(self._cfg['DiscScheduler'], self.optimizer_disc)

        self.generator = self.build_generator(self.upscale_factor).to(self.device)

        self.optimizer_gen = build_optimizer(self._cfg['GenOptimizer'],
                                             self.generator.parameters(),
                                             self.learning_rate_gen)
        self.scheduler_gen = build_scheduler(self._cfg['GenScheduler'], self.optimizer_gen)

    def save_discriminator_model(self):
        checkpoint_path = self.output_folder + "/" + self.discriminator_name + "-disc-" + str(self.iteration) + ".mdl"
        dic = {
            'model_name': self.discriminator_name,
            'upscale': self.upscale_factor,
            'iteration': self.iteration,
            'model': self.discriminator.state_dict(),
            'optimizer': self.optimizer_disc.state_dict(),
            'scheduler': self.scheduler_disc.state_dict()
        }

        save(dic, checkpoint_path)
        self.discriminator.to(self.device)

    def save_generator_model(self):
        checkpoint_path = self.output_folder + "/" + self.discriminator_name + "-gen-" + str(self.iteration) + ".mdl"
        dic = {
            'model_name': self.generator_name,
            'upscale': self.upscale_factor,
            'iteration': self.iteration,
            'model': self.generator.state_dict(),
            'optimizer': self.optimizer_gen.state_dict(),
            'scheduler': self.scheduler_gen.state_dict()
        }

        save(dic, checkpoint_path)
        self.generator.to(self.device)

    def save_model(self):
        self.save_discriminator_model()
        self.save_generator_model()

    def load_discriminator_model(self, model_path: str):
        state = load(model_path)

        if state['model_name'] != self.discriminator_name:
            raise Exception("This snapshot is for model " + state['model_name'] + "!")

        self.iteration = state['iteration']
        self.upscale_factor = state['upscale']

        self.discriminator: Module = self.build_discriminator(self.upscale_factor).to(self.device)
        self.discriminator.load_state_dict(state['model'])

        self.optimizer_disc = build_optimizer(self._cfg['DiscOptimizer'], self.discriminator.parameters,
                                              self.learning_rate_disc)
        self.optimizer_disc.load_state_dict(state['optimizer'])

        self.scheduler_disc = build_scheduler(self._cfg['DiscScheduler'], self.optimizer_disc)
        self.scheduler_disc.load_state_dict(state['scheduler'])

    def load_generator_model(self, model_path: str):
        state = load(model_path)

        self.iteration = state['iteration']
        self.upscale_factor = state['upscale']

        self.generator = self.build_generator(self.upscale_factor).to(self.device)
        self.generator.load_state_dict(state['model'])

        self.optimizer_gen = build_optimizer(self._cfg['GenOptimizer'], self.generator.parameters(),
                                             self.learning_rate_gen)
        self.optimizer_gen.load_state_dict(state['optimizer'])

        self.scheduler_gen = build_scheduler(self._cfg['GenScheduler'], self.optimizer_gen)
        self.scheduler_gen.load_state_dict(state['scheduler'])

    def train_setup(self) -> None:
        assert self._cfg is not None, "Create solver with config to train!"

        try:
            mkdir(self.output_folder)
        except Exception:
            print("Can't create output folder!", file=stderr)
            ex_type, ex_inst, ex_tb = exc_info()
            raise ex_type.with_traceback(ex_inst, ex_tb)

        # save model description
        for file, attr, loss in (('/arch_gen.txt', self.generator, self.compute_generator_loss),
                                 ('/arch_disc.txt', self.discriminator, self.compute_discriminator_loss)):
            with open(self.output_folder + file, 'w') as f:
                f.write(str(attr) + '\n\n')
                f.write(getsource(attr.forward) + '\n\n')
                f.write(getsource(loss))

        with open(self.output_folder + '/config.ini', 'w') as f:
            self._cfg.write(f)

        self.drawer = Drawer(self.output_folder, scale_factor=self.upscale_factor)
        self.logger = Logger(self.output_folder)

        self.generator.train()
        self.discriminator.train()

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def train(self, train_set: DataLoader, test_set: DataLoader) -> None:
        self.train_setup()

        progress_bar = tqdm(total=self.iter_limit)

        while self.iteration <= self.iter_limit:
            for _, (data, target) in enumerate(train_set):
                # TODO: to incude evaluation or not to
                # reset timer in case test iterations were executed
                # progress_bar.last_print_t = time()

                data, target = data.to(self.device), target.to(self.device)
                sr_img = self.generator(data)

                # ######### Train discriminator ########
                response = self.discriminator(cat((target, sr_img)))

                discriminator_train_loss = self.compute_discriminator_loss(response)
                discriminator_train_loss_value = discriminator_train_loss.item()

                # don't ask why, just keep reading ...
                discriminator_train_loss.backward(retain_graph=True)

                self.optimizer_disc.step()
                # TODO: verify if it works also without ReduceLRonPlatau
                self.scheduler_disc.step(discriminator_train_loss_value)

                # ######## Train generator #########
                self.generator.zero_grad()

                generator_train_loss, log_losses = self.compute_generator_loss(response, sr_img, target)
                generator_train_loss_value = generator_train_loss.item()

                generator_train_loss.backward()
                self.optimizer_gen.step()
                self.scheduler_gen.step(generator_train_loss_value)

                sr_img, target = sr_img.cpu(), target.cpu()
                progress_bar.update()

                # ######## Statistics #########
                if self.iteration > 0 and self.iteration % self.iter_to_eval == 0:

                    # store training collage
                    if self.drawer and self.iteration % self.iter_per_image == 0:
                        data = interpolate(data, scale_factor=self.upscale_factor).cpu().numpy().transpose((0, 2, 3, 1))
                        target = target.cpu().numpy().transpose((0, 2, 3, 1))
                        sr_img = sr_img.detach().cpu().numpy().transpose((0, 2, 3, 1))

                        self.drawer.save_images(data, sr_img, target, "Train-" + str(self.iteration))

                    # TODO: refactor
                    # do we even need PSNR?
                    (generator_test_loss_value, discriminator_test_loss_value), \
                    (psnr, psnr_diff), distances = self.evaluate(test_set)

                    if self.logger:
                        line = " ".join([f"Iter:{self.iteration:}",
                                         f"Gen_Train_loss:{round(generator_train_loss_value, 5):.5f}",
                                         f"Disc_Train_loss:{round(discriminator_train_loss_value, 5):.5f}",
                                         f"Gen_Test_loss:{round(generator_test_loss_value, 5):.5f}",
                                         f"Disc_Test_loss:{round(discriminator_test_loss_value, 5):.5f}",
                                         f"PSNR:{round(psnr, 3):.3f}",
                                         f"PSNR_diff:{round(psnr_diff, 3):.3f}",
                                         f"Identity_dist_mean:{round(distances.mean(), 5):.5f}",
                                         f"Identity_dist_var:{round(distances.var(), 5):.5f}",
                                         f"PixelLoss:{log_losses[0]:.7f}",
                                         f"AdvLoss:{log_losses[1]:.7f}",
                                         f"FeatureLoss:{log_losses[2]:.7f}",
                                         ])
                        self.logger.log(line)

                if self.iteration > self.iter_limit:
                    break

                self.iteration += 1

                # snapshot
                if self.iteration % self.iter_per_snapshot == 0:
                    self.save_generator_model()
                    self.save_discriminator_model()

        progress_bar.close()

    def evaluate(self, test_set: DataLoader, identity_only=False):
        assert self.generator is not None, "Generator model not loaded!"
        assert self.discriminator is not None or identity_only, "Discriminator model not loaded!"

        net_psnr = 0.
        interpolation_psnr = 0.

        generator_loss_value = 0.
        discriminator_loss_value = 0.
        identity_dist = 0.
        identity_dists = []

        iterations = 0

        self.generator.eval()
        self.discriminator.eval()

        with no_grad():
            # TODO: propagate image path with the image
            for batch_num, (data, real) in enumerate(test_set):
                data, real = data.to(self.device), real.to(self.device)

                fake = self.generator(data)
                mse = self.mse(fake, real).item()
                net_psnr += 10 * log10(1 / mse)

                # Compute discriminator
                if not identity_only:
                    response = self.discriminator(cat((real, fake)))
                    response.to(device('cpu'))
                    discriminator_loss_value = self.compute_discriminator_loss(response).item()

                    # ##### Test generator #####
                    tmp, _ = self.compute_generator_loss(response, fake, real)
                    generator_loss_value += tmp.item()

                resized_data = interpolate(data, scale_factor=self.upscale_factor, mode='bilinear', align_corners=True)
                interpolation_loss = self.mse(resized_data, real).item()
                interpolation_psnr += 10 * log10(1 / interpolation_loss)

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
                if self.test_iter is not None and iterations % self.test_iter == 0:
                    break

        self.generator.train()
        self.discriminator.train()

        net_psnr /= iterations
        interpolation_psnr /= iterations

        generator_loss_value /= iterations
        discriminator_loss_value /= iterations

        identity_dists = np.array(identity_dists)
        identity_dist /= iterations * self.batch_size

        if self.drawer is not None and self.iteration % self.iter_per_image == 0:
            data = resized_data.transpose((0, 2, 3, 1))
            fake = fake.transpose((0, 2, 3, 1))
            real = real.transpose((0, 2, 3, 1))

            self.drawer.save_images(data, fake, real, "Test-" + str(self.iteration))

        return (generator_loss_value, discriminator_loss_value), \
               (net_psnr, net_psnr - interpolation_psnr), \
               identity_dists

    def test(self, test_set: DataLoader):
        pass

    def single_pass(self, image: Image, downscale: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert self.generator is not None, "Model is not loaded!"

        self.generator.to(self.device)

        factor = self.upscale_factor if downscale else 1

        # in_transform = Compose([CenterCrop((208, 176)), Resize((208 // factor, 176 // factor)), ToTensor()])
        # eval_transform = Compose([CenterCrop((216, 176)), ToTensor()])

        in_transform = Compose([ToTensor()])
        eval_transform = Compose([ToTensor()])

        # transform = Compose([Resize((256 // factor, 256 // factor)), ToTensor()])

        # add dimension so that tensor is 4d
        inp = in_transform(image).to(self.device).unsqueeze(0)

        with no_grad():
            result = self.generator(inp)

        # security through obscurity
        result = result.cpu().detach().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]

        if eval_transform is not None:
            image = eval_transform(image).numpy().transpose((1, 2, 0))[:, :, ::-1]

        return image, result
