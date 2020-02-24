import dlib
import numpy as np

from configparser import SectionProxy, ConfigParser
from datetime import datetime
from inspect import getsource, getframeinfo, currentframe, getfile
from numpy import log10
from os import mkdir, listdir
from os.path import dirname, abspath, join
from PIL import Image
from pytorch_msssim import ssim
from shutil import copyfile
from sys import exc_info, stderr, path
from time import time
from tqdm import tqdm
from typing import Tuple, Optional

from torch import load, save, no_grad, device, optim, cuda, cat, tensor, uint8
from torch.nn import MSELoss, Module
from torch.nn.functional import interpolate
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
from torch.utils.data import DataLoader

from factory import build_optimizer, build_scheduler, build_feature_extractor

try:
    from utils import Drawer, Logger
except ModuleNotFoundError:
    script = getframeinfo(currentframe()).filename
    root_dir = dirname((abspath(script)))
    path.append(root_dir)
    from utils import Drawer, Logger

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

        self.mse_loss = MSELoss()
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
    def compute_discriminator_loss(self, fake_img, real_img, precomputed=None, train=True, *args, **kwargs):
        pass

    @abstractmethod
    def compute_generator_loss(self, label, fake_img, real_img, precomputed=None, *args, **kwargs):
        pass

    def post_backward_generator(self):
        """
        May be used for gradient clipping
        """
        pass

    def post_backward_discriminator(self):
        """
        May be used for gradient clipping
        """
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

        # self.iteration = state['iteration']
        self.upscale_factor = state['upscale']

        self.discriminator: Module = self.build_discriminator(self.upscale_factor).to(self.device)
        self.discriminator.load_state_dict(state['model'])

        self.optimizer_disc = build_optimizer(self._cfg['DiscOptimizer'], self.discriminator.parameters(),
                                              self.learning_rate_disc)
        self.optimizer_disc.load_state_dict(state['optimizer'])

        self.scheduler_disc = build_scheduler(self._cfg['DiscScheduler'], self.optimizer_disc)
        self.scheduler_disc.load_state_dict(state['scheduler'])

    def load_generator_model(self, model_path: str):
        state = load(model_path)

        # self.iteration = state['iteration']
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

        # copy all python modules found in directory of trained model
        module_folder = dirname(getfile(self.generator.__class__))
        files = [file for file in listdir(module_folder) if not file.startswith("_") and file.endswith(".py")]

        for file in files:
            copyfile(join(module_folder, file), join(self.output_folder, file))

        copyfile(join(dirname(module_folder), "abstract_gan_solver.py"), join(self.output_folder, "abstract_solver.py"))

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

        gen_train_values = []
        disc_train_values = []

        while self.iteration <= self.iter_limit:
            for _, (labels, input_img, target_img) in enumerate(train_set):
                if len(labels) < self.batch_size:
                    continue

                # TODO: to incude evaluation or not to
                # reset timer in case test iterations were executed
                progress_bar.last_print_t = time()

                input_img, target_img = input_img.to(self.device), target_img.to(self.device)

                fake_img = self.generator(input_img)

                # ######## Train generator #########
                self.optimizer_gen.zero_grad()

                generator_train_loss, log_losses, precomputed = self.compute_generator_loss(labels, fake_img, target_img)
                gen_train_values.append(generator_train_loss.item())

                generator_train_loss.backward(retain_graph=True)
                self.post_backward_generator()
                self.optimizer_gen.step()

                # ######### Train discriminator ########
                self.optimizer_disc.zero_grad()

                discriminator_train_loss = self.compute_discriminator_loss(fake_img.detach(), target_img, precomputed)
                disc_train_values.append(discriminator_train_loss.item())

                # don't ask why, just keep reading ...
                discriminator_train_loss.backward()
                self.post_backward_discriminator()
                self.optimizer_disc.step()

                self.iteration += 1
                fake_img, target_img = fake_img.cpu(), target_img.cpu()

                # TODO: create function for sched step upfront to avoid more isinstance() calls
                if isinstance(self.scheduler_gen, optim.lr_scheduler.ReduceLROnPlateau):
                    old_lr = self.optimizer_gen.state_dict()['param_groups'][0]['lr']
                    self.scheduler_gen.step(generator_train_loss)
                    new_lr = self.optimizer_gen.state_dict()['param_groups'][0]['lr']

                    if old_lr != new_lr and self.logger:
                        self.logger.log("GeneratorLearningRateAdapted")
                else:
                    self.scheduler_gen.step()

                if isinstance(self.scheduler_disc, optim.lr_scheduler.ReduceLROnPlateau):
                    old_lr = self.optimizer_disc.state_dict()['param_groups'][0]['lr']
                    self.scheduler_disc.step(discriminator_train_loss)
                    new_lr = self.optimizer_disc.state_dict()['param_groups'][0]['lr']

                    if old_lr != new_lr and self.logger:
                        self.logger.log("DiscriminatorLearningRateAdapted")
                else:
                    self.scheduler_disc.step()

                # ######## Statistics #########
                if self.iteration > 0 and self.iteration % self.iter_to_eval == 0:

                    # store training collage
                    if self.drawer and self.iteration % self.iter_per_image == 0:
                        input_img = interpolate(input_img, scale_factor=self.upscale_factor).cpu().numpy().transpose(
                            (0, 2, 3, 1))
                        target_img = target_img.cpu().numpy().transpose((0, 2, 3, 1))
                        fake_img = fake_img.detach().cpu().numpy().transpose((0, 2, 3, 1))

                        self.drawer.save_images(input_img, fake_img, target_img, "Train-" + str(self.iteration))

                    # TODO: refactor
                    # do we even need PSNR?
                    (generator_test_loss_value, discriminator_test_loss_value), \
                    (psnr, psnr_diff, ssim_val), distances = self.evaluate(test_set)

                    if self.logger:
                        line = " ".join([f"Iter:{self.iteration:}",
                                         f"Gen_Train_loss:{np.round(np.mean(gen_train_values), 5):.5f}",
                                         f"Disc_Train_loss:{np.round(np.mean(disc_train_values), 5):.5f}",
                                         f"Gen_Test_loss:{round(generator_test_loss_value, 5):.5f}",
                                         f"Disc_Test_loss:{round(discriminator_test_loss_value, 5):.5f}",
                                         f"PSNR:{round(psnr, 3):.3f}",
                                         f"PSNR_diff:{round(psnr_diff, 3):.3f}",
                                         f"SSIM:{round(ssim_val, 5)}",
                                         f"Identity_dist_mean:{round(distances.mean(), 5):.5f}",
                                         f"PixelLoss:{log_losses[0]:.7f}",
                                         f"AdvLoss:{log_losses[1]:.7f}",
                                         f"FeatureLoss:{log_losses[2]:.7f}",
                                         ])
                        self.logger.log(line)

                    gen_train_values.clear()
                    disc_train_values.clear()

                # snapshot
                if self.iteration % self.iter_per_snapshot == 0:
                    self.save_generator_model()
                    self.save_discriminator_model()

                if self.iteration > self.iter_limit:
                    break

                progress_bar.update()

        progress_bar.close()

    def evaluate(self, test_set: DataLoader, identity_only=False):
        assert self.generator is not None, "Generator model not loaded!"
        assert self.discriminator is not None or identity_only, "Discriminator model not loaded!"

        net_psnr = 0.
        interpolation_psnr = 0.
        ssim_val = 0.

        generator_loss_value = 0.
        discriminator_loss_value = 0.
        identity_dist = 0.
        identity_dists = []

        iterations = 0

        self.generator.eval()
        self.discriminator.eval()

        with no_grad():
            # TODO: propagate image path with the image
            for batch_num, (labels, input_img, target_img) in enumerate(test_set):
                input_img, target_img = input_img.to(self.device), target_img.to(self.device)

                fake_img = self.generator(input_img)
                mse = self.mse_loss(fake_img, target_img).item()
                net_psnr += 10 * log10(1 / mse)

                # Compute discriminator
                if not identity_only:
                    fake_response, real_response = self.discriminator(cat((fake_img, target_img))).split(
                        fake_img.size(0))
                    # response.to(device('cpu')) # ??
                    discriminator_loss_value = self.compute_discriminator_loss(fake_img, target_img, train=False).item()

                    # ##### Test generator #####
                    tmp, _, _ = self.compute_generator_loss(labels, fake_img, target_img, real_response)
                    generator_loss_value += tmp.item()

                resized_data = interpolate(input_img, scale_factor=self.upscale_factor, mode='bilinear',
                                           align_corners=True)
                interpolation_loss = self.mse_loss(resized_data, target_img).item()
                interpolation_psnr += 10 * log10(1 / interpolation_loss)

                resized_data = resized_data.cpu().numpy()

                # TODO: rewrite so that whole batch can be passed
                for label, res_img, tar_img in zip(labels, fake_img, target_img):
                    target_identity, result_identity = self.identity_extractor(label, tar_img, res_img)

                    if target_identity is None:
                        continue

                    # TODO: verify for senet
                    # TODO: mtcnn detections
                    identity_dists.append(self.identity_extractor.identity_dist(result_identity, target_identity))

                fake_img = fake_img.cpu()
                target_img = target_img.cpu()

                ssim_val += ssim(tensor(fake_img * 255, dtype=uint8),
                                 tensor(target_img * 255, dtype=uint8),
                                 nonnegative_ssim=True).mean()

                fake_img = fake_img.numpy()
                target_img = target_img.numpy()

                iterations += 1
                if self.test_iter is not None and iterations % self.test_iter == 0:
                    break

        self.generator.train()
        self.discriminator.train()

        net_psnr /= iterations
        interpolation_psnr /= iterations
        ssim_val /= iterations

        generator_loss_value /= iterations
        discriminator_loss_value /= iterations

        identity_dists = np.array(identity_dists)
        identity_dist /= iterations * self.batch_size

        if self.drawer is not None and self.iteration % self.iter_per_image == 0:
            input_img = resized_data.transpose((0, 2, 3, 1))
            fake_img = fake_img.transpose((0, 2, 3, 1))
            target_img = target_img.transpose((0, 2, 3, 1))

            self.drawer.save_images(input_img, fake_img, target_img, "Test-" + str(self.iteration))

        return (generator_loss_value, discriminator_loss_value), \
               (net_psnr, net_psnr - interpolation_psnr, ssim_val), \
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
