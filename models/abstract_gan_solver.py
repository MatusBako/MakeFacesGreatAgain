import dlib
from inspect import getsource
import numpy as np
from numpy import log10
from PIL import Image
from sys import exc_info, stderr
from typing import Tuple

from torch import load, save, no_grad, device, optim, cuda, cat, tensor
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.nn.functional import interpolate
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from utils import Drawer, Logger
from utils.config import GanConfig

from abc import ABC, abstractmethod


class AbstractGanSolver:
    def __init__(self, cfg: GanConfig = None):
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

        # TODO: get cuda device from config
        self.device = device("cuda" if cuda.is_available() else "cpu")

        if cfg is not None:
            self._cfg = cfg
            self.batch_size = cfg.batch_size
            self.upscale_factor = cfg.scale_factor
            self.generator_name = cfg.generator_module

            # TODO: separate generator and disc. LR in config
            self.learning_rate_disc = cfg.discriminator_learning_rate
            self.learning_rate_gen = cfg.generator_learning_rate

            self.iter_limit = cfg.iter_limit
            self.iter_per_snapshot = cfg.iter_per_snapshot
            self.iter_per_image = cfg.iter_per_image
            self.iter_to_eval = cfg.iter_per_eval
            self.test_iter = cfg.test_iter

            self.output_folder = cfg.output_folder

            self.face_detector = dlib.get_frontal_face_detector()
            self.face_localizer = dlib.shape_predictor(cfg.shape_predictor)
            self.identityNN = dlib.face_recognition_model_v1(cfg.identityNN)

        self.mse = MSELoss()

        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed(self.seed)

    @property
    @abstractmethod
    def discriminator_name(self):
        pass

    @abstractmethod
    def get_discriminator_instance(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_generator_instance(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_discriminator_loss(self, response, *args, **kwargs):
        pass

    @abstractmethod
    def compute_generator_loss(self, fake_response, fake_img, real_img, *args, **kwargs):
        pass

    @staticmethod
    def create_generator_scheduler(optimizer):
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, cooldown=5000,
        #                                                      patience=200, factor=0.5)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20000, 40000, 60000, 80000], gamma=0.2)

    @staticmethod
    def create_discriminator_scheduler(optimizer):
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, cooldown=5000,
        #                                                      patience=200, factor=0.5)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20000, 40000, 60000, 80000], gamma=0.2)

    def build_models(self):
        self.discriminator = self.get_discriminator_instance(self.upscale_factor).to(self.device)
        self.optimizer_disc = optim.SGD(self.discriminator.parameters(), lr=self.learning_rate_disc)
        self.scheduler_disc = self.create_discriminator_scheduler(self.optimizer_disc)

        self.generator = self.get_generator_instance(self.upscale_factor).to(self.device)
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=self.learning_rate_gen)
        self.scheduler_gen = self.create_generator_scheduler(self.optimizer_gen)

    def save_discriminator_model(self):
        path = self.output_folder + "/" + self.discriminator_name + "-disc-" + str(self.iteration) + ".mdl"
        dic = {
            'model_name': self.discriminator_name,
            'upscale': self.upscale_factor,
            'iteration': self.iteration,
            'model': self.discriminator.state_dict(),
            'optimizer': self.optimizer_disc.state_dict(),
            'scheduler': self.scheduler_disc.state_dict()
        }

        save(dic, path)
        self.discriminator.to(self.device)

    def save_generator_model(self):
        path = self.output_folder + "/" + self.discriminator_name + "-gen-" + str(self.iteration) + ".mdl"
        dic = {
            'model_name': self.generator_name,
            'upscale': self.upscale_factor,
            'iteration': self.iteration,
            'model': self.generator.state_dict(),
            'optimizer': self.optimizer_gen.state_dict(),
            'scheduler': self.scheduler_gen.state_dict()
        }

        save(dic, path)
        self.generator.to(self.device)

    def save_model(self):
        self.save_discriminator_model()
        self.save_generator_model()

    def load_discriminator_model(self, path: str):
        state = load(path)

        if state['model_name'] != self.discriminator_name:
            raise Exception("This snapshot is for model " + state['model_name'] + "!")

        self.iteration = state['iteration']
        self.upscale_factor = state['upscale']

        self.discriminator = self.get_discriminator_instance(self.upscale_factor).to(self.device)

        self.discriminator.load_state_dict(state['model'])

        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate_disc)
        self.scheduler_disc = self.create_discriminator_scheduler(self.optimizer_disc)

        self.optimizer_disc.load_state_dict(state['optimizer'])
        self.scheduler_disc.load_state_dict(state['scheduler'])

    def load_generator_model(self, path: str):
        state = load(path)

        if state['model_name'] != self.generator_name:
            # raise Exception("This snapshot is for model " + state['model_name'] + "!")
            print("Using generator from model " + state['model_name'], file=stderr)

        self.iteration = state['iteration']
        self.upscale_factor = state['upscale']

        self.generator = self.get_generator_instance(self.upscale_factor).to(self.device)

        self.generator.load_state_dict(state['model'])

        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=self.learning_rate_gen)
        self.scheduler_gen = self.create_generator_scheduler(self.optimizer_gen)

        self.optimizer_gen.load_state_dict(state['optimizer'])
        self.scheduler_gen.load_state_dict(state['scheduler'])

    def train_setup(self) -> None:
        assert self._cfg is not None, "Create solver with config to train!"

        # save model description
        for file, attr, loss in (('/arch_gen.txt', self.generator, self.compute_generator_loss),
                                 ('/arch_disc.txt', self.discriminator, self.compute_discriminator_loss)):
            with open(self.output_folder + file, 'w') as f:
                f.write(str(attr) + '\n\n')
                f.write(getsource(attr.forward) + '\n\n')
                f.write(getsource(loss))

        self._cfg.save(self.output_folder)

        self.drawer = Drawer(self.output_folder, scale_factor=self.upscale_factor)
        self.logger = Logger(self.output_folder)

        self.generator.train()
        self.discriminator.train()

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def train(self, train_set: DataLoader, test_set: DataLoader) -> None:
        self.train_setup()

        while self.iteration <= self.iter_limit:
            for _, (data, target) in enumerate(train_set):
                if data.size()[0] != self.batch_size:
                    continue

                data, target = data.to(self.device), target.to(self.device)
                sr_img = self.generator(data)

                # ######### Train discriminator ########
                response = self.compute_discriminator(sr_img, target)

                discriminator_train_loss = self.compute_discriminator_loss(response)
                discriminator_train_loss_value = discriminator_train_loss.item()

                # don't ask why, just keep reading ...
                discriminator_train_loss.backward(retain_graph=True)

                self.optimizer_disc.step()
                self.scheduler_disc.step()

                # ######## Train generator #########
                self.generator.zero_grad()

                generator_train_loss, log_losses = self.compute_generator_loss(response, sr_img, target)
                generator_train_loss_value = generator_train_loss.item()

                generator_train_loss.backward()
                self.optimizer_gen.step()
                self.scheduler_gen.step()

                sr_img, target = sr_img.cpu(), target.cpu()

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
                    (psnr, psnr_diff), (distance_mean, distance_var) = self.evaluate(test_set)

                    if self.logger:
                        line = " ".join([f"Iter: {self.iteration:}",
                                         f"Gen_Train_loss: {round(generator_train_loss_value, 5):.5f}",
                                         f"Disc_Train_loss: {round(discriminator_train_loss_value, 5):.5f}",
                                         f"Gen_Test_loss: {round(generator_test_loss_value, 5):.5f}",
                                         f"Disc_Test_loss: {round(discriminator_test_loss_value, 5):.5f}",
                                         f"PSNR: {round(psnr, 3):.3f}",
                                         f"PSNR_diff: {round(psnr_diff, 3):.3f}",
                                         f"Identity_dist_mean: {round(distance_mean, 5):.5f}",
                                         f"Identity_dist_var: {round(distance_var, 5):.5f}",
                                         f"PixelLoss: {round(log_losses[0], 6):.6f}",
                                         f"AdvLoss: {round(log_losses[1], 6):.6f}",
                                         f"FeatureLoss: {round(log_losses[2], 6):.6f}",
                                         ])
                        self.logger.log(line)

                if self.iteration > self.iter_limit:
                    break

                self.iteration += 1

                # snapshot
                if self.iteration % self.iter_per_snapshot == 0:
                    self.save_generator_model()
                    self.save_discriminator_model()

                    # for ReduceLRonPlateau
                    # self.scheduler.step(train_loss)

    def compute_discriminator(self, sr_img: tensor, hr_img: tensor):
        """
        Args:
            sr_img(tensor): tensor containing generated image (output of network)
            hr_img(tensor): tensor containing ground truth image

        Returns:
            vector of discriminator responses to each image
            first half are responses to ground truth and second half to generated image
        """
        disc_input = cat((hr_img, sr_img)).to(device(self.device))

        indices = np.array(range(disc_input.size()[0]))
        # numbers 0-15: real, 16-31: fake
        np.random.shuffle(indices)

        response = self.discriminator(disc_input[indices])
        disc_input.to(device('cpu'))

        # sort reponse to be in the same order as concatenated arrays
        response = response[np.argsort(indices)]

        return response

    def evaluate(self, test_set):
        assert self.generator and self.discriminator is not None, "Net model not loaded!"

        net_psnr = 0.
        interpolation_psnr = 0.
        interpolation_loss = 0.

        generator_loss_value = 0.
        discriminator_loss_value = 0.
        identity_dist = 0.
        identity_dists = []

        iterations = 0

        # self.net.eval()

        with no_grad():
            for batch_num, (data, target) in enumerate(test_set):
                data, target = data.to(self.device), target.to(self.device)

                result = self.generator(data)

                # Compute discriminator
                response = self.compute_discriminator(result, target)
                discriminator_loss_value = self.compute_discriminator_loss(response).item()

                # ##### Test generator #####
                mse = self.mse(result, target).item()
                net_psnr += 10 * log10(1 / mse)
                tmp, _ = self.compute_generator_loss(response, result, target)
                generator_loss_value += tmp.item()

                resized_data = interpolate(data, scale_factor=self.upscale_factor)
                interpolation_loss += self.mse(resized_data, target).item()
                interpolation_psnr += 10 * log10(1 / interpolation_loss)

                resized_data = resized_data.cpu().numpy()
                result = result.cpu().numpy()
                target = target.cpu().numpy()
                # result = (result.cpu().numpy() * 256).astype(np.uint8)[:, :, :, ::-1]
                # target = (target.cpu().numpy() * 256).astype(np.uint8)[:, :, :, ::-1]

                # security through obscurity
                # TODO: refactor
                for res_img, tar_img in zip(np.transpose((result * 256).astype(np.uint8)[:, :, :, ::-1], (0, 2, 3, 1)),
                                            np.transpose((target * 256).astype(np.uint8)[:, :, :, ::-1], (0, 2, 3, 1))):

                    # res_detections = self.face_detector(res_img)
                    detections = self.face_detector(tar_img)

                    # face not detected
                    if not len(detections):
                        continue

                    # res_face_shape = self.face_localizer(res_img, res_detections[0])
                    face_shape = self.face_localizer(tar_img, detections[0])

                    result_identity = np.array(self.identityNN.compute_face_descriptor(res_img, face_shape))
                    target_identity = np.array(self.identityNN.compute_face_descriptor(tar_img, face_shape))

                    identity_dist += np.sum(result_identity ** 2) + np.sum(target_identity ** 2) \
                                     - 2 * np.dot(result_identity, target_identity)
                    identity_dists.append(np.sum(result_identity ** 2) + np.sum(target_identity ** 2)
                                          - 2 * np.dot(result_identity, target_identity))

                iterations += 1
                if self.test_iter is not None and iterations % self.test_iter == 0:
                    break

        net_psnr /= iterations
        interpolation_psnr /= iterations

        generator_loss_value /= iterations
        discriminator_loss_value /= iterations

        identity_dists = np.array(identity_dists)
        identity_dist /= iterations * self.batch_size

        if self.drawer is not None and self.iteration % self.iter_per_image == 0:
            data = resized_data.transpose((0, 2, 3, 1))
            result = result.transpose((0, 2, 3, 1))
            target = target.transpose((0, 2, 3, 1))

            self.drawer.save_images(data, result, target, "Test-" + str(self.iteration))

        # self.net.train()
        return (generator_loss_value, discriminator_loss_value), \
               (net_psnr, net_psnr - interpolation_psnr), \
               (identity_dists.mean(), identity_dists.var())

    def test(self):
        pass

    def single_pass(self, image: Image, downscale: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert self.generator is not None, "Model is not loaded!"

        self.generator.to(self.device)

        factor = self.upscale_factor if downscale else 1

        # in_transform = Compose([CenterCrop((216, 176)), Resize((216 // factor, 176 // factor)), ToTensor()])
        # eval_transform = Compose([CenterCrop((216, 176)), ToTensor()])

        in_transform = Compose([ToTensor()])
        eval_transform = Compose([ToTensor()])

        # transform = Compose([Resize((256 // factor, 256 // factor)), ToTensor()])

        # add dimension so that tensor is 4d
        inp = in_transform(image).to(self.device).unsqueeze(0)

        # TODO: fix
        with no_grad():
            result = self.generator(inp)

        # security through obscurity
        result = result.cpu().detach().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]

        if eval_transform is not None:
            image = eval_transform(image).numpy().transpose((1, 2, 0))[:, :, ::-1]

        return image, result
