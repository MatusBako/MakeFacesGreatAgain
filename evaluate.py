#!/usr/bin/env python3

import argparse
import dlib
from numpy import log10, round

from os import listdir
from os.path import dirname, abspath
from inspect import getframeinfo, currentframe
from sys import path

from torch.utils.data import DataLoader
from importlib import import_module

from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop, RandomChoice
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('arch')
    parser.add_argument('checkpoint')
    parser.add_argument('dataset', default='FFHQ')
    parser.add_argument('-s', '--scale', default=4, type=int)
    return parser.parse_args()


def build_input_transform(h=256, w=256, upscale_factor=4):
    return Compose([
        CenterCrop((h, w)),
        Resize((h // upscale_factor, w // upscale_factor), interpolation=Image.BICUBIC),
        ToTensor(),
    ])


class Config:
    def __init__(self, arch, batch_size):
        self.device = 'cuda'
        self.batch_size = batch_size
        self.generator_module = arch

def main():
    args = get_args()

    if args.dataset.lower() == 'ffhq':
        dataset_module = 'DatasetFFHQ'
        dataset_path = '/media/hacky/DATA2/FFHQ/test'
        h=256
        w=256
    elif args.dataset.lower() == 'celeba':
        dataset_module = 'DatasetCelebA'
        dataset_path = '/home/hacky/datasets/CelebA/Img/aligned/test'
        h=208
        w=176
    else:
        raise NotImplementedError("This dataset is not implemented.")

    # get project directory
    script = getframeinfo(currentframe()).filename
    proj_dir = dirname(abspath(script))

    # add modules to pythonpath
    path.append(proj_dir + "/models")
    path.append(proj_dir)
    assert args.arch in listdir(proj_dir + '/models')
    # TODO: assert existence of dataset module

    Solver = __import__(args.arch, fromlist=['Solver']).Solver
    solver = Solver()

    try:
        solver.load_model(args.checkpoint)
    except AttributeError:
        solver.load_generator_model(args.checkpoint)
        solver.load_discriminator_model('/src/Trained-FFHQ/ESRGAN-c1-2019.03.17-14:19:20/ESRGAN-disc-100000.mdl')

    solver.face_detector = dlib.get_frontal_face_detector()
    solver.face_localizer = dlib.shape_predictor(
        '/home/hacky/Dropbox/Skola/Projekty/POVa/du04/data/shape_predictor_5_face_landmarks.dat')
    solver.identityNN = dlib.face_recognition_model_v1(
        '/home/hacky/Dropbox/Skola/Projekty/POVa/du04/data/dlib_face_recognition_resnet_model_v1.dat')

    Dataset = getattr(import_module("datasets"), dataset_module)

    # TODO: add scale factor
    dataset = Dataset(dataset_path, length=1000, input_transform=build_input_transform(h, w, upscale_factor=args.scale))
    data_loader = DataLoader(dataset=dataset, batch_size=8)

    (_, _), \
        (psnr, psnr_diff), distanes = solver.evaluate(data_loader)#, identity_only=True)
    #print(round(psnr, 3))
    print("Mean Identity distance: " + str(round(distanes.mean(), 5)))
    print("Variance in Identity distance: " + str(round(distanes.var(), 5)))
    print("PSNR: " + str(round(psnr, 3)))
    print("PSNR diff: " + str(round(psnr_diff, 3)))


if __name__ == "__main__":
    main()
