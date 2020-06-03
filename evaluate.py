#!/usr/bin/env python3

import argparse
from numpy import log10, round

from configparser import ConfigParser
from os import listdir
from os.path import dirname, realpath
from sys import path

from torch.utils.data import DataLoader
from importlib import import_module

from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop, RandomChoice
from PIL import Image

from feature_extractor import DlibFeatureExtractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Parsed config object")
    parser.add_argument('checkpoint', help="Path to checkpoint file.")
    parser.add_argument('dataset', type=str.lower, default='ffhq', choices=['ffhq', 'celeba'])
    parser.add_argument('-m', "--module", default="src")
    parser.add_argument('-s', '--scale', default=4, type=int)
    return parser.parse_args()


def build_input_transform(h=256, w=256, upscale_factor=4):
    return Compose([
        CenterCrop((h, w)),
        Resize((h // upscale_factor, w // upscale_factor), interpolation=Image.BICUBIC),
        ToTensor(),
    ])

def build_target_transform(h=208, w=176):
    return Compose([
        CenterCrop((h, w)),
        ToTensor(),
    ])

# class Config:
#     def __init__(self, arch, batch_size):
#         self.device = 'cuda'
#         self.batch_size = batch_size
#         self.generator_module = arch


def main():
    args = get_args()

    if args.dataset == 'ffhq':
        dataset_module = 'DatasetFFHQ'
        dataset_path = '/run/media/hacky/DATA2/FFHQ/validate'
        h = 256
        w = 256

        input_transform = build_input_transform(h, w, upscale_factor=args.scale)
        target_transform = ToTensor()

    elif args.dataset == 'celeba':
        dataset_module = 'DatasetCelebA'
        dataset_path = '/home/hacky/datasets/CelebA/Img/aligned/validate'
        h = 208
        w = 176

        input_transform = build_input_transform(h, w, upscale_factor=args.scale)
        target_transform = build_target_transform(h, w)
    else:
        raise NotImplementedError("This dataset is not implemented.")

    config = ConfigParser()
    config.optionxform = str
    config.read(args.config)

    # get project directory
    proj_dir = realpath(dirname(__file__))
    path.append(proj_dir)

    # TODO: assert existence of dataset module

    Solver = __import__(args.module, fromlist=['Solver']).Solver
    solver = Solver(config, "eval")

    if hasattr(solver, 'load_model'):
        solver.load_model(args.checkpoint, "eval")
    else:
        solver.load_generator_model(args.checkpoint)
        # print("Using hardcoded discriminator!")
        # solver.load_discriminator_model('/src/Trained-Celeba/Up4/ESRGAN-2019.03.22-01:31:28/ESRGAN-disc-100000.mdl')

    # solver.identity_extractor = DlibFeatureExtractor(
    #     shape_predictor='/home/hacky/Dropbox/Skola/Projekty/POVa/du04/data/shape_predictor_5_face_landmarks.dat',
    #     extractor_path='/home/hacky/Dropbox/Skola/Projekty/POVa/du04/data/dlib_face_recognition_resnet_model_v1.dat')

    Dataset = getattr(import_module("datasets"), dataset_module)

    # TODO: add scale factor
    dataset = Dataset(dataset_path, length=5000, input_transform=input_transform, target_transform=target_transform)
    data_loader = DataLoader(dataset=dataset, batch_size=8)

    _, (psnr, psnr_diff, ssim), distances = solver.evaluate(data_loader)
    # print(round(psnr, 3))
    print(" ".join(["MeanID:" + str(round(distances.mean(), 5)),
                    "PSNR:" + str(round(psnr, 3)),
                    "PSNR_diff:" + str(round(psnr_diff, 3)),
                    "SSIM:" + str(round(ssim, 3))
                    ])
          )


if __name__ == "__main__":
    main()
