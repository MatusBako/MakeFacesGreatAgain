#!/usr/bin/env python3

"""
Script used for enlarging single photo using given trained model.
"""

# from models.SRGAN import Solver
#from models.DBPN import Solver
#from models.DRCN import Solver
#from models.EDSR import Solver
#from models.ESPCN import Solver
#from .models.SRResNet import Solver
#from models.MFGN import Solver
#from models.SRCNN import Solver

import argparse
from cv2 import imwrite, resize
from PIL import Image
import numpy as np
from numpy import mean, log10, round

from os import listdir
from os.path import dirname, abspath
from inspect import getframeinfo, currentframe
from sys import path

from torchvision.transforms import ToTensor, Compose, CenterCrop, Resize

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('arch')
    parser.add_argument('checkpoint')
    parser.add_argument('dataset', type=str.lower, choices=["celeba", "ffhq"])
    parser.add_argument('input', help='Path to image.')
    parser.add_argument('output', help='Path to output.')
    parser.add_argument('-g', '--ground-truth', help='Original image for PSNR computation.')
    return parser.parse_args()


def main():
    args = get_args()

    try:
        # Solver = __import__("models." + args.arch, fromlist=['Solver']).Solver
        Solver = __import__("src", fromlist=['Solver']).Solver
    except Exception:
        print(f"Can't import model {args.arch}. (check PYTHONPATH)")
        raise

    solver = Solver(cfg=None, mode="single")

    try:
        solver.load_model(args.checkpoint, mode="single")
    except AttributeError:
        solver.load_generator_model(args.checkpoint, mode="single")

    name = args.arch

    upscale_factor = 4
    image = Image.open(args.input)

    if args.dataset == "celeba":
        image = image.crop((1, 5, 177, 213))
        inp = image.resize((image.size[0] // upscale_factor, image.size[1] // upscale_factor), Image.BICUBIC)
    elif args.dataset == "ffhq":
        inp = image.resize((image.size[0] // upscale_factor, image.size[1] // upscale_factor), Image.BICUBIC)
    else:
        raise Exception("Wrong dataset.")

    inp, out = solver.single_pass(inp, False)

    transform = Compose([ToTensor()])

    if args.ground_truth is not None:
        im = Image.open(args.ground_truth)
        gt = transform(im).numpy().transpose((1, 2, 0))[:, :, ::-1]
        mse = ((gt - out)**2).mean(axis=None)
        gt *= 256
    else:
        orig = transform(image).numpy().transpose((1, 2, 0))[:, :, ::-1]
        mse = ((orig - out)**2).mean(axis=None)

    inp *= 256
    out *= 256

    if args.ground_truth is not None:
        imwrite(args.output + "/"+name+"-input.png", gt if args.ground_truth else image)

    imwrite(args.output, out)
    # image.save(args.output + "/orig.png")

    psnr = 10 * log10(1 / mse)
    print(round(psnr, 3))


if __name__ == "__main__":
    main()
