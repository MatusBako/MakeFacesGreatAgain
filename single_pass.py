#!/usr/bin/env python3

# from models.SRGAN import Solver
#from models.DBPN import Solver
#from models.DRCN import Solver
#from models.EDSR import Solver
#from models.ESPCN import Solver
#from .models.SRResNet import Solver
#from models.MFGN import Solver
#from models.SRCNN import Solver

import argparse
from cv2 import imwrite
from PIL import Image
import numpy as np
from numpy import mean, log10, round

from os import listdir
from os.path import dirname, abspath
from inspect import getframeinfo, currentframe
from sys import path

from torchvision.transforms import ToTensor, Compose

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('arch')
    parser.add_argument('checkpoint')
    parser.add_argument('input', help='Path to image.')
    parser.add_argument('output', help='Path to output.')
    parser.add_argument('-g', '--ground-truth', help='Original image for PSNR computation.')
    return parser.parse_args()


def main():
    args = get_args()
    
    #get project directory
    script = getframeinfo(currentframe()).filename
    proj_dir = dirname(abspath(script))

    path.append(proj_dir + "/models")
    assert args.arch in listdir(proj_dir + '/models')

    Solver = __import__(args.arch, fromlist=['Solver']).Solver
    solver = Solver()
    solver.load_model(args.checkpoint)

    name = args.arch
    image = Image.open(args.input)

    inp, out = solver.single_pass(image, True)

    if args.ground_truth is not None:
        transform = Compose([ToTensor()])
        im = Image.open(args.ground_truth)
        gt = transform(im).numpy().transpose((1, 2, 0))[:, :, ::-1]
        mse = ((gt - out)**2).mean(axis=None)

        gt *= 256
    else:
        mse = ((inp - out)**2).mean(axis=None)

    inp *= 256
    out *= 256

    if args.ground_truth is not None:
        imwrite(args.output + "/"+name+"-input.jpg", gt if args.ground_truth else inp)
    imwrite(args.output + "/"+name+".jpg", out)

    psnr = 10 * log10(1 / mse)
    print(round(psnr, 3))

if __name__ == "__main__":
    main()
