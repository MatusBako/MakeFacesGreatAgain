#!/usr/bin/env python3

import argparse
from numpy import log10, round

from os import listdir
from os.path import dirname, abspath
from inspect import getframeinfo, currentframe
from sys import path

from torch.utils.data import DataLoader
from importlib import  import_module


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('arch')
    parser.add_argument('checkpoint')
    parser.add_argument('dataset_module', help='Dataset module name.')
    parser.add_argument('dataset_path', help='Path to dataset.')
    parser.add_argument('-s', '--scale')
    return parser.parse_args()


def main():
    args = get_args()
    
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
    solver.load_model(args.checkpoint)

    Dataset = getattr(import_module("datasets"), args.dataset_module)

    # TODO: add scale factor
    dataset = Dataset(args.dataset_path)
    data_loader = DataLoader(dataset=dataset, batch_size=1)

    _, psnr, _ = solver.evaluate(data_loader)
    print(round(psnr, 3))

if __name__ == "__main__":
    main()
