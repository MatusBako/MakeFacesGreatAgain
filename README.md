# MakeFacesGreatAgain

This repository contains practical part of my [masters thesis](https://www.overleaf.com/read/rmkmpnsbbdpv) focused on face super-resolution. I try to keep the implementation as modular as possible, so the solvers are abstract classes and every architecture implements its loss. For computing face identity distance, I use dlib library with models from [this repository](https://github.com/davisking/dlib-models) and pretrained SE-ResNet architecture from [VGG-Face2 repository](https://github.com/ox-vgg/vgg_face2).

### Run

To use neural networks on NVIDIA GPU, ensure that you have proper NVIDIA and CUDA drivers installed.

To install Python modules, use: `pip install -r requirements.txt`.

To start training, run `./main.py config_path` with path to configuration file as the only argument.

### Configuration file format

The format is currently not deeply explained (and might change slightly), but examles of config files can be found in `configs` folder. The examples are easily understable and editable. 

### Datasets

For training I use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (aligned and cropped to 208x176) and  [FFHQ](https://github.com/NVlabs/ffhq-dataset) (cropped to 256x256) dataset.

### Evaluation

TODO 
