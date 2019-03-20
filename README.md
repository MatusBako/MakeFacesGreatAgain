# MakeFacesGreatAgain

This repository contains practical part of my [masters thesis](https://www.overleaf.com/read/rmkmpnsbbdpv) focused on face super-resolution. I try to keep the implementation as modular as possible,  so the solvers are abstract classes and every architecture implements its loss. For computing face distance, I use dlib library with models from [this ](https://github.com/davisking/dlib-models) repository.

### Run

To start training, run `./main.py` with path to config file as the only argument.

### Configuration file format

The format is currently not explained, but examles of config files can be found in `configs` folder. 
The examples are easily understable and editable. 

### Datasets

For training I use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (aligned and cropped to 208x176) and  [FFHQ](https://github.com/NVlabs/ffhq-dataset) (cropped to 256x256) dataset.

### Evaluation

TODO 
