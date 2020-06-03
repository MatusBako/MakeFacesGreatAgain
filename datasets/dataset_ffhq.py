import torch.utils.data as data

from copy import deepcopy
from io import BytesIO
import numpy as np
from os import listdir
from os.path import join, basename
from PIL import Image
from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop, RandomChoice, RandomHorizontalFlip
from torchvision.transforms.functional import crop

from .transforms import JpegCompress, AddNoise, MotionBlur, DefocusBlur, ColorJitter

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_image(path):
    img = Image.open(path)
    ret = img.copy()
    img.close()
    return ret


def get_img_size(path):
    img = Image.open(path)
    return img.size


def build_input_transform(rng, h, w, upscale_factor: int):
    return Compose([
        ColorJitter(rng),
        RandomApply([AddNoise()], 0.3),
        RandomChoice([RandomApply([MotionBlur()], 0.3), RandomApply([DefocusBlur()], 0.3)]),
        RandomApply([JpegCompress()], 0.3),
        Resize((h // upscale_factor, w // upscale_factor), interpolation=Image.BICUBIC),
        ToTensor(),
    ])


def build_target_transform(rng):
    return Compose([
        ColorJitter(rng),
        ToTensor(),
    ])


class DatasetFFHQ(data.Dataset):
    def __init__(self, image_dir, upscale_factor: int = 2, input_transform=None, target_transform=None, length=None,
                 ):
        super().__init__()
        self.image_filenames = sorted(listdir(image_dir))
        self.image_paths = [join(image_dir, x) for x in self.image_filenames]
        self.image_labels = [join(basename(image_dir), fname) for fname in self.image_filenames]
        self.length = length if length and length <= len(self.image_paths) else len(self.image_paths)

        # load images to memory
        # self.images = [open_image(path) for path in self.image_filenames[:self.length]]

        self.w, self.h = get_img_size(self.image_paths[0])

        seed = np.random.randint(0, 2 ** 32 - 1)

        if input_transform is not None and target_transform is not None:
            self.input_transform = input_transform
            self.target_transform = target_transform
        elif input_transform is not None or target_transform is not None:
            assert False, "Both or neither input transformations must be set."
        else:
            self.input_transform = build_input_transform(np.random.RandomState(seed), self.w, self.h, upscale_factor)
            self.target_transform = build_target_transform(np.random.RandomState(seed))

    def __getitem__(self, index):
        # input_image = self.images[index]
        input_image = open_image(self.image_paths[index])
        target = input_image.copy()

        input_image = self.input_transform(input_image)
        target = self.target_transform(target)

        return self.image_labels[index], input_image, target

    def __len__(self):
        return self.length
