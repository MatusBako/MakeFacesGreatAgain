import torch.utils.data as data

from io import BytesIO
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop, RandomChoice
from torchvision.transforms.functional import crop

from .transforms import JpegCompress, AddNoise, MotionBlur, DefocusBlur


def open_image(path):
    return Image.open(path)


def get_img_size(path):
    img = Image.open(path)
    return img.size


def build_input_transform(h, w, upscale_factor):
    return Compose([
        Resize((h // upscale_factor, w // upscale_factor), interpolation=Image.BICUBIC),
        RandomApply([AddNoise()], 0.3),
        RandomChoice([RandomApply([MotionBlur()], 0.3), RandomApply([DefocusBlur()], 0.3)]),
        RandomApply([JpegCompress()], 0.3),
        ToTensor(),
    ])


def build_target_transform():
    return Compose([
        ToTensor(),
    ])


# TODO: possibly add producer-consumner (multiprocessing) to use less RAM
class DatasetFFHQ(data.Dataset):
    def __init__(self, image_dir, upscale_factor=2, input_transform=None, target_transform=None, length=None):
        super().__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]

        self.length = length if length and length <= len(self.image_filenames) else len(self.image_filenames)

        # load images to memory
        self.images = [Image.open(path) for path in self.image_filenames]

        self.w, self.h = get_img_size(self.image_filenames[0])

        if input_transform:
            self.input_transform = input_transform
        else:
            self.input_transform = build_input_transform(self.w, self.h, upscale_factor)

        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = build_target_transform()

    def __getitem__(self, index):
        input_image = self.images[index]
        target = input_image.copy()

        input_image = self.input_transform(input_image)
        target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return self.length
