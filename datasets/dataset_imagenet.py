import torch.utils.data as data

from io import BytesIO
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageFile
from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop, ColorJitter
from torchvision.transforms.functional import crop

from .transforms import JpegCompress, AddNoise, MotionBlur, DefocusBlur

ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_image(path):
    return Image.open(path)


# TODO: blur
def build_input_transform(w, h, upscale_factor):
    return Compose([
        Resize((h // upscale_factor, w // upscale_factor)),
        RandomApply([AddNoise(), JpegCompress()], 0.5),
        ToTensor(),
    ])


def build_target_transform():
    return Compose([
        ToTensor(),
    ])


class DatasetImagenet(data.Dataset):
    producer = None

    def __init__(self, image_dir, upscale_factor=2, input_transform=None, target_transform=None, length=None):
        super(DatasetImagenet, self).__init__()
        self.image_filenames = [join(image_dir, file) for file in listdir(image_dir)]

        if "train" in image_dir:
            self.image_filenames = self.image_filenames[:50000]
        else:
            self.image_filenames = self.image_filenames[:4000]

        self.length = length if length and length <= len(self.image_filenames) else len(self.image_filenames)

        # load images to memory
        self.images = [Image.open(path).convert('RGB') for path in self.image_filenames]

        if input_transform:
            self.input_transform = input_transform
        else:
            self.input_transform = build_input_transform(256, 256, upscale_factor)

        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = build_target_transform()

    def __getitem__(self, index):
        # input_image = open_image(self.image_filenames[index])
        # input_image = crop(self.images[index], 1, 1, self.img_size[1]-2, self.img_size[0]-2)
        input_image = self.images[index]
        target = input_image.copy()

        input_image = self.input_transform(input_image)
        target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return self.length
