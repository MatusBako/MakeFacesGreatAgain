import torch.utils.data as data

from io import BytesIO
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop
from torchvision.transforms.functional import crop

# TODO: blur
def build_input_transform():
    return Compose([
        ToTensor(),
    ])


def build_target_transform():
    return Compose([
        ToTensor(),
    ])


class DatasetSet5(data.Dataset):
    producer = None

    def __init__(self, image_dir, upscale_factor=2, input_transform=None, target_transform=None, length=None):
        super(DatasetSet5, self).__init__()
        self.lr_image_filenames = [join(image_dir, x) for x in filter(lambda x: 'LR' in x, listdir(image_dir))]
        self.hr_image_filenames = [join(image_dir, x) for x in filter(lambda x: 'HR' in x, listdir(image_dir))]

        self.length = length if length and length <= len(self.lr_image_filenames) else len(self.lr_image_filenames)

        self.lr_image_filenames.sort()
        self.hr_image_filenames.sort()

        # load images to memory
        self.lr_images = [Image.open(path) for path in self.lr_image_filenames]
        self.hr_images = [Image.open(path) for path in self.hr_image_filenames]

        if input_transform:
            self.input_transform = input_transform
        else:
            self.input_transform = build_input_transform()

        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = build_target_transform()

    def __getitem__(self, index):
        input_image = self.lr_images[index]
        target_image = self.hr_images[index]

        # this is only testing dataset, no deformation needed
        input_image = self.target_transform(input_image)
        target_image = self.target_transform(target_image)

        return input_image, target_image

    def __len__(self):
        return self.length
