import torch.utils.data as data

from io import BytesIO
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import Compose, RandomApply, Resize, ToTensor, CenterCrop
from torchvision.transforms.functional import crop


def open_image(path):
    return Image.open(path)


def get_img_size(path):
    img = Image.open(path)
    return img.size


class JpegCompress:
    def __call__(self, image):
        out = BytesIO()
        quality = 95 - int(np.floor(np.clip(np.random.exponential(scale=3), 0, 20)))
        image.save(out, format='jpeg', quality=quality)
        out.seek(0)
        return Image.open(out)


class AddNoise:
    def __call__(self, image):
        image = np.array(image, dtype=np.int32)

        noise_range = np.random.randint(1, 3)
        noise = np.random.normal(0, noise_range, size=image.shape).astype(np.int32)
        return Image.fromarray(np.clip(image + noise, 0, 255).astype(np.uint8))


# TODO: blur
def build_input_transform(h, w, upscale_factor):
    return Compose([
        CenterCrop((h, w)),
        Resize((h // upscale_factor, w // upscale_factor)),#, interpolation=Image.BICUBIC),
        RandomApply([AddNoise(), JpegCompress()], 0.5),
        ToTensor(),
    ])


def build_target_transform():
    return Compose([
        CenterCrop((208, 176)),
        ToTensor(),
    ])


class DatasetCelebA(data.Dataset):
    producer = None

    def __init__(self, image_dir, upscale_factor=2, input_transform=None, target_transform=None, length=None):
        super(DatasetCelebA, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]

        if "train" in image_dir:
            self.image_filenames = self.image_filenames

        self.length = length if length and length <= len(self.image_filenames) else len(self.image_filenames)

        # load images to memory
        self.images = [Image.open(path) for path in self.image_filenames]

        self.w, self.h = get_img_size(self.image_filenames[0])

        if input_transform:
            self.input_transform = input_transform
        else:
            self.input_transform = build_input_transform(208, 176, upscale_factor)

        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = build_target_transform()

    def __getitem__(self, index):
        # input_image = open_image(self.image_filenames[index])
        #input_image = crop(self.images[index], 1, 1, self.img_size[1]-2, self.img_size[0]-2)
        input_image = self.images[index]
        target = input_image.copy()

        input_image = self.input_transform(input_image)
        target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return self.length
