import numpy as np
import pickle

from torch import Tensor, nn
from typing import Union, Optional, Tuple, Dict

from .senet50_ft_dims_2048 import senet50_ft


class Senet50FeatureExtractor:
    def __init__(self, detections_path, weights_path):
        with open(detections_path, "rb") as f:
            self.detections: Dict[str, Tuple] = pickle.load(f)

        self.extractor: nn.Module = senet50_ft(weights_path)

        self.mean = np.array((131.0912, 103.8827, 91.4953)).reshape((1, 1, 3))

    def _input_transform(self, img: Tensor, img_path):
        # to use or not to use detection

        # TODO: get img size
        im_w = (im_h := 256)

        l, t, r, b = self.detections[img_path]
        w = r - l
        h = b - t

        w_extension = w * 1.3 // 2
        h_extension = h * 1.3 // 2

        # extend face crop by 1.3
        l = l - w_extension if l - w_extension > 0 else 0
        r = r + w_extension if r + w_extension < im_w else im_w - 1
        t = t - h_extension if t - h_extension > 0 else 0
        b = b + h_extension if b + h_extension < im_h else im_h - 1

        # recompute crop size
        w = r - l
        h = b - t

        img = (img * 255).clip(0, 255).astype(np.uint8)

        # TODO: skontroluj transpoziciu a poradie mean vectoru
        return np.transpose(img[t:b, l:r], (1, 2, 0))[:, :, ::-1] - self.mean

        # BGR to RGB
        img = img.cpu().numpy() * 255
        img = np.transpose(img, (1, 2, 0))[:, :, ::-1]
        return img.astype(np.uint8)


    def __call__(self, label, img: Tensor, img2: Tensor = None, path=None) -> Optional[Union[np.ndarray, Tuple]]:
        """
        Compute feature descriptor for given image using dlib feature extractor. Uses the same detections from
        the first image, if two images supplied.

        :param label:
        :param img:
        :param img2:
        :param path: path to image used for getting stored detections
        :return:
        """

        raise NotImplementedError("Need Dataset to also return subpath")

        # face not detected
        if path not in self.detections:
            return None if img2 is None else None, None

        img = Senet50FeatureExtractor._input_transform(img, path)

        if img2 is None:
            return np.array(self.extractor(img, face_shape))

        im1_descriptor = np.array(self.extractor(img, face_shape))

        img2 = Senet50FeatureExtractor._input_transform(img2)
        im2_descriptor = np.array(self.extractor(img2, face_shape))

        return im1_descriptor, im2_descriptor

    @staticmethod
    def identity_dist(id1: np.ndarray, id2: np.ndarray):
        return np.sum(id1 ** 2) + np.sum(id2 ** 2) - 2 * np.dot(id1, id2)
