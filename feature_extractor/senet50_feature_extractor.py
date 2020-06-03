import numpy as np
import pickle

from torch import Tensor, tensor, empty, zeros, cat, stack, sum, dot, dist
from torch.nn import Module, MSELoss
from torch.nn.functional import interpolate
from torchvision.transforms import CenterCrop
from typing import Union, Optional, Tuple, Dict

from .senet50_ft_dims_2048 import senet50_ft


class Senet50FeatureExtractor(Module):
    def __init__(self, detections_path, weights_path):
        super().__init__()
        with open(detections_path, "rb") as f:
            self.detections: Dict[str, Tuple] = pickle.load(f)

        self.extractor: Module = senet50_ft(weights_path).eval()
        self.register_buffer("mean", tensor((131.0912, 103.8827, 91.4953)).view((3, 1, 1)) / 255)

        self.descriptor_shape = 2048

    def center_crop(self, img: Tensor, size):
        img_h, img_w = img.shape[-2:]

        l = int(round((img_w - size) / 2.))
        t = int(round((img_h - size) / 2.))

        return img[:, l:l + size, t:t + size]

    def _input_transform(self, label, img: Tensor, img2: Optional[Tensor] = None):
        # im_h, im_w = img.shape[-2:]
        #
        # if label not in self.detections:
        #     return None
        #
        # # get detection
        # l, t, r, b = self.detections[label]
        # w = r - l
        # h = b - t
        #
        # w_extension = int(w * 1.3 // 2)
        # h_extension = int(h * 1.3 // 2)
        #
        # # extend face crop by 1.3
        # l = l - w_extension if l - w_extension > 0 else 0
        # r = r + w_extension if r + w_extension < im_w else im_w - 1
        # t = t - h_extension if t - h_extension > 0 else 0
        # b = b + h_extension if b + h_extension < im_h else im_h - 1
        #
        # # recompute detection size
        # w = r - l
        # h = b - t
        #
        # # resize detection so that shorter size has 256 pixels
        # factor = max(256 / w, 256 / h)
        # img = img[:, t:b, l:r]
        # img = interpolate(img, scale_factor=factor)

        img = interpolate(img.unsqueeze(0), (256, 256), mode="bicubic").squeeze(0)

        # return center crop of size 224
        img = self.center_crop(img, 224) - self.mean

        if img2 is None:
            return img

        # img2 = img2[:, t:b, l:r]
        # img2 = interpolate(img2, scale_factor=factor)
        img2 = interpolate(img2.unsqueeze(0), (256, 256), mode="bicubic").squeeze(0)
        img2 = self.center_crop(img2, 224) - self.mean
        return img, img2

    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def forward(self, label, img: Tensor, img2: Optional[Tensor] = None) -> Optional[Union[Tensor, Tuple]]:
        """
        Compute feature descriptor for given image using senet50 feature extractor. Uses the same detections from
        the first image, if two images supplied.

        :param label:
        :param img:
        :param img2:
        :return:
        """

        batch_size = len(label) if isinstance(label, Tuple) else 1

        if batch_size == 1:
            img = [img]
            img2 = [img2] if img2 is not None else None

        inputs1 = []
        inputs2 = []

        for idx, label in enumerate(label):
            # face not detected
            if label not in self.detections:
                continue

            if img2 is None:
                inputs1.append(self._input_transform(label, img[idx]))
            else:
                i1, i2 = self._input_transform(label, img[idx], img2[idx])
                inputs1.append(i1)
                inputs2.append(i2)

        batch_size = len(inputs1)

        if img2 is None:
            s = stack(inputs1)
            return self.extractor(s)[1].view((batch_size, self.descriptor_shape))

        return self.extractor(stack(inputs1))[1].squeeze(3).squeeze(2), \
               self.extractor(stack(inputs2))[1].squeeze(3).squeeze(2)

    @staticmethod
    def identity_dist(id1: Tensor, id2: Tensor):
        return dist(id1, id2)
