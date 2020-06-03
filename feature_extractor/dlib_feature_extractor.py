import dlib
import numpy as np
import pickle

from torch import Tensor
from typing import Optional, Union, Tuple


class DlibFeatureExtractor:
    def __init__(self, shape_predictor, extractor_path):
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_localizer = dlib.shape_predictor(shape_predictor)
        self.extractor = dlib.face_recognition_model_v1(extractor_path)

    @staticmethod
    def _input_transform(img: Tensor):
        # TODO: change net output from 0-1 to 0-255
        # BGR to RGB
        img = img.cpu().numpy() * 255
        img = np.transpose(img, (1, 2, 0))[:, :, ::-1]
        return img.astype(np.uint8)

    def __call__(self, _, img: Tensor, img2: Tensor = None) -> Optional[Union[np.ndarray, Tuple]]:
        """
        Compute feature descriptor for given image using dlib feature extractor. Uses the same detections from
        the first image, if two images supplied.

        :param img:
        :param img2:
        :return:
        """
        img = DlibFeatureExtractor._input_transform(img)
        detections = self.face_detector(img)

        # face not detected
        if not len(detections):
            return None if img2 is None else None, None

        face_shape = self.face_localizer(img, detections[0])

        if img2 is None:
            return np.array(self.extractor.compute_face_descriptor(img, face_shape))

        im1_descriptor = np.array(self.extractor.compute_face_descriptor(img, face_shape))

        img2 = DlibFeatureExtractor._input_transform(img2)
        im2_descriptor = np.array(self.extractor.compute_face_descriptor(img2, face_shape))

        return im1_descriptor, im2_descriptor

    @staticmethod
    def identity_dist(id1: np.ndarray, id2: np.ndarray):
        return np.sum(id1 ** 2) + np.sum(id2 ** 2) - 2 * np.dot(id1, id2)
