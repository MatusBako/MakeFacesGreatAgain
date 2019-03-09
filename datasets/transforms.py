from cv2 import resize, circle, INTER_AREA, LINE_AA
from io import BytesIO
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from types import LambdaType
from torchvision.transforms import functional as F, Compose

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


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# TODO: create color jitter, so that it's possible to get the same deformation twice
# all factors must be passed on call
def color_jitter(img, brightness, contrast, saturation, hue):
    if brightness > 0:
        brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        img = F.adjust_brightness(img, brightness_factor)

    if contrast > 0:
        contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        img = F.adjust_contrast(img, contrast_factor)

    if saturation > 0:
        saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        img = F.adjust_saturation(img, saturation_factor)

    if hue > 0:
        hue_factor = np.random.uniform(-hue, hue)
        img = F.adjust_hue(img, hue_factor)
    return img


class MotionBlur:
    def __call__(self, image):
        image = np.array(image)
        kernel = DefocusBlur.get_kernel(np.random.randint(1, 5), np.random.randint(1, 5))

        for c in range(3):
            image[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='wrap')

        return Image.fromarray(image)

    @staticmethod
    def get_kernel(start_samples=500, length=300, halflife=0.5, resolution=15):
        supersampling = 5

        # generate random acceleration
        a = np.random.randn(2, length+start_samples)

        # integrate speed
        a[0, :] = MotionBlur.ewma(a[0, :], halflife * length)
        a[1, :] = MotionBlur.ewma(a[1, :], halflife * length)

        # integrate position
        a = np.cumsum(a, axis=1)

        # skip first startSamples
        a = a[:, start_samples:]

        # center the kernel
        a = a - np.mean(a, axis=1).reshape((2, 1))

        # normalize size
        maxDistance = ((a[0, :]**2 + a[1, :]**2) ** 0.5).max()
        a = a / maxDistance

        psf, t, t = np.histogram2d(a[0, :], a[1, :], bins=resolution * supersampling,
                                   range=[[-1.0, +1.0], [-1.0, +1.0]], normed=True)
        psf = resize(psf, (resolution, resolution), interpolation=INTER_AREA)
        psf = psf.astype(np.float32)
        psf = psf / np.sum(psf)
        return psf

    # ref: https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    @staticmethod
    def ewma(data, halflife, offset=None, dtype=None, order='C', out=None):
        """
        Calculates the exponential moving average over a vector.
        Will fail for large inputs.
        :param data: Input data
        :param alpha: scalar float in range (0,1)
            The alpha parameter for the moving average.
        :param offset: optional
            The offset for the moving average, scalar. Defaults to data[0].
        :param dtype: optional
            Data type used for calculations. Defaults to float64 unless
            data.dtype is float32, then it will use float32.
        :param order: {'C', 'F', 'A'}, optional
            Order to use when flattening the data. Defaults to 'C'.
        :param out: ndarray, or None, optional
            A location into which the result is stored. If provided, it must have
            the same shape as the input. If not provided or `None`,
            a freshly-allocated array is returned.
        """
        data = np.array(data, copy=False)
        row_size = len(data)

        alpha = 1 - np.exp(np.log(0.5) / halflife)

        if dtype is None:
            if data.dtype == np.float32:
                dtype = np.float32
            else:
                dtype = np.float64
        else:
            dtype = np.dtype(dtype)

        if data.ndim > 1:
            # flatten input
            data = data.reshape(-1, order)

        if out is None:
            out = np.empty_like(data, dtype=dtype)
        else:
            assert out.shape == data.shape
            assert out.dtype == dtype

        if data.size < 1:
            # empty input, return empty array
            return out

        if offset is None:
            offset = data[0]

        alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

        # scaling_factors -> 0 as len(data) gets large
        # this leads to divide-by-zeros below
        scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                                   dtype=dtype)

        # create cumulative sum array
        np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                    dtype=dtype, out=out)
        np.cumsum(out, dtype=dtype, out=out)

        # cumsums / scaling
        out /= scaling_factors[-2::-1]

        if offset != 0:
            offset = np.array(offset, copy=False).astype(dtype, copy=False)
            # add offsets
            out += offset * scaling_factors[1:]

        return out


class DefocusBlur:
    def __call__(self, image):
        image = np.array(image)
        kernel = DefocusBlur.get_kernel(np.random.randint(1, 5), np.random.randint(1, 5))

        for c in range(3):
            image[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='wrap')

        return Image.fromarray(image)

    @staticmethod
    def get_kernel(scale, radius):
        psf_radius = int(radius * scale + 0.5)
        center = int((int(radius) + 2) * scale + scale / 2)
        psf = np.zeros((2 * center, 2 * center))
        circle(psf, (center, center), psf_radius, color=1.0, thickness=-1, lineType=LINE_AA)
        psf = resize(psf, dsize=(0, 0), fx=1.0 / scale, fy=1.0 / scale, interpolation=INTER_AREA)
        psf = psf / np.sum(psf)
        return psf#, int(center / scale)

