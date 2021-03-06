from cv2 import resize, circle, INTER_AREA, LINE_AA
from io import BytesIO
import numpy as np
from numpy.random import RandomState
from PIL import Image
from scipy.signal import convolve2d
from torchvision.transforms import functional as F, Compose


class JpegCompress:
    def __init__(self, rng=np.random):
        """
        np.random.RandomState(rng_seed) can be passed
        """
        self.rng = rng

    def __call__(self, image):
        out = BytesIO()
        quality = 95 - int(np.floor(np.clip(self.rng.exponential(scale=3), 0, 20)))
        image.save(out, format='jpeg', quality=quality)
        out.seek(0)
        return Image.open(out)


class AddNoise:
    def __init__(self, rng=np.random):
        """
        np.random.RandomState(rng_seed) can be passed
        """
        self.rng = rng

    def __call__(self, image):
        image = np.array(image, dtype=np.int32)

        noise_range = self.rng.randint(1, 3)
        noise = self.rng.normal(0, noise_range, size=image.shape).astype(np.int32)
        return Image.fromarray(np.clip(image + noise, 0, 255).astype(np.uint8))


class ColorJitter:
    def __init__(self, rng=np.random, brightness=0.12, contrast=0.12, saturation=0.10, hue=0.04):
        """
        np.random.RandomState(rng_seed) can be passed
        """
        self.rng = rng
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        return self.color_jitter(img)

    # TODO: create color jitter, so that it's possible to get the same deformation twice (use seed for rng)
    # all factors must be passed on call
    def color_jitter(self, img):
        if self.brightness > 0:
            brightness_factor = self.rng.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = F.adjust_brightness(img, brightness_factor)

        if self.contrast > 0:
            contrast_factor = self.rng.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            img = F.adjust_contrast(img, contrast_factor)

        if self.saturation > 0:
            saturation_factor = self.rng.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            img = F.adjust_saturation(img, saturation_factor)

        if self.hue > 0:
            hue_factor = self.rng.uniform(-self.hue, self.hue)
            img = F.adjust_hue(img, hue_factor)
        return img


class MotionBlur:
    def __init__(self, rng=np.random):
        """
        np.random.RandomState(rng_seed) can be passed
        """
        self.rng = rng

    def __call__(self, image):
        image = np.array(image)

        kernel = self.get_kernel(resolution=self.rng.randint(5, 12))

        for c in range(3):
            image[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='wrap')

        return Image.fromarray(image)

    def get_kernel(self, start_samples=500, length=300, halflife=0.5, resolution=15):
        supersampling = 5

        # generate random acceleration
        a = self.rng.randn(2, length + start_samples)

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
        maxDistance = ((a[0, :] ** 2 + a[1, :] ** 2) ** 0.5).max()
        a = a / maxDistance

        psf, t, t = np.histogram2d(a[0, :], a[1, :], bins=resolution * supersampling,
                                   range=[[-1.0, +1.0], [-1.0, +1.0]], normed=True)
        psf = resize(psf, (resolution, resolution), interpolation=INTER_AREA)
        psf = psf.astype(np.float32)
        psf = psf / np.sum(psf)
        return psf

    @staticmethod
    def ewma(data, halflife, offset=None, dtype=None, order='C', out=None):
        """
        Calculates the exponential moving average over a vector.
        Will fail for large inputs.

        Source: https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm

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
    def __init__(self, rng=np.random):
        """
        np.random.RandomState(rng_seed) can be passed
        """
        self.rng = rng
        self.max_scale = 8
        self.max_radius = 5

    def __call__(self, image):
        image = np.array(image)

        kernel = DefocusBlur.get_kernel(self.rng.randint(1, self.max_scale), self.rng.randint(1, self.max_radius))

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
        return psf  # , int(center / scale)
