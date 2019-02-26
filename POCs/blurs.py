#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt

# motion blur
def generateMotionBlur(RNG, startSamples=500, length=300, halflife=0.5, resolution=15): 
    superSampling = 5 

    # generate random acceleration 
    a = RNG.randn( 2, length+startSamples) 

    # integrate speed 
    a[0, :] = ewma(a[0, :], halflife * length) 
    a[1, :] = ewma(a[1, :], halflife * length) 

    # integrate position 
    a = np.cumsum(a, axis=1) 

    # skip first startSamples 
    a = a[:, startSamples:] 

    # center the kernel 
    a = a - np.mean(a, axis=1).reshape((2,1)) 

    # normalize size 
    maxDistance = ((a[0,:]**2 + a[1,:]**2) ** 0.5).max() 
    a = a / maxDistance 

    psf, t, t = np.histogram2d(a[0, :], a[1, :], bins=resolution * superSampling, range=[[-1.0, +1.0], [-1.0, +1.0]], normed=True) 
    psf = cv2.resize(psf, (resolution, resolution), interpolation=cv2.INTER_AREA) 
    psf = psf.astype(np.float32) 
    psf = psf / np.sum(psf) 
    return psf 

# out of focus blur kernel
def generatePSF(scale, radius): 
    psfRadius = int(radius * scale + 0.5) 
    center = int((int(radius) + 2) * scale + scale / 2) 
    psf = np.zeros((2 * center, 2 * center)) 
    cv2.circle(psf, (center, center), psfRadius, color=1.0, thickness=-1, lineType=cv2.LINE_AA) 
    psf = cv2.resize(psf, dsize=(0, 0), fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_AREA) 
    psf = psf / np.sum(psf) 
    return psf, int(center / scale) 

# TODO: ref "numpy ewma" (stack overflow myslim)
# https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
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

def saturation():
    img = PIL.Image.open('bus.png')
    converter = PIL.ImageEnhance.Color(img)
    img2 = converter.enhance(0.5)

if __name__ == "__main__":
    plt.imshow(generateMotionBlur(np.random))
    plt.show()

    plt.imshow(generatePSF(2, 5)[0])
    plt.show()

    

