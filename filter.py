import numpy as np
import math
np.seterr(over='ignore')


# some constants
WIDTH = 4.0


# convolve image with gaussian filter
def smooth(src, sigma):
    mask = make_fgauss(sigma)
    mask = normalize(mask)
    dst = convolve_even(src, mask)
    return dst


# gaussian filter
def make_fgauss(sigma):
    sigma = max(sigma, 0.01)
    length = int(math.ceil(sigma * WIDTH)) + 1
    mask = np.zeros(shape=(length, length), dtype=float)
    for i in range(length):
        for j in range(length):
            mask[i, j] = math.exp(-0.5 * (math.pow(i / sigma, 2) + math.pow(j / sigma, 2)))
    return mask

# normalize mask so it integrates to one
def normalize(mask):
    sum = 4 * np.sum(np.absolute(mask)) - 3 * abs(mask[0]) - \
          2 * np.sum(np.absolute(mask[0, :])) - 2 * np.sum(np.absolute(mask[:, 0]))
    return np.divide(mask, sum)


# convolve src with mask.  output is flipped!
def convolve_even(src, mask):
    output = np.zeros(shape=src.shape, dtype=float)
    height, width = src.shape
    length = len(mask)

    for y in range(height):
        for x in range(width):
            sum = float(mask[0, 0] * src[y, x])
            for i in range(0, length):
                for j in range(0, length):
                    if i != 0 and j != 0:
                        sum += mask[i, j] * (src[max(y - j, 0), max(x - i, 0)] + src[max(y - j, 0), min(x + i, width - 1)] + \
                                             src[min(y + j, height - 1), min(x + i, width - 1)] + src[min(y + j, height - 1), max(x - i, 0)])
            output[y, x] = sum
    return output
