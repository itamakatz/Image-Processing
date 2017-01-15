import functools
import numpy as np
import scipy.special
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy.signal import convolve2d


# ===================== ex1 ===================== #

def read_image(filename, representation):
    # filename - file to open as image
    # representation - is it a B&W or color image
    im = imread(filename)
    # check if it is a B&W image
    if(representation == 1):
        im = rgb2gray(im)
    # convert to float and normalize
    return (im / 255).astype(np.float32)

# ===================== ex2 ===================== #

def create_ker(kernel_size):
    # kernel_size - odd integer
    # returns a binomial kernel of size kernel_size X kernel_size approximating a gausian
    bin = scipy.special.binom(kernel_size - 1, np.arange(kernel_size)).astype(np.int64)
    kernel = convolve2d(bin[np.newaxis, :], bin[:, np.newaxis])
    return kernel / np.sum(kernel)

def blur_spatial (im, kernel_size):
    return convolve2d(im, create_ker(kernel_size), mode='same')

# ===================== ex3 ===================== #

def create_filter_vec(filter_size):
    # creates a binomial coefficient of length filter_size
    # filter_size - length of the coefficient array
    # returns - the binomial coefficient array

    # special case of an odd number.
    if filter_size == 1: return np.array([[0]])
    conv_ker =  np.array([[1, 1]])
    filter = conv_ker
    # using an O(logN) algorithm to compute the filter
    log2 = np.log2(filter_size - 1)
    whole = np.floor(log2).astype(np.int64)
    rest = (2**(log2) - 2**(whole)).astype(np.int64)
    for i in range(whole):
        filter = convolve2d(filter, filter).astype(np.float32)
    for i in range(rest):
        filter = convolve2d(filter, conv_ker).astype(np.float32)
    # normalize
    return (filter / np.sum(filter)).astype(np.float32)

def build_gaussian_pyramid(im, max_levels, filter_size):
    # calc the filter array
    filter_vec = create_filter_vec(filter_size)
    # create the entire array for better complexity
    pyr = [0] * (np.min([max_levels, np.log2(im.shape[0]).astype(np.int64) - 3,
                         np.log2(im.shape[1]).astype(np.int64) - 3]))
    pyr[0] = im
    # for each iter, use the last iter to calc the current iter. note i transpose twice. once to calc
    # the y conv and the second to flip back the image
    for i in range(1, len(pyr)):
        pyr[i] = scipy.ndimage.filters.convolve(pyr[i - 1], filter_vec, output = None, mode = 'mirror')
        pyr[i] = scipy.ndimage.filters.convolve(pyr[i].transpose(), filter_vec, output = None, mode = 'mirror')
        pyr[i] = (pyr[i].transpose()[::2, ::2]).astype(np.float32)
    return pyr, filter_vec

def expand(filter_vec, im):
    # method that helps calculate an expanded image given an image and a kernel array
    # filter_vec - kernel used to build the gaussian pyramid
    # im - image to expand
    # return - the expanded image after interpolation
    expand = np.zeros([im.shape[0] * 2, im.shape[1] * 2], dtype=np.float32)
    expand[0::2, 0::2] = im
    expand = scipy.ndimage.filters.convolve(expand, filter_vec, output=None, mode='mirror')
    expand = scipy.ndimage.filters.convolve(expand.transpose(), filter_vec, output=None, mode='mirror')
    return (expand.transpose()).astype(np.float32)

def build_laplacian_pyramid(im, max_levels, filter_size):
    # build the laplacian pyramid from a given image
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    filter_vec *= 2 #  on expansion the kernel should not be completely normalized
    pyr = [0] * len(gauss_pyr) # create the entire array for better complexity
    # using functional programing to avoid a for loop
    pyr[:-1] = np.ndarray.tolist(np.array(gauss_pyr[:-1]) -  \
                                 np.array(list(map(functools.partial(expand, filter_vec), gauss_pyr[1:]))))
    pyr[-1] = gauss_pyr[-1]
    return pyr, filter_vec

def laplacian_to_image(lpyr, filter_vec, coeff):
    im = np.array([[0]]).astype(np.float32)
    # add leyers and expand for next iteration
    for i  in range(len(lpyr) - 1):
        im = expand(filter_vec, im + lpyr[-(i + 1)] * coeff[-(i + 1)])
    # mult by the coefficient
    return (im + lpyr[0] * coeff[0]).astype(np.float32)

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    # calc L1,L2, G1
    im1_lpyr, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_lpyr, _ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_gpyr, _ = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)
    # calc L_out
    out_pyrl = (np.array(mask_gpyr) * np.array(im1_lpyr)) + (1 - np.array(mask_gpyr)) * np.array(im2_lpyr)
    # clip to truncate the laplacian negative values
    return np.clip(laplacian_to_image(out_pyrl, filter_vec, np.ones(len(im1_lpyr))), 0, 1)