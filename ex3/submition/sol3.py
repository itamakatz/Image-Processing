import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
import os

import scipy.special
from scipy.signal import convolve2d
from scipy import ndimage

def read_image(filename, representation):
    # filename - file to open as image
    # representation - is it a B&W or color image

    im = imread(filename)
    # check if it is a B&W image
    if(representation == 1):
        im = rgb2gray(im)
    # convert to float and normalize
    return im.astype(np.float32) / 255

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

###########################################################################################################################3
def create_filter_vec(filter_size):

    conv_ker =  np.array([[1, 1]])
    filter = conv_ker

    #  using a O(logN) to compute the filter
    log2 = np.log2(filter_size)
    whole = np.floor(log2).astype(np.int64)
    rest = (filter_size - whole).astype(np.int64)

    for i in range(whole):
        filter = convolve2d(filter, filter).astype(np.float32)
    for i in range(2**rest):
        filter = convolve2d(conv_ker, filter).astype(np.float32)

    # normalize
    return filter / np.sum(filter)

def build_gaussian_pyramid(im, max_levels, filter_size):
    filter_vec = create_filter_vec(filter_size)

    pyr[np.min([max_levels, np.log2(im.shape[0]) - 4, np.log2(im.shape[1]) - 4])] = 0

    for i in range(len(pyr)):
        im = scipy.ndimage.filters.convolve(im, filter_vec, output = None, mode = 'mirror')
        im = scipy.ndimage.filters.convolve(im.transpose(), filter_vec, output = None, mode = 'mirror')
        pyr[i] = im[::2]

    return pyr

def build_laplacian_pyramid(im, max_levels, filter_size):
    filter_vec = create_filter_vec(filter_size) * 2
    gauss_pyr = build_gaussian_pyramid(im, max_levels + 1, filter_size)

    pyr[len(gauss_pyr) - 1] = 0

    for i in range(len(pyr)):
        tmp = np.zeros(gauss_pyr[i].shape[0], gauss_pyr[i].shape[1])
        tmp[::2,::2] = gauss_pyr[i + 1]
        tmp = scipy.ndimage.filters.convolve(tmp, filter_vec, output = None, mode = 'mirror')
        tmp = scipy.ndimage.filters.convolve(tmp.transpose(), filter_vec, output = None, mode = 'mirror')
        pyr[i] = gauss_pyr[i] - tmp

    return pyr

def laplacian_to_image(lpyr, filter_vec, coeff):
    lpyr *= coeff
    img = sum
    return

def render_pyramid(pyr, levels):
    return

def display_pyramid(pyr, levels):
    return

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    return

def blending_example1():
    return

def blending_example2():
    return

#

im, max_levels, filter_size = 0,0,0

pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

pyr, filter_vec = build_laplacian_pyramid(im, max_levels, filter_size)

#

lpyr, filter_vec, coeff = 0,0,0

img = laplacian_to_image(lpyr, filter_vec, coeff)

#

pyr, levels = 0,0

res = render_pyramid(pyr, levels)

display_pyramid(pyr, levels)

#

im1, im2, mask, max_levels, filter_size_im, filter_size_mask = 0,0,0,0,0,0

im_blend = pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)

#

im1, im2, mask, im_blend = blending_example1()

im1, im2, mask, im_blend = blending_example2()
