import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
import os

import scipy.special
from scipy.signal import convolve2d

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

def build_gaussian_pyramid(im, max_levels, filter_size):
    return

def build_laplacian_pyramid(im, max_levels, filter_size):
    return

def laplacian_to_image(lpyr, filter_vec, coeff):
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
