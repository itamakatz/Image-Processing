import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
import os

import matplotlib.pyplot as plt


import scipy.special
from scipy.signal import convolve2d
from scipy import ndimage

plot_index = 1

def index():
    global plot_index
    plot_index += 1
    return plot_index - 1

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
    return filter / np.sum(filter)

def build_gaussian_pyramid(im, max_levels, filter_size):

    filter_vec = create_filter_vec(filter_size)

    pyr = [0] * (np.min([max_levels, np.log2(im.shape[0]).astype(np.int64) - 3, np.log2(im.shape[1]).astype(np.int64) - 3]))

    pyr[0] = im

    for i in range(1, len(pyr)):
        pyr[i] = scipy.ndimage.filters.convolve(pyr[i - 1], filter_vec, output = None, mode = 'mirror')
        pyr[i] = scipy.ndimage.filters.convolve(pyr[i].transpose(), filter_vec, output = None, mode = 'mirror')
        pyr[i] = (pyr[i].transpose()[::2, ::2]).astype(np.float32)

    return pyr, filter_vec

def expand(im, filter_vec):
    expand = np.zeros([im.shape[0] * 2, im.shape[1] * 2], dtype=np.float32)
    expand[1::2, 1::2] = im
    expand = scipy.ndimage.filters.convolve(expand, filter_vec, output=None, mode='mirror')
    expand = scipy.ndimage.filters.convolve(expand.transpose(), filter_vec, output=None, mode='mirror')
    return (expand.transpose()).astype(np.float32)

def build_laplacian_pyramid(im, max_levels, filter_size):

    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels + 1, filter_size)
    filter_vec *= 2

    # ============================== np.vectorize()
    # pyr = [0] * len(gauss_pyr)
    # v_expand = np.vectorize(expand, excluded=['filter_vecc'])
    # expanded = v_expand(gauss_pyr[:-1], filter_vecc=filter_vec)
    # pyr[:-1] = gauss_pyr[:-1] - expanded
    # pyr[-1] = gauss_pyr[-1]


    pyr = [0] * len(gauss_pyr)

    for i in range(len(pyr) - 1):
        pyr[i] = (gauss_pyr[i] - expand(gauss_pyr[i + 1], filter_vec)).astype(np.float32)

    pyr[-1] = gauss_pyr[-1]
    return pyr, filter_vec

def laplacian_to_image(lpyr, filter_vec, coeff):
    im = np.array([[0]]).astype(np.float32)
    for i  in range(len(lpyr) - 1):
        im = expand(im + lpyr[-(i + 1)] * coeff[-(i + 1)], filter_vec)
    return (im + lpyr[0] * coeff[0]).astype(np.float32)

def calc_log_len(x, levels):
    return

def render_pyramid(pyr, levels):
    length = (pyr[0].shape[1] * 2 * (1 - 2**(-levels)))

    return np.zeros([pyr[0].shape[0], int(length)], dtype=np.float32)

def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    length = 0
    for i in range(levels):
        a = pyr[i].shape[0]
        b = pyr[i].shape[1] + length
        res[0 : a, length : b] = pyr[i]
        length += pyr[i].shape[1]
    plt.figure(index())
    plt.imshow(np.clip(res, 0, 1), plt.cm.gray)
    return

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    return

def blending_example1():
    return

def blending_example2():
    return

#

im = read_image(relpath("givat2.jpg"), 1)
max_levels, filter_size = 10, 3
levels = 6


pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
# pyr, filter_vec = build_laplacian_pyramid(im, max_levels, filter_size)
coeff = np.ones(len(pyr))
re_im = laplacian_to_image(pyr, filter_vec, coeff)
display_pyramid(pyr, levels)

plt.figure(index())

for i in range(len(pyr)):

    half = len(pyr)  //  2 + 1
    plt.subplot(2,half,i + 1)
    plt.imshow(pyr[i], plt.cm.gray)

plt.figure(index())
plt.imshow(re_im, plt.cm.gray)

plt.show()


# ================= test create_filter_vec ================= #
# for i in range (3, 9, 2):
#     print("i is: " + str(i))
#     print(create_filter_vec(i))
#     print("\n")

print("done")


# lpyr, filter_vec, coeff = 0,0,0
#
#
#
# #
#
# pyr, levels = 0,0
#
# res = render_pyramid(pyr, levels)
#
# display_pyramid(pyr, levels)
#
# #
#
# im1, im2, mask, max_levels, filter_size_im, filter_size_mask = 0,0,0,0,0,0
#
# im_blend = pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
#
# #
#
# im1, im2, mask, im_blend = blending_example1()
#
# im1, im2, mask, im_blend = blending_example2()
