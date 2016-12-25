import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
import os

import matplotlib.pyplot as plt


import scipy.special
from scipy.signal import convolve2d
from scipy import ndimage

g_plot_index = 1

def index():
    global g_plot_index
    g_plot_index += 1
    return g_plot_index - 1

def read_image(filename, representation):
    # filename - file to open as image
    # representation - is it a B&W or color image

    im = imread(filename)
    # check if it is a B&W image
    if(representation == 1):
        im = rgb2gray(im)
    # convert to float and normalize
    return (im / 255).astype(np.float32)

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
    im1_lpyr, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_lpyr, _ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_gpyr, _ = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)

    out_pyrl = np.array(mask_gpyr) * np.array(im1_lpyr) + (1 - np.array(mask_gpyr)) * np.array(im2_lpyr)

    return np.clip(laplacian_to_image(out_pyrl, filter_vec, np.ones(len(im1_lpyr))), 0, 1)


def examples(path_1, path_2, mask_path, max_levels, filter_size_im, filter_size_mask):
    im1 = read_image(relpath(path_1), 2)
    im2 = read_image(relpath(path_2), 2)
    mask = read_image(relpath(mask_path), 1) * 255

    im_blend = im1 * 0

    for i in range(3):
        im_blend[:,:,i] += pyramid_blending(im1[:,:,i], im2[:,:,i], mask, max_levels, filter_size_im, filter_size_mask)


    return im1, im2, mask, im_blend


def blending_example1():
    return examples('race2.jpg', 'givat2.jpg', 'mask2.jpg', 6, 3, 3)

def blending_example2():
    return examples("", "", "", 6, 3, 3)

#

im = read_image(relpath("givat2.jpg"), 1)
max_levels, filter_size = 10, 3
levels = 6


pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
# pyr, filter_vec = build_laplacian_pyramid(im, max_levels, filter_size)
coeff = np.ones(len(pyr))
re_im = laplacian_to_image(pyr, filter_vec, coeff)
# display_pyramid(pyr, levels)

# plt.figure(index())

# for i in range(len(pyr)):

    # half = len(pyr)  //  2 + 1
    # plt.subplot(2,half,i + 1)
    # plt.imshow(pyr[i], plt.cm.gray)

# plt.figure(index())
# plt.imshow(re_im, plt.cm.gray)

# plt.show()

# givat = read_image(relpath("givat2.jpg"), 1)
# race = read_image(relpath("race2.jpg"), 1)
# mask = read_image(relpath("mask2.jpg"), 1)

filter_size_im, filter_size_mask = 3,7

# mask *= 255
# mask[mask > 0.5] = 1
# mask[mask <= 0.5] = 0
# im_blend = pyramid_blending(race, givat, mask, max_levels, filter_size_im, filter_size_mask)




im1, im2, mask, im_blend = blending_example1()
# im1, im2, mask, im_blend = blending_example2()

plt.figure(index())

plt.subplot(2,2,1)
plt.imshow(im1)
plt.subplot(2,2,2)
plt.imshow(im2)
plt.subplot(2,2,3)
plt.imshow(mask, plt.cm.gray)
plt.subplot(2,2,4)
plt.imshow(im_blend)

plt.show()

