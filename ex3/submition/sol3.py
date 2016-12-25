import os
import functools
import numpy as np
import scipy.special
from scipy.misc import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.signal import convolve2d

# global parameter to plot as many figures as necessary
g_plot_index = 1

def index():
    # simulates a static variables of g_plot_index.
    # returns - number of figure before increment

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
    # converts relative paths to absolute
    # filename - relative path
    # returns - absolute path

    return os.path.join(os.path.dirname(__file__), filename)

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
    # im - image to expand
    # filter_vec - kernel used to build the gaussian pyramid
    # return - the expanded image after interpolation

    expand = np.zeros([im.shape[0] * 2, im.shape[1] * 2], dtype=np.float32)
    expand[1::2, 1::2] = im
    expand = scipy.ndimage.filters.convolve(expand, filter_vec, output=None, mode='mirror')
    expand = scipy.ndimage.filters.convolve(expand.transpose(), filter_vec, output=None, mode='mirror')
    return (expand.transpose()).astype(np.float32)

def build_laplacian_pyramid(im, max_levels, filter_size):

    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    filter_vec *= 2 #  on expansion the kernel should not be completely normalized
    pyr = [0] * len(gauss_pyr) # create the entire array for better complexity

    # using functional programing to avoid for loops
    pyr[:-1] = np.ndarray.tolist(np.array(gauss_pyr[:-1]) -  \
                                 np.array(list(map(functools.partial(expand, filter_vec), gauss_pyr[1:]))))

    # for i in range(len(pyr) - 1):
    #     pyr[i] = (gauss_pyr[i] - expand(filter_vec, gauss_pyr[i + 1])).astype(np.float32)

    pyr[-1] = gauss_pyr[-1]
    return pyr, filter_vec

def laplacian_to_image(lpyr, filter_vec, coeff):
    im = np.array([[0]]).astype(np.float32)

    vfunc = functools.partial(expand, filter_vec)
    pyr[:-1] = np.ndarray.tolist(np.array(gauss_pyr[:-1]) - np.array(list(map(vfunc, gauss_pyr[1:]))))

    for i  in range(len(lpyr) - 1):
        im = expand(filter_vec, im + lpyr[-(i + 1)] * coeff[-(i + 1)])

    return (im + lpyr[0] * coeff[0]).astype(np.float32)

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

    out_pyrl = (np.array(mask_gpyr) * np.array(im1_lpyr)) + (1 - np.array(mask_gpyr)) * np.array(im2_lpyr)

    return np.clip(laplacian_to_image(out_pyrl, filter_vec, np.ones(len(im1_lpyr))), 0, 1)


def examples(path_1, path_2, mask_path, max_levels, filter_size_im, filter_size_mask):
    im1 = read_image(relpath(path_1), 2)
    im2 = read_image(relpath(path_2), 2)
    mask = read_image(relpath(mask_path), 1) * 255

    im_blend = im1 * 0

    for i in range(3):
        im_blend[:,:,i] += pyramid_blending(im1[:,:,i], im2[:,:,i], mask, max_levels, filter_size_im, filter_size_mask)

    plt.figure(index())

    plt.subplot(221)
    plt.imshow(im1)
    plt.subplot(222)
    plt.imshow(im2)
    plt.subplot(223)
    plt.imshow(mask, plt.cm.gray)
    plt.subplot(224)
    plt.imshow(im_blend)

    plt.show()

    return im1, im2, mask, im_blend


def blending_example1():
    return examples('race2.jpg', 'givat2.jpg', 'mask2.jpg', 6, 1, 3)

def blending_example2():
    return examples("", "", "", 6, 3, 3)

#

im1, im2, mask, im_blend = blending_example1()
# im1, im2, mask, im_blend = blending_example2()



