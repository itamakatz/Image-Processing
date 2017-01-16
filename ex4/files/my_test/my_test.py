from scipy.ndimage import map_coordinates
import os
import functools
import sol4_utils as ut
import sol4_utils_liav as sut
import sol4_add as ad
import matplotlib.pyplot as plt
import scipy.signal as sg
import numpy as np
import itertools
import heapq

from scipy.signal import convolve2d

def relpath(filename):
    # converts relative paths to absolute
    # filename - relative path
    # returns - absolute path
    return os.path.join(os.path.dirname(__file__), filename)

def harris_corner_detector(im):
    der = np.array([[1,0,-1]])

    I_x = convolve2d(im, der, mode='same')
    I_y = convolve2d(im, np.transpose(der), mode='same')

    M = np.array([np.square(I_x), np.multiply(I_x, I_y), np.square(I_y)])
    M = np.array(list(map(functools.partial(ut.blur_spatial_rev, 3), M)))
    R = np.multiply(M[0], M[2]) - np.square(M[1]) - 0.04 * np.square((M[0] + M[2]))

    return np.transpose(np.nonzero(ad.non_maximum_suppression(R)))

def harris_corner_detector_liav(im):
    #############################################################
    # implements harris method for corner detection
    #############################################################
    dx = np.array([[1, 0, -1]])
    dy = dx.transpose()
    Ix = sg.convolve2d(im, dx, mode="same")
    Iy = sg.convolve2d(im, dy, mode="same")
    # blurring
    Ix_blur = sut.blur_spatial(Ix ** 2, 3)
    Iy_blur = sut.blur_spatial(Iy ** 2, 3)
    IxIy_blur = sut.blur_spatial(Ix * Iy, 3)
    # compute determinant and trace of M
    det = Ix_blur * Iy_blur - IxIy_blur ** 2
    tr = Ix_blur + Iy_blur
    R = det - 0.04 * (tr ** 2)
    # return R
    return np.transpose(np.nonzero(sut.non_maximum_suppression(R)))

a = harris_corner_detector(ut.read_image(relpath("oxford2.jpg"), 1))

b = harris_corner_detector_liav(sut.read_image(relpath("oxford2.jpg"), 1))

print("hi")