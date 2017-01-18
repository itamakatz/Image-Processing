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

def spread_out_corners(im, m, n, radius):
    #############################################################
    # takes from the additional files
    #############################################################
    corners = [np.empty((0,2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([y_bound[j], x_bound[i]])[np.newaxis,:]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:,0]>radius) & (corners[:,1]<im.shape[1]-radius) &
             (corners[:,1]>radius) & (corners[:,0]<im.shape[0]-radius))
    return corners[legit,:]


def harris_corner_detector(im):
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
    return np.transpose(np.nonzero(sut.non_maximum_suppression(R)))


def sample_descriptor(im, pos, desc_rad):
    #############################################################
    # descriptor sampling
    #############################################################
    K = 1 + (2 * desc_rad)
    desc = np.zeros((K, K, pos.shape[0]), dtype=np.float32)
    for idx in range(len(pos)):
        x, y = pos[idx][0].astype(np.float32) / 4, pos[idx][1].astype(np.float32) / 4
        # map the coordinates
        X = np.arange(y - desc_rad, y + desc_rad + 1)
        Y = np.arange(x - desc_rad, x + desc_rad + 1)
        indices = np.transpose([np.tile(Y, len(X)), np.repeat(X, len(Y))])
        curr_desc = map_coordinates(im, [indices[:, 0],indices[:, 1]],order=1, prefilter=False).reshape(K, K)
        # normalize the descriptor
        E = np.mean(curr_desc)
        curr_desc = (curr_desc - E) / np.linalg.norm(curr_desc - E)
        desc[:, :, idx] = curr_desc
    return desc


def im_to_points(im):
    #############################################################
    # implements the function in example_panoramas.py
    #############################################################
    pyr, vec = sut.build_gaussian_pyramid(im, 3, 3)
    return find_features(pyr)


def find_features(pyr):
    #############################################################
    # finds features in an image given by its pyramid pyr
    #############################################################
    pos = spread_out_corners(pyr[0], 7, 7, 12)
    desc = sample_descriptor(pyr[2], pos, 3)
    return pos, desc



a = harris_corner_detector(ut.read_image(relpath("oxford2.jpg"), 1))

b = harris_corner_detector_liav(sut.read_image(relpath("oxford2.jpg"), 1))

print("hi")