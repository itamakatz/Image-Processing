from scipy.ndimage import map_coordinates
import sol4_utils as ut
import sol4_add as ad
import matplotlib.pyplot as plt
import scipy.signal as sg
import numpy as np
import itertools
import heapq
import functools
from scipy.signal import convolve2d


# --------------------------3.1-----------------------------#

def harris_corner_detector(im):
    der = np.array([[1,0,-1]])

    I_x = convolve2d(im, der, mode='same')
    I_y = convolve2d(im, np.transpose(der), mode='same')

    M = np.array([np.square(I_x), np.multiply(I_x, I_y), np.square(I_y)])
    M = np.array(list(map(functools.partial(ut.blur_spatial_rev, 3), M)))
    R = np.multiply(M[0], M[2]) - np.square(M[1]) - 0.04 * np.square((M[0] + M[2]))

    return np.transpose(np.nonzero(ad.non_maximum_suppression(R)))

def sample_descriptor(im, pos, desc_rad):
    K = 1 + (2 * desc_rad)
    desc = np.zeros((K, K, pos.shape[0]), dtype=np.float32)
    for idx in range(len(pos)):
        x, y = pos[idx][0].astype(np.float32) / 4, pos[idx][1].astype(np.float32) / 4
        X, Y = np.meshgrid(np.linspace(x - desc_rad, x + desc_rad, K, dtype=np.float32),
                             np.linspace(y - desc_rad, y + desc_rad, K, dtype=np.float32))
        XY = [X.reshape(K**2,), Y.reshape(K**2,)]
        iter_mat = map_coordinates(im, XY, order=1, prefilter=False).reshape(K, K)
        # normalize the descriptor
        mu = np.mean(iter_mat)
        desc[:, :, idx] = (iter_mat - mu) / np.linalg.norm(iter_mat - mu)
    return desc

# NOT REQUIRED
# def im_to_points(im):
#     return


def find_features(pyr):
    feature_loc = ad.spread_out_corners(pyr[0], 7, 7, 12)
    return sample_descriptor(pyr[2], feature_loc, 3), feature_loc


# --------------------------3.2-----------------------------#

max_

def match_features(desc1, desc2, min_score):

    flat1 = np.reshape(desc1, (-1, desc1.shape[2])).transpose().astype(dtype=np.float32)
    flat2 = np.reshape(desc2, (-1, desc2.shape[2])).transpose().astype(dtype=np.float32)

    score_desc1 = np.zeros([flat1.shape[1], 2, 2])
    score_desc2 = np.zeros([flat2.shape[1], 2, 2])

    for (i1, i2) in np.dstack(np.meshgrid(np.arange(flat1.shape[1]), np.arange(flat2.shape[1]))).reshape(-1, 2):

        product = np.inner(flat1[i1], flat2[i2])
        if product <= min_score:
            continue

        if product >= np.amin(score_desc1[i1, :, 0]):
            score_desc1[i1, score_desc1[i1, :, 0].argmin(), :] = product, i2

        if product >= np.amin(score_desc2[i2, :, 0]):
            score_desc2[i2, score_desc2[i2, :, 0].argmin(), :] = product, i1

    ret_desc1 = []
    ret_desc2 = []

    for (i1, i2) in np.dstack(np.meshgrid(np.arange(flat1.shape[1]), np.arange(flat2.shape[1]))).reshape(-1, 2):
        if (score_desc1[i1, 0, 1] == i2 or score_desc1[i1, 1, 1] == i2) and \
                (score_desc2[i2, 0, 1] == i1 or score_desc2[i2, 1, 1] == i1):
            ret_desc1.append(i1)
            ret_desc2.append(i2)

    return np.array(ret_desc1), np.array(ret_desc2)

# --------------------------3.3-----------------------------#

def apply_homography(pos1, H12):
    return

def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    return


def display_matches(im1, im2, pos1, pos2, inliers):
    return

# --------------------------3.3-----------------------------#

def accumulate_homographies(H_successive, m):
    return

# --------------------------4.3-----------------------------#

# NOT REQUIRED
# def prepare_panorama_base(ims, Hs):
#     return


def render_panorama(ims, Hs):
    return

# --------------------------end-----------------------------#


pyr, _ = ut.build_gaussian_pyramid(ut.read_image(ut.relpath("oxford2.jpg"), 1), 3, 3)
pos = ad.spread_out_corners(pyr[0], 7, 7, 12)
desc = sample_descriptor(pyr[2], pos, 3)

print("hi")