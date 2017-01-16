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
    return

# NOT REQUIRED
# def im_to_points(im):
#     return


def find_features(pyr):
    return


# --------------------------3.2-----------------------------#

def match_features(desc1, desc2, min_score):
    return

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
