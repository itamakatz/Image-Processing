from scipy.ndimage import map_coordinates
import sol4_utils as sut
import matplotlib.pyplot as plt
import scipy.signal as sg
import numpy as np
import itertools
import heapq


# --------------------------3.1-----------------------------#

def spread_out_corners(im, m, n, radius):
    return


def harris_corner_detector(im):
    return


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
