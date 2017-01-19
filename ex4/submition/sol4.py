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

def apply_single_homography(H12, x, y):
    coordin = np.array([x, y, 1])
    product = np.dot(H12, coordin)
    return product[:2] / product[2]

def apply_homography(pos1, H12):
    return np.array(list(map(functools.partial(apply_single_homography, H12), pos1[:,0], pos1[:,1])))
#
# def apply_homography_liav(pos1, H12):
#     expand = np.column_stack((pos1, np.ones(len(pos1))))
#     dot = np.dot(H12, expand.T).T
#     normalized = (dot.T / dot[:,2]).T
#     return np.delete(normalized, -1, axis=1)


def ransac_homography(pos1, pos2, num_iters, inlier_tol):

    def all_least_squares(cuad_rand):
        H12 = ad.least_squares_homography(pos1[cuad_rand[:2]], pos2[cuad_rand[-2:]])
        # avoid unstable results
        if H12 is None:
            return 0
        return np.square(np.sum(np.dot(apply_homography(pos1, H12), pos2)))

    rand_array = np.random.random_integers(0, pos1.shape[0], size=(num_iters, 4))
    coordinates = np.apply_along_axis(all_least_squares, 0, rand_array)

    best_coordinates = np.arange(coordinates.shape[0])[coordinates > inlier_tol ** 2]
    H12 = ad.least_squares_homography(pos1[best_coordinates], pos2[best_coordinates])

    return H12, np.array(best_coordinates)


def display_matches(im1, im2, pos1, pos2, inliers):
    pos1, pos2 = np.array(pos1), np.array(pos2)
    ins1, ins2 = pos1[inliers], pos2[inliers]
    out1, out2 = np.delete(pos1, inliers, axis=0), np.delete(pos2, inliers, axis=0)
    plt.figure()
    plt.imshow(np.hstack((im1, im2)), 'gray')
    plt.plot([ins1[:, 1], ins2[:, 1] + im1.shape[1]], [ins1[:, 0], ins2[:, 0]], mfc='r', c='y', lw=1.1, ms=5, marker='o')
    plt.plot([out1[:, 1], out2[:, 1] + im1.shape[1]], [out1[:, 0], out2[:, 0]], mfc='r', c='b', lw=0.4, ms=5, marker='o')
    plt.show()

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

im1 = ut.read_image(ut.relpath("external/oxford1.jpg"), 1)
im2 = ut.read_image(ut.relpath("external/oxford2.jpg"), 1)

pyr1, _ = ut.build_gaussian_pyramid(im1, max_levels=3, filter_size=3)
pyr2, _ = ut.build_gaussian_pyramid(im2, max_levels=3, filter_size=3)

pos1, desc1 = find_features(pyr1)
pos2, desc2= find_features(pyr2)

match1, match2 = match_features(desc1, desc2, min_score=10)

H12, inliers_ = ransac_homography(match1, match2, num_iters = 100, inlier_tol=10)

display_matches(im1, im2, pos1, pos2, inliers_)

print("hi")