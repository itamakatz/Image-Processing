from scipy.ndimage import map_coordinates
import sol4_utils as ut
import sol4_add as ad
import matplotlib.pyplot as plt
import numpy as np
import functools
from scipy.signal import convolve2d

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

        XY = [X.reshape(K ** 2, ), Y.reshape(K ** 2, )]
        iter_mat = map_coordinates(im, np.array(XY), order=1, prefilter=False).reshape(K, K)
        mu = np.mean(iter_mat)
        desc[:, :, idx] = (iter_mat - mu) / np.linalg.norm(iter_mat - mu)
        if np.any(np.isnan(desc[:, :, idx])):
            continue
    return desc

def find_features(pyr):
    pos = ad.spread_out_corners(pyr[0], 7, 7, 12)
    return pos, sample_descriptor(pyr[2], pos, 3)


def match_features(desc1, desc2, min_score):

    flat1 = np.reshape(desc1, (-1, desc1.shape[2])).transpose().astype(dtype=np.float32)
    flat2 = np.reshape(desc2, (-1, desc2.shape[2])).transpose().astype(dtype=np.float32)

    score_desc1 = np.zeros([flat1.shape[0], 2, 2])
    score_desc2 = np.zeros([flat2.shape[0], 2, 2])

    to_iter = np.dstack(np.meshgrid(np.arange(flat1.shape[0]), np.arange(flat2.shape[0]))).reshape(-1, 2)

    for (i1, i2) in to_iter:

        product = np.inner(flat1[i1], flat2[i2])
        if product <= min_score:
            continue

        if product >= score_desc1[i1, 1, 0]:
            if product >= score_desc1[i1, 0, 0]:
                score_desc1[i1, 1, :] = score_desc1[i1, 0, :]
                score_desc1[i1, 0, :] = product, i2
            else:
                score_desc1[i1, 1, :] = product, i2

        if product >= score_desc2[i2, 1, 0]:
            if product >= score_desc2[i2, 0, 0]:
                score_desc2[i2, 1, :] = score_desc2[i2, 0, :]
                score_desc2[i2, 0, :] = product, i1
            else:
                score_desc2[i2, 1, :] = product, i1

    ret_desc1 = []
    ret_desc2 = []

    for (i1, i2) in to_iter:
        if ((score_desc1[i1, 0, 1] == i2 or score_desc1[i1, 1, 1] == i2)) and \
                ((score_desc2[i2, 0, 1] == i1 or score_desc2[i2, 1, 1] == i1)):
            for i, j in np.dstack(np.meshgrid(np.arange(2), np.arange(2))).reshape(-1, 2):
                if score_desc1[i1, i, 0] == score_desc2[i2, j, 0]:
                    ret_desc1.append(i1)
                    ret_desc2.append(i2)
                    break

    return np.array(ret_desc1), np.array(ret_desc2)


def apply_homography(pos1, H12):
    product = np.dot(H12[:, :-1], np.transpose(pos1)) + np.transpose(np.expand_dims(H12[:, -1], axis=0))
    return np.transpose(product[:2] / product[2])


def ransac_homography(pos1, pos2, num_iters, inlier_tol):

    apply_hom_const = functools.partial(apply_homography, pos1)
    rand_array = np.random.random_integers(0, pos1.shape[0] - 1, size=(num_iters, 4))

    def all_least_squares(cuad_rand):
        H12 = ad.least_squares_homography(pos1[cuad_rand], pos2[cuad_rand])
        if H12 is None:
            return 0
        return np.sum(np.ones(pos2.shape[0])[np.sum(np.square(apply_hom_const(H12) - pos2), axis=1) < inlier_tol])

    rand_results= np.apply_along_axis(all_least_squares, 1, rand_array)
    H12 = ad.least_squares_homography(pos1[rand_array[rand_results.argmax()]],
                                      pos2[rand_array[rand_results.argmax()]])
    true_samples= np.arange(pos1.shape[0])[np.sum(np.square(apply_hom_const(H12) - pos2), axis=1) < inlier_tol]
    H12 = ad.least_squares_homography(pos1[true_samples], pos2[true_samples])

    return H12, true_samples


def display_matches(im1, im2, pos1, pos2, inliers):
    pos1, pos2 = np.array(pos1), np.array(pos2)
    plt.figure()
    plt.imshow(np.hstack((im1, im2)), 'gray')
    plt.plot([pos1[inliers][:, 1], pos2[inliers][:, 1] + im1.shape[1]],
             [pos1[inliers][:, 0], pos2[inliers][:, 0]], mfc='r', c='y', lw=0.6, ms=5, marker='o')
    plt.plot([np.delete(pos1, inliers, axis=0)[:, 1], np.delete(pos2, inliers, axis=0)[:, 1] + im1.shape[1]],
             [np.delete(pos1, inliers, axis=0)[:, 0], np.delete(pos2, inliers, axis=0)[:, 0]], mfc='r', c='b', lw=0.4, ms=5, marker='o')
    plt.show()


def accumulate_homographies(H_successive, m):

    def accumulate_dot(mat):
        total = np.eye(3)
        accum = []
        for item in mat:
            total = np.dot(total, item)
            accum.append(total)
        return accum

    if m == 0:
        return [np.eye(3), np.linalg.inv(H_successive[0])]


    ret_H = np.array(accumulate_dot(H_successive[:m][::-1])[::-1] + [np.eye(3)] +
                     accumulate_dot(list(map(np.linalg.inv, H_successive[m:]))))

    for i in range(len(ret_H)):
        ret_H[i] /= ret_H[i,2,2]

    return ret_H


def render_panorama(ims, Hs):
    corners_N_centers = np.zeros((5, len(ims)))
    for i in range(len(ims)):
        rows, cols = ims[i].shape[0], ims[i].shape[1]
        shifted_corners = np.array(apply_homography([[0, 0], [rows - 1, 0], [0, cols - 1],
                                                             [rows - 1, cols - 1]], Hs[i]))
        corners_N_centers[0, i] = np.max(shifted_corners[:, 0])
        corners_N_centers[1, i] = np.min(shifted_corners[:, 0])
        corners_N_centers[2, i] = np.max(shifted_corners[:, 1])
        corners_N_centers[3, i] = np.min(shifted_corners[:, 1])
        corners_N_centers[4, i] = np.array(apply_homography([[(float(rows) - 1) / 2,
                                                              (float(cols) - 1) / 2]], Hs[i]))[0][1]

    Yaxis, Xaxis = np.meshgrid(np.arange(np.min(corners_N_centers[3,:]),
                                         np.max(corners_N_centers[2,:])+1), np.arange(np.min(corners_N_centers[1,:]),
                                                                                      np.max(corners_N_centers[0,:])+1))
    borders = []
    for i in range(len(ims) - 1):
        borders.append(np.round((corners_N_centers[4,i]+corners_N_centers[4,i+1])/2)-np.min(corners_N_centers[3,:]))

    borders = [0] + borders + [Yaxis.shape[1]]
    panorama = np.zeros(Yaxis.shape)
    for i in range(len(ims)):
        left, right = int(borders[i]), int(borders[i+1])
        pos = np.transpose([Xaxis[:,left:right].flatten(), Yaxis[:,left:right].flatten()])
        index_matrix = np.array(apply_homography(pos, np.linalg.inv(Hs[i])))
        panorama[:, left:right] = map_coordinates(ims[i], [index_matrix[:,0],
                                                            index_matrix[:,1]], order=1, prefilter=False)\
                                                            .reshape(panorama[:, left:right].shape)


    return panorama.astype(np.float32)
