import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def read_image(filename, representation):
    im = imread(filename)

    if(representation == 1):
        im = rgb2gray(im)

    im_float = im.astype(np.float32)
    im_float /= 255
    return im_float

def imdisplay(filename, representation):
    if(representation == 1):
        plt.imshow(read_image(filename, representation), plt.cm.gray)
    else:
        plt.imshow(read_image(filename, representation))
    plt.show()

def rgb2yiq(imRGB):

    trans_mat = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    imYIQ = imRGB.copy()
    for i in range(0, 3):
        imYIQ[:, :, i] = trans_mat[i, 0] * imRGB[:, :, 0] + trans_mat[i, 1] * imRGB[:, :, 1] + \
                         trans_mat[i, 2] * imRGB[:, :,2]

    return imYIQ

def yiq2rgb(imYIQ):
    trans_mat = np.linalg.inv(np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]))
    imRGB = imYIQ.copy()
    for i in range(0, 3):
        imRGB[:, :, i] = trans_mat[i, 0] * imYIQ[:, :, 0] + trans_mat[i, 1] * imYIQ[:, :, 1] + \
                         trans_mat[i, 2] * imYIQ[:, :,2]

    return imRGB

def histogram_equalize(im_orig):
    if(im_orig.shape() == 1):
        return

    hist_orig = np.histogram(rgb2yiq(im_orig), 256)
    cumulative_histogram = np.cumsum(hist_orig)

    # find first m for which S(m) != 0
    m_index = (cumulative_histogram[cumulative_histogram > 0])[0]
    # cumulative_histogram = cumulative_histogram/np.max(cumulative_histogram)*255

    linear_streach = 255 * (cumulative_histogram[:] - cumulative_histogram[m_index]) / \
                     (cumulative_histogram[255] - cumulative_histogram[m_index])
    hist_eq = linear_streach.round().astype(np.uint8)

imRGB = yiq2rgb(rgb2yiq(read_image("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 2)))
plt.imshow(imRGB)
plt.show()

imdisplay("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 2)

# [im_eq, hist_orig, hist_eq] = histogram_equalize(im_orig)
