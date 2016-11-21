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
    # if it is B&W
    if(im_orig.dims == 1):
        im = im_orig
    else:
        im = rgb2yiq(im_orig)[:,:,0]

    hist_orig = np.histogram(im, 256)
    cumulative_histogram = np.histogram(rgb2yiq(im_orig)[:, :, 0], 256, None, False, None, True)
    # cumulative_histogram = np.cumsum(hist_orig)
    # cumulative_histogram = cumulative_histogram / cumulative_histogram[255]

    # find first m for which S(m) != 0
    m_value = (cumulative_histogram[cumulative_histogram > 0])[0]

    linear_streach = (cumulative_histogram - m_value) / (cumulative_histogram[255] - m_value) * 255

    calc_hist_vect = np.zeros((256,))
    calc_hist_vect[1:] = linear_streach[:-1]

    hist_eq = (linear_streach - calc_hist_vect).round().astype(np.uint8)

    im_eq = np.interp(im.flatten(), np.arange(255), hist_eq).reshape(im.shape)

    return [im_eq, hist_orig, hist_eq]


imRGB = yiq2rgb(rgb2yiq(read_image("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 2)))
plt.imshow(imRGB)
plt.show()

imdisplay("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 2)

# [im_eq, hist_orig, hist_eq] = histogram_equalize(im_orig)
