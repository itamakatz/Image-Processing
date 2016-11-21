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

    # #
    # temp_im, im = None, None
    #
    # # Checking weather it is a B&W image or comor
    # if(im_orig.ndim == 2):
    #     im = im_orig
    # else:
    #     temp_im = (rgb2yiq(im_orig))
    #     im = temp_im[:,:,0]
    #
    # # Computing histogram of original image
    # hist_orig, bins = np.histogram(im.flatten(), 256)
    # #  Compute thr cumulative histogram
    # cumulative_histogram = np.cumsum(hist_orig)
    #
    # # find first m for which S(m) != 0
    # m_value = (cumulative_histogram[cumulative_histogram > 0])[0]
    # # applay linear streachin
    # cumulative_streach = np.round((cumulative_histogram - m_value) * 255 / (cumulative_histogram[-1] - m_value))
    #
    # # computing the hist_eq using an empty vector
    # # calc_hist_vect = np.zeros((257,))
    # # calc_hist_vect[1:] = cumulative_streach[:]
    # # hist_eq = (cumulative_streach - calc_hist_vect[:-1])
    #
    # la = np.interp(im, bins[:-1], cumulative_streach)
    #
    # if(im_orig.ndim == 2):
    #     im_eq = la / 255
    # else:
    #     temp_im[:,:,0] = la / 255
    #     im_eq = np.clip(yiq2rgb(temp_im), 0 ,1)
    #
    # hist_eq, _ = np.histogram(la, 256)
    #
    # return [im_eq, hist_orig, hist_eq]

    temp_im, im = None, None

    # Checking weather it is a B&W image or comor
    if(im_orig.ndim == 2):
        im = im_orig
    else:
        temp_im = (rgb2yiq(im_orig))
        im = temp_im[:,:,0]

    # Computing histogram of original image
    hist_orig, bins = np.histogram(im.flatten(), 256)
    #  Compute thr cumulative histogram
    cumulative_histogram = np.cumsum(hist_orig)

    # find first m for which S(m) != 0
    m_value = (cumulative_histogram[cumulative_histogram > 0])[0]
    # applay linear streachin
    cumulative_streach = (cumulative_histogram - m_value) * 255 / (cumulative_histogram[-1] - m_value)

    # computing the hist_eq using an empty vector
    # calc_hist_vect = np.zeros((257,))
    # calc_hist_vect[1:] = cumulative_streach[:]
    # hist_eq = (cumulative_streach - calc_hist_vect[:-1])

    if(im_orig.ndim == 2):
        im_eq = np.interp(im.flatten(), bins[:-1], cumulative_streach).reshape(im.shape) / 255
    else:
        temp_im[:,:,0] = np.interp(im.flatten(), bins[:-1], cumulative_streach).reshape(im.shape) / 255
        im_eq = np.clip(yiq2rgb(temp_im), 0 ,1)

    hist_eq, _ = np.histogram(im_eq.flatten(), 256)

    return [im_eq, hist_orig, hist_eq]

# imRGB = yiq2rgb(rgb2yiq())

# # imdisplay("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 2)

[im_eq, hist_orig, hist_eq] = histogram_equalize(read_image \
                                ("/home/itamar/Documents/image_processing/ex1/Files/test files/external/monkey.jpg", 2))
# plt.imshow(im_eq)
# plt.imshow(im_eq, plt.cm.gray)
plt.plot(np.arange(256) ,hist_orig)
plt.show()
plt.plot(np.arange(256) ,hist_eq)
plt.show()