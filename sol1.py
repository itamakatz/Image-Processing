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

    yiq = None

    # check if it is a B&W or color image
    if(im_orig.ndim == 2):
        im = im_orig
    else:
        # transform to the YIQ space
        yiq = rgb2yiq(im_orig)
        im = yiq[:, :, 0]

    # calc the histogram of the image
    hist_orig, bins = np.histogram(im.flatten(), 255)
    # compute the cumulative histogram
    cumulative_histogram = np.cumsum(hist_orig)
    # find first m for which S(m) != 0
    m_val = (cumulative_histogram[cumulative_histogram > 0])[0]
    # apply linear stretching
    cumulative_stretch = np.round(255 * (cumulative_histogram - m_val) / (cumulative_histogram[-1] - m_val))

    if(im_orig.ndim == 2):
        # apply the look up table to the image
        im_eq = np.interp(im.flatten(), bins[:-1], cumulative_stretch).reshape(im.shape)
        #  calc the histogram of the enhanced image
        hist_eq, bins2 = np.histogram(im_eq, 255)
    else:
        # apply the look up table to the image
        yiq[:, :, 0] = np.interp(im.flatten(), np.linspace(0, 1, 255, True), cumulative_stretch).reshape(im.shape) / 255
        #  ensure the values after the transformation are in the [0,1] range by "clipping"
        im_eq = np.clip(yiq2rgb(yiq), 0, 1)
        #  calc the histogram after the clipping
        # yiq = rgb2yiq(yiq2rgb(im_eq))
        hist_eq, bins2 = np.histogram(yiq[:, :, 0].flatten(), 255)

    return [im_eq, hist_orig, hist_eq]

def histogram_equalize2(im_orig):
    #######################################################
    # Performs histogram equalization on a given image,
    # according to the lecture algorithm.
    #######################################################
    isRGB, imYIQ = (len(im_orig.shape) == 3), None
    if isRGB:
        imYIQ = rgb2yiq(im_orig)
        im_orig = imYIQ[:, :, 0]
    #get the histogram of the image
    hist_orig, bins = np.histogram(im_orig.flatten(), 255)
    #compute the cumulative histogram
    cdf = np.cumsum(hist_orig)
    #normalize the cumulative histogram
    cdf = np.round(255 * (cdf - cdf[cdf>0].min()) / (cdf.max() - cdf[cdf>0].min()))
    #use linear interpolation of cdf to find new pixel values
    im_eq = np.interp(im_orig, np.linspace(0, 1, 255), cdf)
    #histogram the new image
    hist_eq, bins = np.histogram(im_eq.flatten(), 255)
    #if we got RGB, return it back to RGB
    if isRGB:
        imYIQ[:, :, 0] = im_eq / 255
        #using clip to zero the negative results of the transformation
        im_eq = np.clip(yiq2rgb(imYIQ), 0, 1)
    return im_eq, hist_orig, hist_eq

def quantize (im_orig, n_quant, n_iter):
    yiq = None

    # check if it is a B&W or color image
    if(im_orig.ndim == 2):
        im = im_orig
    else:
        # transform to the YIQ space
        yiq = rgb2yiq(im_orig)
        im = yiq[:, :, 0]

    if(im_orig.ndim == 2):
        return
    else:
        return

# [im_quant, error] = quantize (im_orig, n_quant, n_iter)

# # imdisplay("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 2)

# im_orig = read_image ("/home/itamar/Documents/image_processing/ex1/Files/test files/external/monkey.jpg", 2)
# im_orig = read_image ("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 2)
im_orig = read_image ("/home/itamar/Documents/image_processing/ex1/Files/test files/external/Low Contrast.jpg", 2)

[im_eq, hist_orig, hist_eq] = histogram_equalize(im_orig)
# [im_eq, hist_orig, hist_eq] = histogram_equalize2(im_orig)

plt.figure(1)
plt.subplot(221)
plt.title("Original Image")
# plt.imshow(im_orig)
plt.imshow(im_orig, plt.cm.gray)

plt.subplot(222)
plt.title("Equalized Image")
# plt.imshow(im_eq)
plt.imshow(im_eq, plt.cm.gray)

plt.subplot(223)
plt.title("Histograms: Original - Red, Equalized - Blue")
plt.plot(np.arange(len(hist_orig)) ,hist_orig, 'r', np.arange(len(hist_eq)), hist_eq, 'b')

plt.subplot(224)
plt.title("PDF Histograms: Original - Red, Equalized - Blue")
pdf_original = np.cumsum(hist_orig);
pdf_eq = np.cumsum(hist_eq);
pdf_eq = pdf_eq / pdf_eq.max() * 255
plt.plot(np.arange(len(pdf_original)) ,np.arange(len(pdf_original)), 'r', np.arange(len(pdf_eq)) ,pdf_eq, 'b')
plt.show()