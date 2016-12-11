import numpy as np
from scipy.misc import imread as imread
from scipy.signal import convolve2d
import scipy.special
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

NUMERIC_ERROR = 13

def General_DFT(x, mult):
    # x - array to transform
    # mult - 1 or -1 to define if DFT or IDFT
    # return transform rounded of numerical error

    # compute vander matrix
    vander = np.vander((np.exp(mult * 2 * np.pi * 1j * np.arange(x.shape[0]) / x.shape[0])), increasing=True)
    # return vander.dot(x)
    return vander.dot(x)

def General_DFT2(x, mult):
    # x - array to transform
    # mult - 1 or -1 to define if DFT or IDFT
    # return transform rounded of numerical error

    # compute vander matrix of both dims to perform v_M*x*v_N
    vander_M = np.vander((np.exp(mult * 2 * np.pi * 1j * np.arange(x.shape[0]) / x.shape[0])), increasing=True)
    vander_N = np.vander((np.exp(mult * 2 * np.pi * 1j * np.arange(x.shape[1]) / x.shape[1])), increasing=True)
    return np.around(vander_M.dot(x.dot(vander_N)), NUMERIC_ERROR)

def DFT(signal):
    return General_DFT(signal, -1)

def IDFT(fourier_signal):
    return np.around(General_DFT(fourier_signal, 1), NUMERIC_ERROR) / fourier_signal.shape[0]


def DFT2(image):
    return General_DFT2(image, -1)

def IDFT2(image):
    return General_DFT2(image, 1) / (image.shape[0] * image.shape[1])

def conv_der(im):
    im = im / 255
    xDer = convolve2d(im, np.array([[1, 0, -1]]), mode='same')
    yDer = convolve2d(im, np.array([[1],[0],[-1]]), mode='same')

    return np.sqrt(np.abs(xDer)**2 + np.abs(yDer)**2)

def fourier_der(im):

    vander_M = np.vander((np.exp(-1 * 2 * np.pi * 1j * np.arange(im.shape[0]) / im.shape[0])), increasing=True)
    xDer =  np.around(vander_M.dot(np.arange(im.shape[0]) * im), NUMERIC_ERROR)

    vander_N = np.vander((np.exp(-1 * 2 * np.pi * 1j * np.arange(im.shape[1]) / im.shape[1])), increasing=True)
    yDer =  np.around((np.arange(im.shape[1]) * im).dot(vander_N), NUMERIC_ERROR)

    return np.sqrt(np.abs(xDer) ** 2 + np.abs(yDer) ** 2)

def create_ker(kernel_size):
    bin = scipy.special.binom(kernel_size - 1, np.arange(kernel_size)).astype(np.int64)
    kernel = convolve2d(bin[np.newaxis, :], bin[:, np.newaxis])
    return kernel / np.sum(kernel)

def blur_spatial (im, kernel_size):
    return convolve2d(im, create_ker(kernel_size))

def blur_fourier (im, kernel_size):
    ker = create_ker(kernel_size)
    ker_f = im * 0
    center = (np.floor(im.shape[0] / 2), np.floor(im.shape[1] / 2))
    ker_f[np.arange(center[0] - kernel_size,  center[0] + kernel_size), \
          np.arange(center[1] - kernel_size,  center[1] + kernel_size)] = ker
    ker_f = np.fft.ifftshift(ker_f)
    ker_f = DFT2(np.copy(ker))
    im_f = DFT2(im)
    return IDFT2(im_f * ker_f)

def read_image(filename, representation):

    im = imread(filename)
    # check if it is a B&W image
    if(representation == 1):
        im = rgb2gray(im)
    # convert to float and normalize
    return im.astype(np.float32)

def imdisplay(filename, representation):
    # check if it is a B&W or color image
    if(representation == 1):
        plt.imshow(read_image(filename, representation), plt.cm.gray)
    else:
        plt.imshow(read_image(filename, representation))
    plt.show()







# ========== Check DFT2 / IDFT2 ================
# im = read_image("/home/itamar/Documents/ip/ex2/files/presummition test/external/monkey.jpg", 1)
#
# x = DFT2(im)
# y = np.fft.fft2(im)
#
# plt.figure(1)
#
# plt.subplot(211)
# plt.imshow(abs(IDFT2(x)), plt.cm.gray)
# plt.subplot(212)
# plt.imshow(abs(np.fft.ifft2(y)), plt.cm.gray)
# # plt.subplot(211)
# # plt.imshow(np.log(1 + np.abs(np.fft.ifftshift(x))), plt.cm.gray)
# # plt.subplot(212)
# # plt.imshow(np.log(1 + np.abs(np.fft.ifftshift(y))), plt.cm.gray)
#
# plt.show()


# ========== Check DFT / IDFT ================
# n = 10000
# sig = np.random.rand(10,1)
# x = IDFT(sig)
# y = np.fft.ifft(np.matrix.transpose(sig))
# dif = x[:,0] - y[0,:]
# print(np.sum(np.around(dif, NUMERIC_ERROR)))