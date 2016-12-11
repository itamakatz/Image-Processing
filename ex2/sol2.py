import numpy as np
from scipy.misc import imread as imread
# from scipy.signal import convolve2d
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
    return np.around(vander.dot(x), NUMERIC_ERROR)

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
    return (General_DFT(fourier_signal, 1)) / fourier_signal.shape[0]


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
    ker_f = DFT2(np.copy(ker))
    im_f = DFT2(im)
    return IDFT2(im_f * ker_f)

im = 0
kernel_size = 3

blur_im = blur_spatial (im, kernel_size)
blur_im2 = blur_fourier (im, kernel_size)

magnitude = fourier_der(im)
magnitude2 = conv_der(im)

A = np.arange(6)
A = A[np.newaxis,:]
z = np.zeros((6,6)) + A

trans_signal = IDFT(A)

# print(trans_signal.shape)
# print(trans_signal)

new_trans_signal = IDFT(z[np.arange(6),:])

# print(new_trans_signal.shape)
# print(new_trans_signal)

print(trans_signal)
print(new_trans_signal[1,:])
print(new_trans_signal[:,1])