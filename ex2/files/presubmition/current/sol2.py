import numpy as np
import scipy.special
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy.signal import convolve2d

# numerical precision tu truncate using the round function
NUMERIC_ERROR = 13

def read_image(filename, representation):
    # filename - file to open as image
    # representation - is it a B&W or color image

    im = imread(filename)
    # check if it is a B&W image
    if(representation == 1):
        im = rgb2gray(im)
    # convert to float and normalize
    return im.astype(np.float32) / 255

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
    # round of numerical error and normalize
    return np.around(General_DFT(fourier_signal, 1), NUMERIC_ERROR) / fourier_signal.shape[0]


def DFT2(image):
    return General_DFT2(image, -1)

def IDFT2(image):
     # normalize by both dimensions
    return General_DFT2(image, 1) / (image.shape[0] * image.shape[1])

def conv_der(im):
    # normalize and compute conv with padding
    im = im / 255
    xDer = convolve2d(im, np.array([[1, 0, -1]]), mode='same')
    yDer = convolve2d(im, np.array([[1],[0],[-1]]), mode='same')

    # compute power
    return np.sqrt(np.abs(xDer)**2 + np.abs(yDer)**2)

def fourier_der(im):

    # compute the frequency derivatives for multiplication
    u, v = np.meshgrid(np.arange(-im.shape[1] / 2, im.shape[1] / 2 - 1 * im.shape[1] % 2),
                       np.arange(-im.shape[0] / 2, im.shape[0] / 2 - 1 * im.shape[0] % 2))

    # no need to normalize since using func General_DFT2
    xDer = 2 * np.pi * 1j * General_DFT2(u * np.fft.fftshift(General_DFT2(im, -1)), 1)
    yDer = 2 * np.pi * 1j * General_DFT2(v * np.fft.fftshift(General_DFT2(im, -1)), 1)

    # compute power
    return np.sqrt(np.abs(xDer) ** 2 + np.abs(yDer)**2)

def create_ker(kernel_size):
    # kernel_size - odd integer
    # returns a binomial kernel of size kernel_size X kernel_size approximating a gausian
    bin = scipy.special.binom(kernel_size - 1, np.arange(kernel_size)).astype(np.int64)
    kernel = convolve2d(bin[np.newaxis, :], bin[:, np.newaxis])
    return kernel / np.sum(kernel)

def blur_spatial (im, kernel_size):
    return convolve2d(im, create_ker(kernel_size), mode='same')

def blur_fourier (im, kernel_size):
    ker = create_ker(kernel_size)

    center = (int(np.floor(im.shape[0] / 2)), int(np.floor(im.shape[1] / 2)))

    # def ker_f to be the size of im and add it the ker
    ker_f = im * 0
    ker_f[np.meshgrid(
        np.arange(center[0] - int((kernel_size - 1) / 2), center[0] + int((kernel_size - 1) / 2) + 1),
        np.arange(center[1] - int((kernel_size - 1) / 2), center[1] + int((kernel_size - 1) / 2) + 1))] = ker

    ker_f = np.fft.ifftshift(ker_f)
    ker_f = DFT2(np.copy(ker_f))
    im_f = DFT2(im)
    # return of type float32
    return (abs(IDFT2(im_f * ker_f))).astype(np.float32)