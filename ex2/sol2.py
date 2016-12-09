import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def General_DFT(x, mult):
    # x - array to transform
    # mult - 1 or -1 to define if DFT or IDFT
    # return transform

    N = signal.shape()
    vander = np.vander(np.exp(np.arange(N)), increasing=True)
    vect = np.exp(mult * 2 * np.pi * 1j * np.arange(N) / N)
    vect = vect * signal
    return vander.dot(vect)

def DFT(signal):
    return General_DFT(signal, -1)

def IDFT(fourier_signal):
    return General_DFT(signal, 1)

def DFT2(image):
    return

def IDFT2(image):
    return


signal = IDFT(fourier_signal)
fourier_signal = DFT(signal)