import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def reaf_image(filename, representation):
    im = imread(filename)
    im_float = im.astype(np.float32)
    im_float /= 255

    if(representation == 1):
        im_g = rgb2gray(im)
        return im_g

    return im

def imdisplay(filename, representation):
    if(representation == 1):
        plt.imshow(reaf_image(filename, representation), plt.cm.gray)
    else:
        plt.imshow(reaf_image(filename, representation))
    plt.show()

imdisplay("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 1)
