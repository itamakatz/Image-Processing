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

image = reaf_image("/home/itamar/Documents/image_processing/ex1/Files/test files/external/jerusalem.jpg", 1)
plt.imshow(image, plt.cm.gray)
plt.show()
