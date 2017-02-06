import random
import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray

def read_image(filename, representation):
    # filename - file to open as image
    # representation - is it a B&W or color image
    im = imread(filename)
    # check if it is a B&W image
    if(representation == 1):
        im = rgb2gray(im)
    # convert to float and normalize
    return (im / 255).astype(np.float32)

def load_dataset(filenames, batch_size, corruption_func, crop_size):

    im_dict = {}
    currupt_im_dict = {}

    while True:
        im, currupt_im  = None, None
        rand_file = random.choice(filenames)

        if rand_file in im_dict:
            im = im_dict[rand_file]
            currupt_im = currupt_im_dict[rand_file]
        else:
            im = read_image(rand_file, 1)
            currupt_im = corruption_func(im) - 0.5
            im_dict[rand_file] = im - 0.5
            currupt_im_dict[rand_file] = currupt_im

        rand = np.random.rand(2, batch_size)
        rand_row = np.multiply(rand[0] , (im.shape[0] - crop_size[0])).astype(np.int_)
        rand_col = np.multiply(rand[1] , (im.shape[1] - crop_size[1])).astype(np.int_)

        yield
def corruption_func(im):
    return

# data_generator = load_dataset(filenames, batch_size, corruption_func, crop_size)