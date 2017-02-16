from skimage.color import rgb2gray
from scipy.misc import imread
from keras import layers as klay, models as kmod
from keras.optimizers import Adam
import sol5_utils as sut
from scipy.ndimage import filters as flt
import numpy as np

gau_bool = False
motion_bool = False

gau = None
motion = None

def read_image(filename, representation):
    """
    reads an image with the given representation
    """
    image = imread(filename).astype(np.float32) / 255
    return image if representation == 2 else rgb2gray(image)

def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    adds gaussian noise to a given image
    """
    return np.clip(image + np.random.normal(scale=np.random.uniform(min_sigma, max_sigma),
                                            size=image.shape), 0, 1)

def add_motion_blur(image, kernel_size, angle):
    """
    adds motion blur to a given image
    """
    return flt.convolve(image, sut.motion_blur_kernel(kernel_size, angle))


def add_gaussian_noise2(image, min_sigma, max_sigma):
    """
    adds gaussian noise to a given image
    """
    global gau
    if not gau_bool:
        gau = np.clip(image + np.random.normal(scale=np.random.uniform(min_sigma, max_sigma),
                                            size=image.shape), 0, 1)
    return gau

def add_motion_blur2(image, kernel_size, angle):
    """
    adds motion blur to a given image
    """
    global motion
    if not motion_bool:
        motion = flt.convolve(image, sut.motion_blur_kernel(kernel_size, angle))
    return motion


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Generator function for creating dataset of patches
    """
    # images = {}
    # while True:
    #     source, target = np.zeros((batch_size, 1, crop_size[0], crop_size[1])), \
    #                      np.zeros((batch_size, 1, crop_size[0], crop_size[1]))
    #     rand_files = np.random.choice(filenames, batch_size)
    #     for idx,_file_ in enumerate(rand_files):
    #         if _file_ not in images:
    #             images[_file_] = np.array(read_image(_file_, 1))
    #         image = images[_file_]
    #         corrupted_image = corruption_func(image)
    #         rand_X = np.random.randint(image.shape[0]-crop_size[0])
    #         rand_Y = np.random.randint(image.shape[1]-crop_size[1])
    #         source[idx,0,:,:] = corrupted_image[rand_X:rand_X + crop_size[0], rand_Y:rand_Y + crop_size[1]]
    #         target[idx,0,:,:] = image[rand_X:rand_X+crop_size[0], rand_Y:rand_Y+crop_size[1]]
    #     yield (source-0.5, target-0.5)

    im_dict = {}
    currupt_im_dict = {}
    while True:
        source, target = np.zeros((batch_size, 1, crop_size[0], crop_size[1]), np.float32), \
                         np.zeros((batch_size, 1, crop_size[0], crop_size[1]), np.float32)
        rand_files = np.random.choice(filenames, batch_size)
        for idx,_file_ in enumerate(rand_files):
            if _file_ in im_dict:
                im = im_dict[_file_]
                currupt_im = currupt_im_dict[_file_]
            else:
                im = read_image(_file_, 1)
                currupt_im = corruption_func(im)
                im_dict[_file_] = im
                currupt_im_dict[_file_] = currupt_im
            rand_X = np.random.randint(im.shape[0]-crop_size[0])
            rand_Y = np.random.randint(im.shape[1]-crop_size[1])
            source[idx,0,:,:] = currupt_im[rand_X:rand_X + crop_size[0], rand_Y:rand_Y + crop_size[1]]
            target[idx,0,:,:] = im[rand_X:rand_X+crop_size[0], rand_Y:rand_Y+crop_size[1]]
        yield (source - 0.5, target - 0.5)
