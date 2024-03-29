#############################################################
# FILE: sol5.py
# WRITER: Liav Steinberg
# EXERCISE : Image Processing ex5
#############################################################


from same_funcs import load_dataset, add_gaussian_noise2 as add_gaussian_noise, add_motion_blur2 as add_motion_blur


from skimage.color import rgb2gray
from scipy.misc import imread
# from keras import layers as klay, models as kmod
from keras.optimizers import Adam
from scipy.ndimage import filters
import sol5_utils as ut
import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, merge

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.misc import imread
from keras import layers as klay, models as kmod
from keras.optimizers import Adam
from scipy.ndimage import filters as flt
import sol5_utils as sut
import numpy as np


def read_image(filename, representation):
    """
    reads an image with the given representation
    """
    image = imread(filename).astype(np.float32) / 255
    return image if representation == 2 else rgb2gray(image)


def learn_X_model(quick_mode, load_func, corr_func, cs, nc, ne):
    """
    to reduce code repetition for section 7.2.1, 7.2.2
    """
    def corruption_func(im):
        """
        wrapper for the corruption function
        """
        return add_gaussian_noise(im, 0, 0.2) if corr_func == "gaussian_noise" \
            else random_motion_blur(im, [7])
    #######################################################
    images = load_func()
    batch_size = 100
    sam_per_epoch = 10000
    num_epochs = ne
    num_valid_sample = 1000
    if quick_mode:
        batch_size = 10
        sam_per_epoch = 30
        num_epochs = 2
        num_valid_sample = 30
    model = build_nn_model(cs, cs, nc)
    train_model(model, images, corruption_func, batch_size,
                sam_per_epoch, num_epochs, num_valid_sample)
    return model, nc

#--------------------------3-----------------------------#

#--------------------------4-----------------------------#

def resblock(input_tensor, num_channels):
    """
    represent a single block in the network flow
    """
    conv = klay.Convolution2D(num_channels, 3, 3, border_mode="same")(input_tensor)
    relu = klay.Activation("relu")(conv)
    second_conv = klay.Convolution2D(num_channels, 3, 3, border_mode="same")(relu)
    return klay.merge([input_tensor, second_conv], mode="sum")


def build_nn_model(height, width, num_channels):
    """
    builds an untrained Keras model
    """
    def repeat(func, x, y, n):
        """
        high-order function for repeating func on inputs
        x, y as times as n
        """
        for i in range(n):
            x = func(x, y)
        return x
    #######################################################
    input_ = klay.Input(shape=(1, height, width))
    conv = klay.Convolution2D(num_channels, 3, 3, border_mode="same")(input_)
    relu = klay.Activation("relu")(conv)
    after_resblocks = repeat(resblock, relu, num_channels, 5)
    final = klay.merge([relu, after_resblocks], mode="sum")
    output = klay.Convolution2D(1, 3, 3, border_mode="same")(final)
    return kmod.Model(input_, output)

#--------------------------5-----------------------------#

def train_model(model, images, corruption_func, batch_size,
                samples_per_epoch, num_epochs, num_valid_sample):
    """
    trains the given model according to a dataset
    """
    training, validation = np.split(images, [int(len(images)*0.8)])
    training_set = load_dataset(training, batch_size, corruption_func, model.input_shape[2:])
    validation_set = load_dataset(validation, batch_size, corruption_func, model.input_shape[2:])
    model.compile(loss="mean_squared_error", optimizer=Adam(beta_2=0.9))
    model.fit_generator(training_set, samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=validation_set, nb_val_samples=num_valid_sample)

#--------------------------6-----------------------------#

def restore_image(corrupted_image, base_model, num_channels):
    """
    restores the corrupted image according to the learning model
    """
    height, width = corrupted_image.shape[0], corrupted_image.shape[1]
    corrupted_image = np.array(corrupted_image).reshape(1, height, width)-0.5
    model = build_nn_model(height, width, num_channels)
    model.set_weights(base_model.get_weights())
    prediction = model.predict(corrupted_image[np.newaxis,...])[0] + 0.5
    return np.clip(prediction,0,1).reshape(height, width)

#--------------------------7.1.1-----------------------------#

#--------------------------7.1.2-----------------------------#

def learn_denoising_model(quick_mode=False):
    """
    returns trained model for the denoising set
    """
    return learn_X_model(quick_mode, sut.images_for_denoising, "gaussian_noise", 24, 48, 5)

#--------------------------7.2.1-----------------------------#


def random_motion_blur(image, list_of_kernel_sizes):
    """
    adds random motion blur to a given image
    """
    return add_motion_blur(image, np.random.choice(list_of_kernel_sizes), np.random.uniform(high=np.pi))

#--------------------------7.2.2-----------------------------#

def learn_deblurring_model(quick_mode=False):
    """
    returns trained model for the deblurring set
    """
    return learn_X_model(quick_mode, sut.images_for_deblurring, "random_motion_blur", 16, 32, 10)

#--------------------------end-----------------------------#
