from same_funcs import load_dataset, add_gaussian_noise2 as add_gaussian_noise, add_motion_blur2 as add_motion_blur

import numpy as np
import sol5_utils as ut
from scipy.misc import imread
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from scipy.ndimage import filters
from skimage.color import rgb2gray
from keras.layers import Input, Convolution2D, Activation, merge

# def read_image(filename, representation):
#     # filename - file to open as image
#     # representation - is it a B&W or color image
#     im = imread(filename) / 255
#     # check if it is a B&W image
#     if(representation == 1):
#         im = rgb2gray(im)
#     # convert to float and normalize
#     return im.astype(np.float32)

def read_image(filename, representation):
    """
    reads an image with the given representation
    """
    image = imread(filename).astype(np.float32) / 255
    return image if representation == 2 else rgb2gray(image)

conv_func = lambda channels, tensor: Convolution2D(channels, 3, 3, border_mode = "same")(tensor)

def resblock(input_tensor, num_channels):
    return merge([input_tensor,
                  conv_func(num_channels, Activation("relu")(conv_func(num_channels, input_tensor)))], mode = "sum")

def build_nn_model(height, width, num_channels):

    input = Input(shape = (1, height, width))
    original_relu = Activation("relu")(conv_func(num_channels, input))

    relu = original_relu
    for i in range(5):
        relu = resblock(relu, num_channels)

    output = conv_func(1, merge([original_relu, relu], mode = "sum"))
    return Model(input, output)
#
# def build_nn_model(height, width, num_channels):
#
#     def applay_resblock(ReLu, iter_times):
#         if iter_times: return ReLu
#         return applay_resblock(resblock(ReLu, num_channels), iter_times - 1)
#
#     input = Input(shape = (1, height, width))
#     relu = Activation("relu")(conv_func(num_channels, input))
#     output = conv_func(1, merge([relu, applay_resblock(relu, 5)], mode = "sum"))
#     return Model(input, output)

def train_model(model, images, corruption_func, batch_size,
            samples_per_epoch, num_epochs, num_valid_sample):

    training_set = images[:int(len(images) * 0.8)]
    validation_set = images[int(len(images) * 0.8):]
    load_func = lambda set: load_dataset(set, batch_size, corruption_func, model.input_shape[2:])

    model.compile(loss = "mean_squared_error", optimizer = Adam(beta_2 = 0.9))
    model.fit_generator(load_func(training_set), samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data = load_func(validation_set), nb_val_samples=num_valid_sample)

def restore_image(corrupted_image, base_model, num_channels):

    new_model = build_nn_model(corrupted_image.shape[0], corrupted_image.shape[1], num_channels)
    new_model.set_weights(base_model.get_weights())
    prediction = new_model.predict(corrupted_image[np.newaxis, np.newaxis] - 0.5)[0] + 0.5
    return np.clip(prediction,0,1).reshape(corrupted_image.shape).astype(np.float32)

def learn_denoising_model(quick_mode = False):

    if quick_mode:
        batch_size = 10
        sam_per_epoch = 30
        num_epochs = 2
        num_valid_sample = 30
    else:
        batch_size = 100
        sam_per_epoch = 10000
        num_epochs = 5
        num_valid_sample = 1000

    model = build_nn_model(24, 24, 48)

    train_model(model, ut.images_for_denoising(),
                lambda im: add_gaussian_noise(im, 0, 0.2),
                batch_size, sam_per_epoch, num_epochs, num_valid_sample)

    return model, 48

def random_motion_blur(image, list_of_kernel_sizes):

    return add_motion_blur(image, np.random.choice(list_of_kernel_sizes), np.random.uniform(high = np.pi))

def learn_deblurring_model(quick_mode = False):

    if quick_mode:
        batch_size = 10
        sam_per_epoch = 30
        num_epochs = 2
        num_valid_sample = 30
    else:
        batch_size = 100
        sam_per_epoch = 10000
        num_epochs = 10
        num_valid_sample = 1000

    model = build_nn_model(16, 16, 32)
    train_model(model, ut.images_for_deblurring(),
                lambda im: random_motion_blur(im, [7]),
                batch_size, sam_per_epoch, num_epochs, num_valid_sample)

    return model, 32