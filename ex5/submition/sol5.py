import random
import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, merge
from keras.optimizers import Adam

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
        rand_row = np.trunc(np.multiply(rand[0] , (im.shape[0] - crop_size[0]))).astype(np.int_)
        rand_col = np.trunc(np.multiply(rand[1] , (im.shape[1] - crop_size[1]))).astype(np.int_)

        source_batch = np.empty(batch_size, 1, crop_size[0], crop_size[1])
        target_batch = np.empty(source_batch.shape)

        for i in range(batch_size):
            source_batch[i, 0] = currupt_im[rand_row[i]:rand_row[i] + currupt_im.shape[0],
                                            rand_col[i]:rand_col[i] + currupt_im.shape[1]]
            target_batch[i, 0] = im[rand_row[i]:rand_row[i] + im.shape[0], rand_col[i]:rand_col[i] + im.shape[1]]

        yield (source_batch, target_batch)

conv_func = lambda channels, tensor: Convolution2D(channels, 3, 3, border_mode = "same")(tensor)

def resblock(input_tensor, num_channels):
    return merge([input_tensor,
                  conv_func(num_channels, Activation("relu")(conv_func(num_channels, input_tensor)))], mode = "sum")


def build_nn_model(height, width, num_channels):

    def applay_resblock(ReLu, iter_times):
        if iter_times: return ReLu
        return applay_resblock(resblock(ReLu, num_channels), iter_times - 1)

    input = Input(shape = (1, height, width))
    relu = Activation("relu")(conv_func(num_channels, input))
    output = conv_func(1, merge([relu, applay_resblock(relu, 5)], mode = "sum"))
    return Model(input, output)

def train_model(model, images, corruption_func, batch_size,
            samples_per_epoch, num_epochs, num_valid_sample):

    training_set = images[np.floor(len(images) * 0.8)]
    validation_set = images[np.ceil(len(images) * 0.2)]
    load_func = lambda set: load_dataset(set, batch_size, corruption_func, model.input_shape[2:])

    model.compile(loss = "mean_squared_error", optimizer = Adam(beta_2 = 0.9))
    model.fit_generator(load_func(training_set), samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data = load_func(validation_set), nb_val_samples=num_valid_sample)

def restore_image(corrupted_image, base_model, num_channels):

    new_model = build_nn_model(corrupted_image.shape[0], corrupted_image.shape[1], num_channels)
    new_model.set_weights(base_model.get_weights())
    prediction = new_model.predict(corrupted_image[np.newaxis, np.newaxis] - 0.5)[0] + 0.5
    return np.clip(prediction,0,1).reshape(corrupted_image.shape)

def add_gaussian_noise(image, min_sigma, max_sigma):

    return np.clip(image + np.random.normal(scale=np.random.uniform(min_sigma, max_sigma),
                                            size=image.shape), 0, 1)
def learn_denoising_model(quick_mode=False):

    return learn_X_model(quick_mode, sut.images_for_denoising, "gaussian_noise", 24, 48, 5)

def add_motion_blur(image, kernel_size, angle):

    return flt.convolve(image, sut.motion_blur_kernel(kernel_size, angle))

def random_motion_blur(image, list_of_kernel_sizes):

    return add_motion_blur(image, np.random.choice(list_of_kernel_sizes), np.random.uniform(high=np.pi))

def learn_deblurring_model(quick_mode=False):

    return learn_X_model(quick_mode, sut.images_for_deblurring, "random_motion_blur", 16, 32, 10)