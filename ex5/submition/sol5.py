import numpy as np
import sol5_utils as ut
from scipy.misc import imread
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from scipy.ndimage import filters
from skimage.color import rgb2gray
from keras.layers import Input, Convolution2D, Activation, merge


RELU_DEEPNESS = 7

def read_image(filename, representation):
    im = imread(filename) / 255

    if(representation == 1):
        im = rgb2gray(im)

    return im.astype(np.float32)


def load_dataset(filenames, batch_size, corruption_func, crop_size):

    im_dict = {}
    currupt_im_dict = {}

    source = np.empty((batch_size, 1, crop_size[0], crop_size[1]), np.float32)
    target = np.empty_like(source)

    while True:

        rand_files = np.random.choice(filenames, batch_size)

        for i in range(batch_size):

            if rand_files[i] in im_dict:
                im = im_dict[rand_files[i]]
                currupt_im = currupt_im_dict[rand_files[i]]

            else:
                im = read_image(rand_files[i], 1)
                currupt_im = corruption_func(im)
                im_dict[rand_files[i]] = im
                currupt_im_dict[rand_files[i]] = currupt_im

            rows = np.random.randint(im.shape[0] - crop_size[0])
            cols = np.random.randint(im.shape[1] - crop_size[1])

            source[i, 0] = currupt_im[rows:rows + crop_size[0], cols:cols + crop_size[1]]
            target[i, 0] = im[rows:rows + crop_size[0], cols:cols + crop_size[1]]

        yield (source - 0.5, target - 0.5)


conv_func = lambda channels, tensor: Convolution2D(channels, 3, 3, border_mode = "same")(tensor)

def resblock(input_tensor, num_channels):
    return merge([input_tensor,
                  conv_func(num_channels, Activation("relu")(conv_func(num_channels, input_tensor)))], mode = "sum")

def build_nn_model(height, width, num_channels):

    input = Input(shape = (1, height, width))
    original_relu = Activation("relu")(conv_func(num_channels, input))

    relu = original_relu
    for i in range(RELU_DEEPNESS):
        relu = resblock(relu, num_channels)

    output = conv_func(1, merge([original_relu, relu], mode = "sum"))
    return Model(input, output)

def train_model(model, images, corruption_func, batch_size,
            samples_per_epoch, num_epochs, num_valid_sample):

    training_set = images[:int(len(images) * 0.8)]
    validation_set = images[int(len(images) * 0.8):]
    load_func = lambda set: load_dataset(set, batch_size, corruption_func, model.input_shape[2:])

    model.compile(loss = "mean_squared_error", optimizer = Adam(beta_2 = 0.9))
    model.fit_generator(load_func(training_set), samples_per_epoch = samples_per_epoch, nb_epoch = num_epochs,
                        validation_data = load_func(validation_set), nb_val_samples = num_valid_sample)

def restore_image(corrupted_image, base_model, num_channels):

    new_model = build_nn_model(corrupted_image.shape[0], corrupted_image.shape[1], num_channels)
    new_model.set_weights(base_model.get_weights())
    prediction = new_model.predict(corrupted_image[np.newaxis, np.newaxis] - 0.5)[0] + 0.5
    return np.clip(prediction,0,1).reshape(corrupted_image.shape).astype(np.float32)

def add_gaussian_noise(image, min_sigma, max_sigma):

    return np.clip(image + np.random.normal(scale = np.random.uniform(min_sigma, max_sigma), size = image.shape).astype(np.float32),
                   0, 1)

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

def add_motion_blur(image, kernel_size, angle):

    return filters.convolve(image, ut.motion_blur_kernel(kernel_size, angle))

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


if __name__ == "__main__":

    test = 3
    if test == 1:
        im = read_image("image_dataset/train/35010.jpg", 1)
        noised_im = add_gaussian_noise(im, 0, 0.2)
        plt.imshow(noised_im, cmap=plt.cm.gray)
        plt.show()

    if test == 2:
        im = read_image("text_dataset/train/0000005_orig.png", 1)
        noised_im = random_motion_blur(im, [7])

        model, channels = learn_deblurring_model()
        restored = restore_image(noised_im, model, channels)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, label="noise")
        ax1.title.set_text("noise")
        plt.imshow(noised_im, cmap=plt.cm.gray)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.title.set_text("restored")
        plt.imshow(restored, cmap=plt.cm.gray)
        plt.show()

    if test == 3:
        im = read_image("image_dataset/train/35010.jpg", 1)
        noised_im = add_motion_blur(im, 11, 0.2)

        model, channels = learn_denoising_model()
        restored = restore_image(noised_im, model, channels)

        fig = plt.figure()

        ax1 = fig.add_subplot(2, 2, 1, label="noise")
        ax1.title.set_text("noise")
        plt.imshow(noised_im, cmap=plt.cm.gray)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.title.set_text("restored")
        plt.imshow(restored, cmap=plt.cm.gray)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.title.set_text("Original")
        plt.imshow(im, cmap=plt.cm.gray)

        plt.show()