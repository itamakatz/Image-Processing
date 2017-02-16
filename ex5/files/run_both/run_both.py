import numpy as np
import same_funcs as same
import sol5_itamar as ita
import sol5_liav as liav
import matplotlib.pyplot as plt

if __name__ == "__main__":

    test = 3
    if test == 1:
        im = same.read_image("train/2092.jpg", 1)
        noised_im = same.add_gaussian_noise(im, 0, 0.2)
        plt.imshow(noised_im, cmap=plt.cm.gray)
        plt.show()

    # if test == 2:
    #     im = same.read_image("text_dataset/train/0000005_orig.png", 1)
    #     noised_im = same.random_motion_blur(im, [7])
    #
    #     model, channels = learn_deblurring_model()
    #     restored = restore_image(noised_im, model, channels)
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1, 2, 1, label="noise")
    #     ax1.title.set_text("noise")
    #     plt.imshow(noised_im, cmap=plt.cm.gray)
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     ax2.title.set_text("restored")
    #     plt.imshow(restored, cmap=plt.cm.gray)
    #     plt.show()

    if test == 3:
        im = same.read_image("image_dataset/train/2092.jpg", 1)
        noised_im = same.add_motion_blur(im, 7, 0.2)

        model_liav, channels_liav = liav.learn_denoising_model()
        restored_liav = liav.restore_image(noised_im, model_liav, channels_liav)

        model_ita, channels_ita = ita.learn_denoising_model()
        restored_ita = ita.restore_image(noised_im, model_ita, channels_ita)

        fig = plt.figure()

        ax1_liav = fig.add_subplot(3, 2, 1, label="noise")
        ax1_liav.title.set_text("noise - liav")
        plt.imshow(noised_im, cmap=plt.cm.gray)

        ax1_itamar = fig.add_subplot(3, 2, 2)
        ax1_itamar.title.set_text("noise - itamar")
        plt.imshow(noised_im, cmap=plt.cm.gray)

        ax2_liav = fig.add_subplot(3, 2, 3)
        ax2_liav.title.set_text("restored - liav")
        plt.imshow(restored_liav, cmap=plt.cm.gray)

        ax2_itamar = fig.add_subplot(3, 2, 4)
        ax2_itamar.title.set_text("restored - itamar")
        plt.imshow(restored_ita, cmap=plt.cm.gray)

        ax3_liav = fig.add_subplot(3, 2, 5)
        ax3_liav.title.set_text("Original - liav")
        plt.imshow(im, cmap=plt.cm.gray)

        ax3_itamar = fig.add_subplot(3, 2, 6)
        ax3_itamar.title.set_text("Original - itamar")
        plt.imshow(im, cmap=plt.cm.gray)

        plt.show()