import os, sys, traceback
import numpy as np
from scipy.misc import imread
from skimage import color
import current.sol5 as sol5
import current.sol5_utils as sol5_utils

def read_image(filename, representation):
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im).astype(np.float32)
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    return im

def presubmit():
    print ('ex5 presubmission script')
    disclaimer="""
    Disclaimer
    ----------
    The purpose of this script is to make sure that your code is compliant
    with the exercise API and some of the requirements
    The script does not test the quality of your results.
    Don't assume that passing this script will guarantee that you will get
    a high grade in the exercise
    """
    print (disclaimer)
    
    print('=== Check Submission ===\n')
    if not os.path.exists('current/README'):
        print ('No readme!')
        return False
    with open ('current/README') as f:
        lines = f.readlines()
    print ('login: ', lines[0])
    print ('submitted files:\n' + '\n'.join(map(lambda x: x.strip(), lines[1:])))
    
    print('\n=== Answers to questions ===')
    for q in [1]:
        if not os.path.exists('current/answer_q%d.txt'%q):
            print ('No answer_q%d.txt!'%q)
            return False
        print ('\nAnswer to Q%d:'%q)
        with open('current/answer_q%d.txt'%q) as f:
            print (f.read())
    
    print ('\n=== Section 3 ===\n')
    image_sets = [sol5_utils.images_for_denoising(), sol5_utils.images_for_deblurring()]
    batch_size = 100
    corruption_func = lambda x: x
    crop_size = (16, 16)
    print('Trying to load basic datasets...')
    for image_set in image_sets:
        try:
            train_set = sol5.load_dataset(image_set, batch_size, corruption_func, crop_size)
            for i in range(3):
                source, target = next(train_set)
                if source.dtype != np.float32 or target.dtype != np.float32:
                    print('source dtype: ', source.dtype)
                    raise ValueError('Wrong dtype in mini-batch!')
                if source.shape != target.shape or source.shape != (batch_size, 1, crop_size[0], crop_size[1]):
                    raise ValueError('Wrong dimensions returned by source / target mini-batches!')
        except:
            print(traceback.format_exc())
            return False
    print('\tPassed!')

    print ('\n=== Section 4 ===\n')
    print ('Trying to build a model...')
    num_channels = 32
    try:
        model = sol5.build_nn_model(crop_size[0], crop_size[1], num_channels)
        if model is None:
            raise ValueError('Returned None!')
        if model.input_shape[-3:] != (1, crop_size[0], crop_size[1]):
            raise ValueError('Wrong input shape to model!')
    except:
        print(traceback.format_exc())
        return False
    print('\tPassed!')

    print ('\n=== Section 5 ===\n')
    for image_set in image_sets:
        print ('Trying to train a model...')
        try:
            train_func = lambda x: x
            batch_size = 10
            samples_per_epoch = 20
            epochs = 2
            valid_samples = 20
            sol5.train_model(model, image_set, train_func, batch_size,
                samples_per_epoch, epochs, valid_samples)
        except:
            print(traceback.format_exc())
            return False
        print ('\tPassed!')

    print ('\n=== Section 6 ===\n')
    print ('Trying to restore image... (not checking for quality!)')
    im_orig = read_image('presubmit_externals/text.png', 1)
    try:
        im_fixed = sol5.restore_image(im_orig, model, num_channels)
        if im_fixed.shape != im_orig.shape:
            raise ValueError('Returned image has a different shape than original!')
        if im_fixed.dtype != np.float32:
            raise ValueError('Returned image is not float32!')
    except:
        print(traceback.format_exc())
        return False
    print ('\tPassed!')

    print ('\n=== Section 7 ===\n')
    print('=== Image Denoising ===')
    try:
        im_orig = read_image('presubmit_externals/image.jpg', 1)
        print('Trying to apply random noise on image...')
        im_corrupted = sol5.add_gaussian_noise(im_orig, 0, 0.2)
        if im_orig.shape != im_corrupted.shape:
            raise ValueError('Returned dimensions of noisy image do not match input!')
        if im_corrupted.dtype != np.float32:
            raise ValueError('Returned noisy image is not float32!')
        print('\tPassed!')
        print('Trying to learn a denoising model...')
        model, num_channels = sol5.learn_denoising_model(quick_mode=True)
        print('\tPassed!')
        print('Trying to use learned model for denoising... (not checking for quality!)')
        im_fixed = sol5.restore_image(im_corrupted, model, num_channels)
        if im_corrupted.shape != im_corrupted.shape:
            raise ValueError('Returned dimensions of fixed image do not match input!')
        if im_fixed.dtype != np.float32:
            raise ValueError('Returned fixed image is not float32!')
        print('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    print('=== Image Deblurring ===')
    try:
        im_orig = read_image('presubmit_externals/text.png', 1)
        print('Trying to apply random motion blur on image...')
        im_corrupted = sol5.random_motion_blur(im_orig, [7])
        if im_orig.shape != im_corrupted.shape:
            raise ValueError('Returned dimensions of blurred image do not match input!')
        if im_corrupted.dtype != np.float32:
            raise ValueError('Returned blurred image is not float32!')
        print('\tPassed!')
        print('Trying to learn a deblurring model...')
        model, num_channels = sol5.learn_deblurring_model(quick_mode=True)
        print('\tPassed!')
        print('Trying to use learned model for deblurring... (not checking for quality!)')
        im_fixed = sol5.restore_image(im_corrupted, model, num_channels)
        if im_corrupted.shape != im_corrupted.shape:
            raise ValueError('Returned dimensions of fixed image do not match input!')
        if im_fixed.dtype != np.float32:
            raise ValueError('Returned fixed image is not float32!')
        print('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    print ('\n=== All tests have passed ===');
    print ('=== Pre-submission script done ===\n');
    
    print ("""
    Please go over the output and verify that there are no failures/warnings.
    Remember that this script tested only some basic technical aspects of your implementation
    It is your responsibility to make sure your results are actually correct and not only
    technically valid.""")
    return True
