"""
Image enhancement algorithms.
"""

import cv2
import numpy as np

# TODO: explore notch filters for moire artifact removal.
#       references:
#       * https://ijournals.in/wp-content/uploads/2017/07/5.3106-Khanjan.compressed.pdf
#       * https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2018.5707
#       * https://eej.aut.ac.ir/article_94_befd8a642325852c3a0d41ece10b3feb.pdf
#       * https://arxiv.org/abs/1701.09037
#       * (OpenCV 3.4) https://docs.opencv.org/3.4.15/d2/d0b/tutorial_periodic_noise_removing_filter.html

# TO READ:
# * http://onsignalandimageprocessing.blogspot.com/2017/02/lucy-richardson-deconvolution.html

def _clip(img, lower, higher):
    """
    Clips an image in between two bounds.

    :param img: cv2/np.ndarray image
    :param lower: lower bound, usually 0
    :param higher: higher bound, either 255 for uint8 images or 1.0 for floats
    """
    return np.maximum(lower, np.minimum(img, higher))


def lucy_richardson_deconv(img, num_iterations, sigmag, *, clip=True):
    """"
    Lucy-Richardson Deconvolution.

    :param img: cv2/np.darray imge
    :param num_iterations: number of iterations
    :param sigma: sigma of (Gaussian) point spread function (PSF)
    :return: deconvolution result
    """

    epsilon = 2.2204e-16
    win_size = 8 * sigmag + 1   # Window size of PSF

    dtype = img.dtype

    if img.dtype == "uint8":
        clip_max = 255
        clip_min = 0
    elif img.dtype == "uint16":
        clip_max = 65535
        clip_min = 0
    else:
        # we must have a float here
        clip_max = 1.0
        clip_min = 0.0

    # Initializations Numpy
    j1 = img.copy()
    j2 = img.copy()
    w_i = img.copy()
    im_r = img.copy()

    t1 = np.zeros(img.shape, dtype=np.float32)
    t2 = np.zeros(img.shape, dtype=np.float32)
    tmp1 = np.zeros(img.shape, dtype=np.float32)
    tmp2 = np.zeros(img.shape, dtype=np.float32)
    # size = (w, h, channels), grayscale -> channels = 1

    # Lucy - Rich.Deconvolution CORE
    lambda_ = 0
    for j in range(1, num_iterations):
        # gotta clean this up, maybe a warmup before the for-loop
        if j > 1:
            # calculation of lambda
            # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#multiply
            tmp1 = t1 * t2
            tmp2 = t2 * t2

            # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#sum
            lambda_ = cv2.sumElems(tmp1)[0] / (cv2.sumElems(tmp2)[0] + epsilon)

        # y = j1 + (lambda_ * (j1 - j2))
        y = j1 + np.multiply(lambda_, np.subtract(j1, j2))

        y[(y < 0)] = 0

        # applying Gaussian filter
        re_blurred = cv2.GaussianBlur(y, (int(win_size), int(win_size)), sigmag)
        re_blurred[(re_blurred <= 0)] = epsilon

        cv2.divide(w_i, re_blurred, im_r, 1, cv2.CV_64F)
        im_r = im_r + epsilon

        # applying Gaussian filter
        im_r = cv2.GaussianBlur(im_r, (int(win_size), int(win_size)), sigmag)

        # updates before the next iteration
        j2 = j1.copy()
        j1 = y * im_r
        t2 = t1.copy()
        t1 = j1 - y

    result = _clip(j1.copy(), clip_min, clip_max)
    return result.astype(dtype)
