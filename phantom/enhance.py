"""
Image enhancement algorithms.
"""

import cv2
import numpy as np

# TO READ:
# * http://onsignalandimageprocessing.blogspot.com/2017/02/lucy-richardson-deconvolution.html


def lucy_richardson_deconv(img, num_iterations, sigmag):
    """" Lucy-Richardson Deconvolution Function
    // input-1 img: NxM matrix image
    // input-2 num_iterations: number of iterations
    // input-3 sigma: sigma of point spread function (PSF)
    // output result: deconvolution result
    """

    epsilon = 2.2204e-16
    win_size = 8 * sigmag + 1   # Window size of PSF

    # Initializations Numpy
    j1 = img.copy()
    j2 = img.copy()
    w_i = img.copy()
    im_r = img.copy()

    t1 = np.zeros((np.size(img, 0), np.size(img, 1), 1), dtype="double")
    t2 = np.zeros((np.size(img, 0), np.size(img, 1), 1), dtype="double")
    tmp1 = np.zeros((np.size(img, 0), np.size(img, 1), 1), dtype="double")
    tmp2 = np.zeros((np.size(img, 0), np.size(img, 1), 1), dtype="double")
    # size = (w, h, channels), grayscale -> channels = 1

    # Lucy - Rich.Deconvolution CORE
    lambda_ = 0
    for j in range(1, num_iterations):
        if j > 1:
            # calculation of lambda
            # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#multiply
            cv2.multiply(t1, t2, tmp1)   # Calculates the per-element scaled product of two arrays.
            cv2.multiply(t2, t2, tmp2)

            # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#sum
            lambda_ = cv2.sumElems(tmp1)[0] / (cv2.sumElems(tmp2)[0] + epsilon)

        # y = j1 + (lambda_ * (j1 - j2))
        #subtract =
        multiply = np.multiply(lambda_, np.subtract(j1, j2))
        y = j1 + multiply

        y[(y < 0)] = 0

        # applying Gaussian filter
        re_blurred = cv2.GaussianBlur(y, (int(win_size), int(win_size)), sigmag)
        re_blurred[(re_blurred <= 0)] = epsilon

        cv2.divide(w_i, re_blurred, im_r)
        im_r = im_r + epsilon

        # applying Gaussian filter
        im_r = cv2.GaussianBlur(im_r, (int(win_size), int(win_size)), sigmag)

        j2 = j1.copy()
        cv2.multiply(y, im_r, j1)

        t2 = t1.copy()
        t1 = j1 - y

    result = j1.copy()
    return result
