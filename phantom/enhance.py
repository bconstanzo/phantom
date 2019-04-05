"""
Image enhancement algorithms.
"""

import cv2
import numpy as np


def lucy_richardson_deconv(img, num_iterations, sigmag):
    """" Lucy-Richardson Deconvolution Function
    // input-1 img: NxM matrix image
    // input-2 num_iterations: number of iterations
    // input-3 sigma: sigma of point spread function (PSF)
    // output result: deconvolution result
    """

    epsilon = 2.2204e-16

    # Window size of PSF
    win_size = 8 * sigmag + 1

    # Initializations Numpy
    J1 = img.copy()
    J2 = img.copy()
    wI = img.copy()
    imR = img.copy()

    # T1, T2, tmpMat1, tmpMat2 -> Numpy
    T1 = np.zeros((np.size(img, 0), np.size(img, 1), 1), dtype="double")
    T2 = np.zeros((np.size(img, 0), np.size(img, 1), 1), dtype="double")
    tmpMat1 = np.zeros((np.size(img, 0), np.size(img, 1), 1), dtype="double")
    tmpMat2 = np.zeros((np.size(img, 0), np.size(img, 1), 1), dtype="double")
    # size = (w, h, channels), grayscale -> channels = 1

    # Lucy - Rich.Deconvolution CORE
    lambda_ = 0
    for j in range(1, num_iterations):
        if j > 1:
            # calculation of lambda
            # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#multiply
            cv2.multiply(T1, T2, tmpMat1)   # Calculates the per-element scaled product of two arrays.
            cv2.multiply(T2, T2, tmpMat2)

            # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#sum
            lambda_ = cv2.sumElems(tmpMat1)[0] / (cv2.sumElems(tmpMat2)[0] + epsilon)
            print("lambda [", j, "]", lambda_)

        # Y = J1 + (lambda_ * (J1 - J2))
        resta = np.subtract(J1, J2)
        multi = np.multiply(lambda_, resta)
        Y = J1 + multi

        Y[(Y < 0)] = 0

        # 1) applying Gaussian filter
        reBlurred = cv2.GaussianBlur(Y, (int(win_size), int(win_size)), sigmag)
        reBlurred[(reBlurred <= 0)] = epsilon

        # 2)
        cv2.divide(wI, reBlurred, imR)
        imR = imR + epsilon

        # 3) applying Gaussian filter
        imR = cv2.GaussianBlur(imR, (int(win_size), int(win_size)), sigmag)

        # 4)
        J2 = J1.copy()
        cv2.multiply(Y, imR, J1)

        T2 = T1.copy()
        T1 = J1 - Y

    result = J1.copy()
    return result


# example
# sigmaG: sigma of point spread function
sigma = 6.0
winSize = 8 * sigma + 1

img = cv2.imread('doge.jpg', cv2.IMREAD_GRAYSCALE)    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', img)

img = np.float64(img)   # convert to -> img.convertTo(img, CV_64F);

# Blur the original image
img = cv2.GaussianBlur(img, (int(winSize), int(winSize)), sigma, sigma)

cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
cv2.imshow('original_blur', img)

res_img = lucy_richardson_deconv(img, 100, sigma)
cv2.normalize(res_img, res_img, 0, 1, cv2.NORM_MINMAX)

cv2.imshow('result', res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
