"""
Tests Lucy-Richardson deconvolution as implemented in phantom.

---
Made by: ayrton30
"""
import numpy as np
import cv2


from phantom.enhance import lucy_richardson_deconv


# example
# sigmaG: sigma of point spread function
sigma = 6.0
winSize = 8 * sigma + 1

img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', img)

img = np.float64(img)   # convert to -> img.convertTo(img, CV_64F);

# Blur the original image
img = cv2.GaussianBlur(img, (int(winSize), int(winSize)), sigma, sigma)
# en C++ ->GaussianBlur(img_origen, img_destino, Size(winSize,winSize), sigmaG, sigmaG )

cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
cv2.imshow('original_blur', img)

result = lucy_richardson_deconv(img, 200, sigma)
cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()