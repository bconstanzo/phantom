"""
Hashing techniques aplicable to images.
"""

# Read on:
# * SIFT, SURF and ORB descriptors
# * Perceptual Hashes: aHash, pHash, dHash and wHash
#   * http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.htmlv
#   * http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
#   * https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5
# * HoOG
# * 
# 

import cv2
import numpy as np
import scipy.fftpack


def compare(h1, h2, threshold=12):
    """
    Compare two hashes (in numpy array form) and return True for equivalence,
    False otherwise.

    :param h1: first hash to compare
    :param h2: second hash to compare
    :param threshold: similarity threhshold on the Hamming distance
    """
    return np.sum(h1 ^ h2) <= threshold


def a_hash(source):
    """
    Image similarity hash based on the average colour of an image.

    Read more on:

    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    """
    pre = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    pre = cv2.resize(pre, (8, 8), interpolation=cv2.INTER_AREA)
    average = np.average(pre)
    hash_ = np.zeros((8, 8), dtype=np.uint8)
    hash_[pre > average] = 1
    return hash_


def d_hash(source):
    """
    Image similarity hash based on horizontal gradients.

    Read more on:

    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
    """
    pre = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    pre = cv2.resize(pre, (9, 8), interpolation=cv2.INTER_AREA)
    pre = pre.astype(np.int16)
    diff = pre[:,:8] - pre[:, 1: 9]
    hash_ = (np.sign(diff) + 1) / 2
    return hash_.astype(np.uint8)


def p_hash(source):
    """
    Image similarity hash based on the DCT components of an image.

    Read more on:

    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    """
    pre = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    pre = cv2.resize(pre, (32, 32), interpolation=cv2.INTER_AREA)
    pre = scipy.fftpack.dct(pre, type=2)
    average = np.average(pre[:8, :8].flatten()[1:])
    hash_ = np.zeros((8, 8), dtype=np.uint8)
    hash_[pre[:8, :8] > average] = 1
    return hash_
