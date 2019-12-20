import numpy as np
from cv2 import cv2
from matplotlib import pyplot
import glob
from collections import defaultdict
from sklearn.cluster import DBSCAN
import math

# tip:
# * https://dsp.stackexchange.com/questions/1288/what-are-some-free-alternatives-to-sift-surf-that-can-be-used-in-commercial-app
# 
# Polygons and testing a point to see if it falls inside:
# * https://stackoverflow.com/questions/13786088/determine-if-a-point-is-inside-or-outside-of-a-shape-with-opencv
# * https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/
#   cv2.convexHull is used in the face morphing example to make the mask for
#   the faces.

MIN_MATCH_COUNT = 10

def find_matches(desc1, desc2, kps2):
    """
    Finds a list of matches between two images, and groups them in clusters

    :param desc1: numpy.ndarray of descriptors from the first image
    :param desc2: numpy.ndarray of descriptors from the second image
    :param kps2: list of keypoints from the second image (list[cv2.KeyPoint])
    :return: None if not enough matches were found, or a dictionary of
        clusters, where the key is the number of cluster, and the value is a
        list of matches
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1,desc2,k=2)
    
    good_matches = []
    for match in matches:
        if len(match)>1 and match[0].distance < match[1].distance*0.7:
            good_matches.append(match[0])

    if len(good_matches) < MIN_MATCH_COUNT:
        return None

    clustered_matches = matches_filter(good_matches, kps2)

    return clustered_matches

def count_different_points(matches, kps):
    """
    Counts the amount of different points in the second image,
    that matched with at least one of the points in the first image

    :param matches: list of matches between the two images
    :param kps: list of kps from the second image
    :return: Number of different points that matched
    """
    dst_pts = np.float32([ kps[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    unique_pts = []
    for point in dst_pts:
        if (point[0][0], point[0][1]) in unique_pts:
            continue
        unique_pts.append((point[0][0], point[0][1]))  

    return len(unique_pts)  

def matches_filter(matches, kps2):
    """
    Removes matches if they are too far away from the rest or all in the same
    position, and regroups in new clusters if needed

    :param matches: List of matches between two images
    :param kps2: List of keypoints from the second image
    :return: Dictionary of clusters, where the key is the number of cluster, and the
        value is a list of matches (at least 3). If all the matches are on the same
        position, returns None
    """
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    pts = [point[0] for point in dst_pts]

    total = 0
    dist = 0
    for i, point in enumerate(pts):
        for second_point in pts[i:]:
            dist += math.sqrt((point[0]-second_point[0])**2 + (point[1]-second_point[1])**2)
            total += 1
    e = dist/total 
    if e == 0:
        return None

    db = DBSCAN(eps=e, min_samples=3).fit(pts)
    labels = db.labels_

    clustered_matches = defaultdict(list)
    for match, label in zip(matches, labels):
        if label < 0:
            continue
        clustered_matches[label].append(match)

    return clustered_matches

def draw_object(img1, kp1, img2, kps2, matches):
    """
    Draws the object found on the second image

    :param img1: First image
    :param kp1: List of keypoints from the first image
    :param img2: Second image
    :param kps2: List of keypoints from the second image
    :param matches: List of matches between the two images
    :return: modified image
    """
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)[0]

    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img_res = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    return img_res

def draw_keymatches (matches, kps, img, color=0):
    """
    Draws a circle on the position of the matches between two images

    :param matches: List of matches between two images
    :param kps: List of keypoints from the second image
    :param img: Second image
    :param color: Color of the circles
    :return: Modified image
    """
    colors = [[0,255,0],[0,0,255],[255,255,255],[100,100,100],[200,200,200],[0,0,0],[0,0,128],[0,128,0]]
    if color > 7:
        color = [255,0,0]
    else:
        color = colors[color]

    img_res = img
    dst_pts = np.float32([ kps[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    points = []
    for point in dst_pts:
        if (point[0][0], point[0][1]) in points:
            continue
        points.append((point[0][0], point[0][1]))
    for point in points:
        img_res = cv2.circle(img_res, point, 10, color, thickness=2)

def keypoint_clustering(kps, desc, img, e=0.025):
    """
    Groups the keypoints and the Descriptors of an image in different clusters according
    to their distance with eachother

    :param kps: List of keypoints
    :param desc: List of Descriptors
    :param img: Image
    :param e: Percentage of the image minimum axis, to determine the maximum distance
        between each point for the clustering. Default = 0.025
    :return: Dictionary of Keypoints, where the key is the cluster number and the value
        is the list of keypoints, and a Dictionary of Descriptors, where the key is the
        cluster number and the value is the list of descriptors
    """
    # Clustering of keyPoints
    pts = np.int32([kp.pt for kp in kps])
    eps = e * min(img.shape[:2])
    db = DBSCAN(eps=eps, min_samples=3).fit(pts)
    labels = db.labels_

    # Groups keypoints and descriptors with their respective cluster,
    # filtering the ones that are "noice"
    good_kps = defaultdict(list)
    good_desc = defaultdict(list)
    for kp, label, d in zip(kps, labels, desc):
        if label < 0:
            continue
        good_kps[label].append(kp)
        good_desc[label].append(d)
    
    return good_kps, good_desc

def find_object_in_image(img1, img2, e=0.025, drawKM=False):
    """
    Finds an object or image on another image, and draws it

    :param img1: Image to be found
    :param img2: Image to be found on
    :param e: Percentage of the image minimum axis, to determine the maximum distance
        between each point for the clustering. Default = 0.025
    :param drawKM (default=False): Boolean (True -> Also draws keyMatches)
    :return: Number of occurencies of the first image on the second one, and a
        new image with the position of the occurencies
    """

    # Create SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    kps1, desc1 = sift.detectAndCompute(img1,None)
    kps2, desc2 = sift.detectAndCompute(img2,None)
    good_kps, good_desc = keypoint_clustering(kps2, desc2, img2, e=e)

    img_res = img2

    found = 0
    for key in good_kps:
        if len(good_desc[key]) < 10:
            continue
        good_matches = find_matches(desc1, np.array(good_desc[key], dtype=np.float32), good_kps[key])

        if not(good_matches):
            continue

        for key_m in good_matches:
            matches = good_matches[key_m]
            unique_pts = count_different_points(matches, good_kps[key])
            if unique_pts >= MIN_MATCH_COUNT:
                found += 1
                if drawKM:
                    draw_keymatches(matches, good_kps[key], img_res, found-1)
                try:
                    img_res = draw_object(img1, kps1, img_res, good_kps[key], matches)
                except cv2.error:
                    pass
    return found, img_res

def find_object_in_directory(img1, path, e=0.025, drawKM=False):
    """
    [GENERATOR] Finds an object or image on a directory, searching every image, and drawing it

    :param img1: Image to be found
    :param path: path/to/directory/*.jpg (Could use any file extension, or none)
    :param e (default=0.025): Percentage of the image minimum axis, to determine the maximum distance
        between each point for the clustering. Default = 0.025
    :param drawKM (default=False): Boolean, True -> Also draws keyMatches
    :return: [GENERATOR] Number of occurencies of the first image on the second one, and a new image with the position of the occurencies
    """

    sift = cv2.xfeatures2d.SIFT_create()

    # Detecting keypoints and descriptors from the first image
    kps1, desc1 = sift.detectAndCompute(img1,None)

    for i, filepath in enumerate(glob.iglob(path), start=1):
        print(f"Procesando imagen {i}: {filepath}")
        found = False

        # Loading and detecting keypoints and descriptors from Image
        img2 = cv2.imread(filepath)
        kps2, desc2 = sift.detectAndCompute(img2,None)

        good_kps, good_desc = keypoint_clustering(kps2, desc2, img2, e=e)

        imgRes = img2

        # Finding Matches
        found = 0
        for key in good_kps:
            if len(good_desc[key]) < 10:
                continue
            good_matches = find_matches(desc1, np.array(good_desc[key], dtype=np.float32), good_kps[key])

            if not(good_matches):
                continue

            for keyM in good_matches:
                matches = good_matches[keyM]
                amount = count_different_points(matches, good_kps[key])
                if amount >= MIN_MATCH_COUNT:
                    found += 1
                    if drawKM:
                        draw_keymatches(matches, good_kps[key], imgRes, found-1)
                    try:
                        imgRes = draw_object(img1, kps1, imgRes, good_kps[key], matches)
                    except cv2.error:
                        pass
        yield found, imgRes

# ------------------------------------------------------- #

# EXAMPLE WITH ONE IMAGE
""" img1 = cv2.imread('C:/InfoLab/Python/find-object/img/libro.jpeg')
img2 = cv2.imread('C:/InfoLab/Python/find-object/img/completa.jpeg')
number, img = find_object_in_image(img1, img2, drawKM=True)

if number > 0:
    print('Object found ' + str(number) + ' times')
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    cv2.imshow('Img', img)
    cv2.waitKey()
else:
    print('The object is not in the image') """

# EXAMPLE WITH A DIRECTORY
""" img1 = cv2.imread('C:/InfoLab/Python/find-object/img/libro.jpeg')
path = 'C:/InfoLab/Python/find-object/img/*.jpg'
res = find_object_in_directory(img1, path, drawKM=True)

for i, (number, img) in enumerate(res):
    if number > 0:
        print('Object found ' + str(number) + ' times')
        cv2.imwrite(f'C:/InfoLab/Python/find-object/res/{i} - encontrado {str(number)}.jpg', img)
    else:
        print('The object is not in the image') """