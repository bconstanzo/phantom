# pylint: disable=E0001
# pylint: disable=E1101

import cv2
import sys
import datetime
import dlib
import glob
import numpy as np
import os


from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from phantom.faces import compare, detect, encode
from phantom.utils import image_grid
from pprint import pprint
from sklearn.cluster import DBSCAN, KMeans


path  = r"D:\Storage-post-SSD\Wapp\WhatsApp Images"
path  = r"D:\Storage-post-SSD\gender\test\small"    # keeping the other path as a stress-test
procs = 3
DEBUG_TIMER = True
output_folder_path = r"D:\Storage-post-SSD\gender\cluster_test"


def read_and_find(path):
    """
    Reads an image from a path, and locates and encodes any face on it.

    :param path: path to an image
    :return: tuple of (cv2/np.ndarray, list of encodings, list of locations)
        or empty tuple if no face was found
    """
    print(f"Reading {path}...")
    img = cv2.imread(path)
    if img is None:
        return tuple()
    locations = detect(img)
    if not(locations):
        return tuple()
    return img, encode(img, locations=locations), locations, path


def cluster(resultset):
    """
    Works over the provided resultset to cluster the faces on it. resultset
    comes from `read_and_find`.

    :param resultset: list of results (tuples) from `read_and_find`. Each tuple
        has (cv2 image, list of encodings, list of face locations, path)
    """
    images = []          # list of cv2 images
    paths = []           # list of paths for each image
    faces = []           # flat list of encodings
    face_images = []     # list of cv2 images of each face found
    images_x_faces = []  # this way we can zip the encodings to an image/path

    for idx, packed in enumerate(resultset):
        img, encodings, locations, path = packed
        images.append(img)
        paths.append(path)
        for enc, loc in zip(encodings, locations):
            faces.append(enc)
            left, top, right, bottom = loc
            roi = img[top:bottom, left:right]
            face_images.append(roi)
            images_x_faces.append(idx)
    # the idea is simple, we apply DBSCAN with a basic configuration and use
    # its result to apply k-means clustering:
    print(f"Number of faces detected: {len(faces)}")
    t0 = datetime.datetime.now()
    db = DBSCAN(eps=0.475, min_samples=2).fit(faces)
    t1 = datetime.datetime.now()
    # we can now approximate how many people are present...
    num_people = len(set(i for i in db.labels_ if i >= 0))
    # ...and use k-means to identify all the labels that DBSCAN couldn't
    km = KMeans(init="k-means++", n_clusters=num_people, n_init=10).fit(faces)
    t2 = datetime.datetime.now()
    # now we group all the images for each cluster into a grid
    grid_images = defaultdict(list)
    count_outlier = 0
    for idx, (img, label) in enumerate(zip(face_images, km.labels_)):
        if img is not None:
            centroid = km.cluster_centers_[label]
            distance = compare(centroid, faces[idx])
            if distance < 0.475:
                try:
                    grid_images[label].append(cv2.resize(img, (96, 96)))
                except cv2.error:
                    print(f"Raised -: {paths[images_x_faces[idx]]}")
                    pass
            else:
                print(f"Clustered face too far away from the centroid. ({distance})")
                try:
                    out = cv2.resize(img, (96, 96))
                    cv2.imwrite(f"{output_folder_path}/outlier_grid_{label}_outlier_{count_outlier}.jpg", out)
                    count_outlier += 1
                except cv2.error:
                    pass

    labels_set = set(km.labels_)
    for label in labels_set:
        # TODO: change this to a more flexible approach
        if len(grid_images[label]) < 10:
            grid_size = (3,3)
        elif len(grid_images[label]) <= 25:
            grid_size = (5,5)
        elif len(grid_images[label]) <  50:
            grid_size = (7,7)
        elif len(grid_images[label]) <= 81:
            grid_size = (9,9)
        elif len(grid_images[label]) <= 121:
            grid_size = (11,11)
        elif len(grid_images[label]) <= 400:
            grid_size = (20,20)
        else:
            grid_size = (30, 30)
        out = image_grid(grid_images[label], grid_size, size=(96, 96))
        cv2.imwrite(f"{output_folder_path}/grid_{label}.jpg", out)
    t3 = datetime.datetime.now()
    print(f"Number of people found: {num_people}")
    print(f"DBSCAN took {t1 - t0}")
    print(f"KMeans took {t2 - t1}")
    print(f"image_grid() and saving took ")
    return None


def main():
    t0 = datetime.datetime.now()
    with ProcessPoolExecutor(max_workers=procs) as executor:
        futures = []
        for filename in glob.glob(os.path.join(path, "*.jpg")):
            futures.append(executor.submit(read_and_find, filename))
    results = []
    for f in futures:
        result = f.result()
        if result:
            results.append(result)
    # now we have to process the faces...
    t1 = datetime.datetime.now()
    cluster(results)
    t2 = datetime.datetime.now()

    # for i, label in enumerate(labels):
    #     img = images[i]
    #     file_path = os.path.join(output_folder_path, f"{label}_face_{i}.jpg")
    #     cv2.imwrite(file_path, img)

    if DEBUG_TIMER:
        print(f"Time taken encoding  : {t1 - t0}")
        print(f"Tome taken clustering: {t2 - t1}")
        print(f"Total time           : {t2 - t0}")

if __name__ == "__main__":
    main()