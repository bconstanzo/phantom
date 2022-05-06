"""
Analyzes a directory of images, looking for faces. Each face found is encoded
into a 128-vector space. Then all the faces are grouped by similarity in this
space, using DBSCAN clustering.

---
Made by: bconstanzo
"""
# pylint: disable=E0001
# pylint: disable=E1101

import numpy as np
import cv2
import sys
import datetime
import dlib
import glob
import os


from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from phantom.faces import compare, detect, encode, Atlas, Face
from phantom.utils import image_grid
from pprint import pprint
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics


# Constants...
C_GRID_SIZE  = (96, 96)
C_LOAD_ATLAS = False
C_SAVE_ATLAS = True

# ...and variables
path  = r"D:\Storage-post-SSD\Wapp\WhatsApp Images"
path  = r"D:\Storage-post-SSD\gender\test\small"  # keeping the other path as a stress-test

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
    if C_LOAD_ATLAS:  # we don't have to do the encodings in this case
        return img, [], locations, path
    return img, encode(img, locations=locations), locations, path


def lerp_color(d):
    """
    Used to assign a color to the Silhouette scores for every sample when
    drawing the image grids.
    """
    p0 = (0, 0, 255)
    p1 = (0, 255, 0)
    p2 = (255, 0, 0)
    if d < 0:
        a = p0
        b = p1
        d = -d
    else:
        a = p2
        b = p1
    return (
        int(a[0] * d + b[0] * (1 - d)),
        int(a[1] * d + b[1] * (1 - d)),
        int(a[2] * d + b[2] * (1 - d)),
        )


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
    if C_SAVE_ATLAS:
        elements = []
        for idx in range(len(faces)): #  ugly, but we'll manage it for now
            e = faces[idx]
            try:
                i = cv2.resize(face_images[idx], C_GRID_SIZE)
            except cv2.error:
                print(f"cv2.error resizing for the atlas...")
                i = np.zeros((C_GRID_SIZE[0], C_GRID_SIZE[1], 3))
            o = paths[images_x_faces[idx]]
            elements.append((e, i, o))
        atlas = Atlas([Face(encoding=e, image=i, origin=o) 
                       for e, i, o in elements], "faces.atlas")
        atlas.save()
        
    # the idea is simple, we apply DBSCAN with a basic configuration and use
    # its result to apply k-means clustering:
    print(f"Number of faces detected: {len(faces)}")
    t0 = datetime.datetime.now()
    db = DBSCAN(eps=0.475, min_samples=2).fit(faces)
    t1 = datetime.datetime.now()
    # we can now approximate how many people are present...
    num_people = len(set(i for i in db.labels_ if i >= 0))
    # ...and use k-means to identify all the labels that DBSCAN couldn't
    #km = KMeans(init="k-means++", n_clusters=num_people, n_init=10).fit(faces)
    k_set = set()
    k_init = []
    for f, label in zip(faces, db.labels_):
        if label < 0:
            continue
        if label in k_set:
            continue
        k_init.append(f)
        k_set.add(label)
    km = KMeans(init=np.array(k_init), n_clusters=num_people, n_init=1).fit(faces)
    t2 = datetime.datetime.now()
    # now we group all the images for each cluster into a grid
    grid_images = defaultdict(list)
    grid_colors = defaultdict(list)
    grid_scores = defaultdict(list)
    count_outlier = 0
    s_scores = metrics.silhouette_samples([e.encoding for e in atlas.elements], db.labels_)
    for idx, (img, label, score) in enumerate(zip(face_images, db.labels_, s_scores)):
        if img is not None:
            #centroid = km.cluster_centers_[label]
            #distance = compare(centroid, faces[idx])
            distance = 0.5
            if distance < 0.9625:
                try:
                    grid_images[label].append(cv2.resize(img, C_GRID_SIZE))
                    grid_colors[label].append(lerp_color(score))
                    grid_scores[label].append(score)
                except cv2.error:
                    print(f"Raised -: {paths[images_x_faces[idx]]}")
                    pass
            else:
                print(f"Clustered face too far away from the centroid."
                      f"({label}_{count_outlier}, {distance})")
                try:
                    out = cv2.resize(img, C_GRID_SIZE)
                    cv2.imwrite(f"{output_folder_path}/outlier_grid_{label}_{count_outlier}.jpg", out)
                    count_outlier += 1
                except cv2.error:
                    pass

    labels_set = set(db.labels_)
    for label in labels_set:
        square_size = int(np.sqrt(len(grid_images[label]))) + 1
        grid_size = (square_size, square_size)
        out = image_grid(grid_images[label], grid_size, borders=True,
                         colors=grid_colors[label], size=C_GRID_SIZE)
        score = np.mean(grid_scores[label])
        height = out.shape[0]
        ypos = height - 32
        cv2.putText(out, f"{score:0.3f}", (12, ypos + 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"{score:0.3f}", (10, ypos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f"{output_folder_path}/grid_{label}.jpg", out)
    t3 = datetime.datetime.now()
    print(f"Number of people found: {num_people}")
    print(f"DBSCAN took {t1 - t0}")
    print(f"KMeans took {t2 - t1}")
    print(f"image_grid() and saving took {t3 - t2}")
    return None


def multiprocess_read_images():
    with ProcessPoolExecutor(max_workers=procs) as executor:
        futures = []
        for filename in glob.glob(os.path.join(path, "*.jpg")):
            futures.append(executor.submit(read_and_find, filename))
    results = []
    for f in futures:
        result = f.result()
        if result:
            results.append(result)
    return results


def main():
    t0 = datetime.datetime.now()
    results = multiprocess_read_images()
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