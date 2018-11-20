# pylint: disable=E1101
# pylint: disable=E0001

import cv2
import sys
import datetime
import dlib
import glob
import numpy as np
import os


from concurrent.futures import ProcessPoolExecutor
from phantom.faces import detect, encode
from pprint import pprint


path  = "C:/Users/Administrador/Desktop/Cosas/Database/peq"
procs = 2
DEBUG_TIMER = True
output_folder_path = "clusters"


def read_and_find(path):
    print(f"Reading {path}...")
    img = cv2.imread(path)
    if img is None:
        return []
    locations = detect(img)
    if not(locations):
        return []
    return encode(img, locations=locations)


def main():
    t0 = datetime.datetime.now()
    with ProcessPoolExecutor(max_workers=procs) as executor:
        futures = []
        for filename in glob.glob(os.path.join(path, "*.jpg")):
            futures.append(executor.submit(read_and_find, filename))
    results = []
    for f in futures:
        results.extend(f.result())
    # now we have to process the faces...
    tdelta = datetime.datetime.now() - t0

    d = []

   # for k, d in enumerate(dets):
   #     shape = sp(img, d)

    #    face_descriptor = facerec.compute_face_descriptor(img, shape)
    #    descriptors.append(face_descriptor)
    #  images.append((img, shape))

    labels = dlib.chinese_whispers_clustering(d, 0.5)
    num_classes = len(set(labels))
    print("Number of clusters: {}".format(num_classes))

    print("Saving faces in largest cluster to output folder...")

    for i, label in enumerate(labels):

     images = []      
     img, shape = images[i]
     images.append((img, shape)) 
     file_path = os.path.join(output_folder_path, f"{label}_face_{i}")
     dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)

    if DEBUG_TIMER:
        print(f"Time taken: {tdelta}")

if __name__ == "__main__":
    main()