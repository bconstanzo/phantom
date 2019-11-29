"""
Simple script to extract all the faces from an image and place them on an
output directory. Takes away complexity from tagger.py.
"""

import cv2
import datetime
import glob
import numpy as np
import os


from concurrent.futures import ProcessPoolExecutor
from phantom.faces import detect

procs = 3
IN_PATH = r"D:\Test\dlib_faces_5points\images"
#IN_PATH = r"C:\Test\infoconf2019\imagenes\fotos_full"
OUT_PATH = r"D:\Test\dlib_faces_5points\sub_faces"


def read_and_find(in_paths, out_path):
    """
    Reads an image from a path, and locates and encodes any face on it.

    :param path: path to an image
    :return: tuple of (cv2/np.ndarray, list of encodings, list of locations)
        or empty tuple if no face was found
    """
    for path in in_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        locations = detect(img)
        h, w = img.shape[:2]
        basepath, sep, name = path.rpartition("\\")
        name, dot, ext = name.rpartition(".")
        count = 0
        print(f"Found {len(locations)} faces in {path}...")
        for loc in locations:
            l, t, r, b = loc
            fw, fh = r - l, b - t
            dw, dh = int(fw * 0.5), int(fh * 0.5)
            l, t, r, b = l - dw, t - dh, r + dw, b + dh
            x0, y0 = max(0, l), max(0, t)
            x1, y1 = min(w, r), min(h, b)
            sub_img = img[y0:y1, x0:x1]
            cv2.imwrite(f"{out_path}/{name}_{count}_{(x0, y0, x1, y1)}.{ext}", sub_img)
    return None


def multiprocess_read_images(in_path, out_path):
    with ProcessPoolExecutor(max_workers=procs) as executor:
        futures = []
        all_files = glob.glob(os.path.join(in_path, "*.jpg"))
        for sub_files in ([all_files[i*10:i*10 + 10] for i in range((len(all_files) // 10) + 1) ]):
            futures.append(executor.submit(read_and_find, sub_files, out_path))
    results = []
    for f in futures:
        result = f.result()
        if result:
            results.append(result)
    return results

def main():
    t0 = datetime.datetime.now()
    multiprocess_read_images(IN_PATH, OUT_PATH)
    t1 = datetime.datetime.now()
    print("\n")
    print(f"Time: {t1 -t0}")

if __name__ == "__main__":
    main()
