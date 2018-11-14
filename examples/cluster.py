import cv2
import datetime
import dlib
import glob
import numpy as np
import os


from concurrent.futures import ProcessPoolExecutor
from phantom.faces import detect, encode
from pprint import pprint


path  = "d:/storage-post-ssd/wapp/WhatsApp Images/"
procs = 3
DEBUG_TIMER = True


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
    pprint(results)
    if DEBUG_TIMER:
        print(f"Time taken: {tdelta}")

if __name__ == "__main__":
    main()