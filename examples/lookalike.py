import argparse
import os.path

thresholds = {
    "ahash": 3,
    "dhash": 6,
    "phash": 12,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read the images from `needles` directory and then scan the "
                    "`haystack` directory looking for them based on visual "
                    "similarity."
    )
    parser.add_argument(
        "--hash",
        dest="hash",
        help="what hashing algorithm use for the images",
        choices=("ahash", "dhash", "phash"),
        default="dhash"
    )
    parser.add_argument(
        "needles",
        help="directory to read the needles from (the images you want to find)"
    )
    parser.add_argument(
        "haystack",
        help="directory where you want to find your needles (or similar)"
    )
    args = parser.parse_args()
    if not(os.path.isdir(args.needles)):
        print("Error: needles is not a directory!\n")
        parser.print_help()
        exit(-1)
    if not(os.path.isdir(args.haystack)):
        print("Error: haystack is not a directory!\n")
        parser.print_help()
        exit(-1)
    return args

def main():
    args = parse_args()
    # if we get here, it's all good
    import cv2
    import datetime
    import numpy as np
    import glob
    from collections import defaultdict
    from phantom.similarity import d_hash, p_hash, a_hash
    from pprint import pprint

    if args.hash == "ahash":
        hash_func = a_hash
        thresh = thresholds["ahash"]
    elif args.hash == "dhash":
        hash_func = d_hash
        thresh = thresholds["dhash"]
    else:
        hash_func = p_hash
        thresh = thresholds["phash"]
    needles = []
    print(f"Loading needles from {args.needles}...")
    print(f"    - Matching threshold set at {thresh}")
    for path in glob.glob(f"{args.needles}/*.jpg"):
        img = cv2.imread(path)
        if img is None:
            continue  # we couldn't read it
        value = hash_func(img)
        needles.append((value, path))
    print(f"Hash: {args.hash}")
    print(f"Scanning dir {args.haystack}...")
    counts = defaultdict(int)
    counts["matches"] = 0
    t0 = datetime.datetime.now()
    for path in glob.glob(f"{args.haystack}/*.jpg"):
        img = cv2.imread(path)
        if img is None:
            print(f"-- Warning: couldn't read {path}")
            continue
        counts["total_files"] += 1
        h = hash_func(img)
        for needle, needle_path in needles:
            value = np.sum(h ^ needle)
            if value <= thresh:
                counts["matches"] += 1
                counts[f"match-{needle_path}"] += 1
                print(f"Match: {path} matched {needle_path} ({value}/64)")
    t1 = datetime.datetime.now()
    print("-- Finished!\n")
    print(f"Time taken: {t1 - t0}")
    pprint(counts)
    

if __name__ == "__main__":
    main()