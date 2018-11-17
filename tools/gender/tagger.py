"""
Tool for quickly tagging face images to generate a training dataset for a
binary classifier (to detect gender).
"""

import cv2
import glob
import numpy as np


from itertools import cycle
from phantom.faces import detect, landmark
from phantom.utils import draw_faces


FLAG_TAG   = True
FLAG_SAVE  = False
FLAG_TRAIN = True
FLAG_TEST  = True
PATH_TRAIN = "D:/Storage-post-SSD/dlib_faces_5points/images"
PATH_SAVE  = "./model.pickle"
PATH_TEST  = ""
tags_female = +1
tags_male   = -1
tagged = []
color_cycle = cycle([
    (  0, 255,   0),
    (255,   0,   0),
    (  0,   0, 255),
    (255, 255,   0),
    (  0, 255, 255),
    (255,   0, 255),
])

def tag():
    """
    Show images from a path waiting for user input to tag them.

    :return: list of tuples (path, tag) for each image
    """
    def redraw(img, face, color, text):
        face = ((draw_faces(img, faces, color=color).astype(np.float32) * 0.5) + 
                (img * 0.5)).astype(np.uint8)
        noface = img.copy()
        height, width = img.shape[:2]
        if height > 960  or width > 1800:
            face   = cv2.resize(face, (int(width/2), int(height/2)))
            noface = cv2.resize(noface, (int(width/2), int(height/2)))
        
        for y, line in enumerate(text.split("\n")):
            cv2.putText(face, line, (0, y * 20 + 20), cv2.FONT_HERSHEY_IMPLEX, 0.75, color
            cv2.putText(noface, line, (0, y * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color)
        return face, noface


    tagged = []
    count_f = 0
    count_m = 0
    color = next(color_cycle)
    for filename in glob.glob(f"{PATH_TRAIN}/*.jpg"):
        img = cv2.imread(filename)
        faces = landmark(img)
        if len(faces) != 1:
            continue
        toggle_face = True
        text = (f"total : {count_f + count_m}\n"
                f"female: {count_f}\n"
                f"male  : {count_m}")
        frame_face, frame_noface = redraw(img, faces, color, text)
        cv2.imshow("Tagger", frame_face)
        key = chr(cv2.waitKey()).lower()
        while key not in "q mf":
            key = chr(cv2.waitKey()).lower()
            if key == "k":
                color = next(color_cycle)
                frame_face, frame_noface = redraw(img, faces, color, text)
                toggle_face = True
                cv2.imshow("Tagger", frame_face)
            if key == "l":
                if toggle_face:
                    cv2.imshow("Tagger", frame_noface)
                    toggle_face = False
                else:
                    cv2.imshow("Tagger", frame_face)
                    toggle_face = True
        if key == "q":
            break
        if key == " ":
            continue
        if key == "m":
            tag = tags_male
            count_m += 1
        if key == "f":
            tag = tags_female
            count_f += 1
        tagged.append((filename, tag))
    for t in tagged:
        print(t)
    return tagged


def load_tagged():
    """
    Loads the tagged image data from a file. WIP, for now it just returns
    an empty list.

    :return: empty list
    """
    return []


def main():
    if FLAG_TAG:
        tagged = tag()
    else:
        tagged = load_tagged()


if __name__ == "__main__":
    main()
