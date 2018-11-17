"""
Tool for quickly tagging face images to generate a training dataset for a
binary classifier (to detect gender).
"""

import cv2
import glob
import numpy as np


from collections import namedtuple
from itertools import cycle
from phantom.faces import detect, landmark
from phantom.utils import draw_faces


# Here be ~dragons~ the part were you can tweak configs
FLAG_TAG     = False
FLAG_SAVE    = False
FLAG_TRAIN   = True
FLAG_TEST    = True
PATH_TRAIN   = "D:/Storage-post-SSD/dlib_faces_5points/images"
PATH_TAGFILE = "./tagged_faces.csv"
PATH_SVMFILE = "./model.pickle"
PATH_TEST    = ""  # point to a directory were you can easily check!
CONST_FONT   = cv2.FONT_HERSHEY_SIMPLEX

# and that's it for constants, now a few variables
tags_female = +1
tags_male   = -1
tagged = []
color_cycle = cycle([
    (255,   0,   0),
    (  0, 255,   0),
    (  0,   0, 255),
    (255, 255,   0),
    (  0, 255, 255),
    (255,   0, 255),
])


# Classes/namedtuples
class TaggedFace:
    """
    Class to hold the results from the tagging.
    
    Formerly a named tuple, the __repr__ method was annoying on the console.

    :param tag: the gender tag assigned
    :param path: path to the image file
    :param img: cv2/numpy array with the loaded image
    :param face: list of facial landmarks in the image, result of
        `phantom.faces.detect()`
    """
    def __init__(self, tag, path, img, faces):
        self.tag = tag
        self.path = path
        self.img = img
        self.faces = faces
    
    def __repr__(self):
        return f"{self.__class__.__name__}(tag={self.tag}, path={self.path})"


def tag():
    """
    Show images from a path waiting for user input to tag them.

    We're keeping only the images were dlibs HOG face detector picks up only one
    face. Using dlibs 5 point face landmark dataset a balanced female/male ratio
    is found at about ~600 images. We also skip images were landmarking goes
    wrong, as they can affect the result of facial encoding.

    :return: list of TaggedFace for each image
    """
    def redraw(img, face, color, text):
        face = ((draw_faces(img, faces, color=color).astype(np.float32) * 0.5) + 
                (img * 0.5)).astype(np.uint8)
        noface = img.copy()
        height, width = img.shape[:2]
        if height > 960  or width > 1800:  # hardcoded and hacky, but works
            face   = cv2.resize(face, (int(width/2), int(height/2)))
            noface = cv2.resize(noface, (int(width/2), int(height/2)))
        
        for y, line in enumerate(text.split("\n")):
            cv2.putText(face,   line, (0, y * 20 + 20), CONST_FONT, 0.75, color, 2)
            cv2.putText(noface, line, (0, y * 20 + 20), CONST_FONT, 0.75, color, 2)
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
        tagged.append(TaggedFace(tag, filename, img, faces))
    for t in tagged:
        print(t)
    return tagged


def load_tagged(path):
    """
    Loads the tagged image data from a CSV file.

    :return: list of TaggedFace
    """
    tagged = []
    with open(path) as filehandle:
        _header = filehandle.readline()
        for line in filehandle:
            path, tag = line.strip().split(",")
            tag = int(tag)
            img = cv2.imread(path)
            faces = landmark(img)
            tagged.append(TaggedFace(tag, path, img, faces))
    return tagged


def save_tagged(tagged, path):
    """
    Saves the tagged image data to a CSV file.

    :param tagged: list of TaggedFace
    :param path: file path to save to
    """
    header = "path,tag\n"
    with open(path, "w") as filehandle:
        filehandle.write(header)
        for t in tagged:
            filehandle.write(f"{t.path},{t.tag}\n")
    return None


def main():
    if FLAG_TAG:
        tagged = tag()
        if FLAG_SAVE:
            save_tagged(tagged, PATH_TAGFILE)
    else:
        tagged = load_tagged(PATH_TAGFILE)


if __name__ == "__main__":
    main()
