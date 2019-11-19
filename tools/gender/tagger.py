"""
Tool for quickly tagging face images to generate a training dataset for a
binary classifier (to detect gender).
"""
# pylint: disable=E1101
#   - because of pylint not recognizing dlib.svm_c_trainer_radial_basis

import cv2
import dlib
import glob
import numpy as np
import pickle


from collections import namedtuple
from itertools import cycle
from phantom.faces import detect, encode, landmark
from phantom.utils import draw_faces


# Here be ~dragons~ the part were you can tweak configs
FLAG_TAG     = False
FLAG_SAVE    = True
FLAG_TRAIN   = True
FLAG_TEST    = False
PATH_TRAIN   = r"C:\Users\valen\Desktop\dataset_infoconf2019\fotos_full"
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
    def __init__(self, tag, path, img):
        self.tag = tag
        self.path = path
        self.img = img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(tag={self.tag}, path={self.path})"


def tag():
    """
    Show images from a path waiting for user input to tag them.

    We're keeping only the images were dlibs HOG face detector picks up only one
    face. Using dlibs 5 point face landmark dataset a balanced female/male ratio
    is found at about ~600 images. We also skip images were landmarking goes
    wrong, as they can negatively affect the result of facial encoding.

    :return: list of TaggedFace for each image
    """
    def redraw(img, face, locations, color, text):
        """
        Groups together all the frame drawing logic, since it was needed on
        many places.
        """
        point1 = (locations[0][0], locations[0][1])
        point2 = (locations[0][2], locations[0][3])
        img_ = img.copy()
        face_and_rect = cv2.rectangle(img_, point1, point2, color=color, thickness=2)
        face_and_rect = draw_faces(img_, faces, color=color)
        face = ((face_and_rect * 0.5) + 
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
        locations = detect(img)
        faces = landmark(img, locations=locations)
        if len(faces) != 1:
            continue
        toggle_face = True
        text = (f"total : {count_f + count_m}\n"
                f"female: {count_f}\n"
                f"male  : {count_m}")
        frame_face, frame_noface = redraw(img, faces, locations, color, text)
        cv2.imshow("Tagger", frame_face)
        key = chr(cv2.waitKey()).lower()
        while key not in "q mf":
            key = chr(cv2.waitKey()).lower()
            if key == "k":
                color = next(color_cycle)
                frame_face, frame_noface = redraw(img, faces, locations, color, text)
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
        tagged.append(TaggedFace(tag, filename, img))
    for t in tagged:
        print(t)
    cv2.destroyAllWindows()
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
            tagged.append(TaggedFace(tag, path, img))
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


def train(tagged):
    """
    Trains an SVM classifier based on the training data passed.

    Mostly based on http://dlib.net/svm_binary_classifier.py.html.

    :param tagged: list of TaggedFace to train on
    :return: dlib.svm
    """
    x = dlib.vectors()  # will carry the facial encodings
    y = dlib.array()    # will carry the gender label
    print("Preparing dataset...")
    total = len(tagged)
    for i, t in enumerate(tagged):
        print(f"\rEncoding {t.path} ({i + 1}/{total})...", end="")
        faces = encode(t.img)
        x.append(dlib.vector(faces[0]))
        y.append(t.tag)
        img = t.img
        for _ in range(5):
            faces = encode(img)
            if not faces:
                break
            x.append(dlib.vector(faces[0]))
            y.append(t.tag)
            img = cv2.resize(img,None,fx=0.7,fy=0.7)

    print("Training SVM...")
    trainer = dlib.svm_c_trainer_radial_basis()
    #trainer.be_verbose()
    trainer.set_c(10)
    model = trainer.train(x, y)
    with open(PATH_SVMFILE, "wb") as filehandle:
        pickle.dump(model, filehandle)
    return None


def test():
    """
    See:
    * https://talhassner.github.io/home/publication/2015_CVPR
    * https://talhassner.github.io/home/projects/Adience/Adience-data.html
    * https://talhassner.github.io/home/projects/frontalize/CVPR2015_frontalize.pdf
    """
    pass


def main():
    if FLAG_TAG:
        tagged = tag()
        if FLAG_SAVE:
            save_tagged(tagged, PATH_TAGFILE)
    else:
        tagged = load_tagged(PATH_TAGFILE)
    if FLAG_TRAIN:
        train(tagged)


if __name__ == "__main__":
    main()
