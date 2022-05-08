"""
Tool for quickly tagging face images to generate a training dataset for a
binary classifier (to detect gender).
"""
# pylint: disable=E1101
#   - because of pylint not recognizing dlib.svm_c_trainer_radial_basis

import numpy as np
import cv2
import dlib
import glob
import phantom
import pickle


from common import (
    TaggedFace, age_tags, tags_female, tags_male, load_tagged, save_tagged
)


from collections import defaultdict, namedtuple
from itertools import cycle
from phantom.faces import detect, encode, landmark
from phantom.utils import draw_faces
from PIL import Image, ImageDraw, ImageFont 


# Here be ~dragons~ the part where you can tweak configs
FLAG_TAG     = False
FLAG_SAVE    = False
FLAG_TRAIN   = True
FLAG_TEST    = False
PATH_TRAIN   = "d:/Test/dlib_faces_5points/sub_faces"
PATH_TAGFILE = "./tagged_faces.csv"
PATH_SVMFILE = "./model.pickle"
PATH_TEST    = ""  # point to a directory were you can easily check!
CONST_FONT   = ImageFont.truetype("consola.ttf", 16)

# and that's it for constants, now a few variables
tagged = []
color_cycle = cycle([
    (255,   0,   0),
    (  0, 255,   0),
    (  0,   0, 255),
    (255, 255,   0),
    (  0, 255, 255),
    (255,   0, 255),
])



def draw_text(canvas, text, pos, font, size, color, *, shadow=True):
    # img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # canvas = Image.fromarray(img_)
    color = color[::-1]
    draw = ImageDraw.Draw(canvas)
    if shadow:
        draw.text((pos[0] + 1, pos[1] + 1), text, (0, 0, 0), font=font)
    draw.text(pos, text, color, font=font)
    # ret = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    return canvas  # ret


def tag():
    """
    Show images from a path waiting for user input to tag them.

    We're keeping only the images were dlibs HOG face detector picks up only one
    face. Using dlibs 5 point face landmark dataset a balanced female/male ratio
    is found at about ~600 images. We also skip images were landmarking goes
    wrong, as they can negatively affect the result of facial encoding.

    :return: list of TaggedFace for each image
    """
    def age_table(age_counter, age, padding=2):
        lines = [["  "], ["m "], ["f "], ["  "], ["  "]]
        for idx, (low, high, name) in enumerate(age_tags[1:], start=1):
            lines[0].append(f"{low:{padding}}-{high:{padding}}")
            lines[1].append(f"{age_counter['m'][idx]:{padding * 2 + 1}}")
            lines[2].append(f"{age_counter['f'][idx]:{padding * 2 + 1}}")
            lines[3].append(f"{idx:^5}")
            lines[4].append(f"{'▲' if age == idx else ' ':^5}")
        lines[0].append("Total")
        lines[1].append(f"{sum(age_counter['m'][1:]): 5}")
        lines[2].append(f"{sum(age_counter['f'][1:]): 5}")
        lines[3].append(f"{sum(age_counter['m'][1:]) + sum(age_counter['f'][1:]): 5}")

        ret = ["|".join(l) if i < 3 else " ".join(l) for i, l in enumerate(lines)]
        return "\n".join(ret)

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
        while height > 768  or width > 768:  # hardcoded and hacky, but works
            face   = cv2.resize(face, (int(width/2), int(height/2)))
            noface = cv2.resize(noface, (int(width/2), int(height/2)))
            height, width = face.shape[:2]
        if height < 768 or width < 768:
            new_face = np.zeros((768, 1024, 3), dtype=np.uint8)
            new_noface = np.zeros((768, 1024, 3), dtype=np.uint8)
            new_face[384 - (height // 2): 384 - (height // 2) + height,
                     512 - (width // 2) : 512 - (width // 2)  + width] = face
            new_noface[384 - (height // 2): 384 - (height // 2) + height,
                       512 - (width // 2) : 512 - (width // 2)  + width] = noface
            face   = new_face
            noface = new_noface
        
        # a bit of an optimization to avoid multiple conversion between ndarray
        # and PIL.Image structures
        face   = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        noface = Image.fromarray(cv2.cvtColor(noface, cv2.COLOR_BGR2RGB))
        for y, line in enumerate(text.split("\n")):
            face   = draw_text(face,   line, (0, y * 20 + 20), CONST_FONT, 16, color)
            noface = draw_text(noface, line, (0, y * 20 + 20), CONST_FONT, 16, color)
        face   = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
        noface = cv2.cvtColor(np.array(noface), cv2.COLOR_RGB2BGR)
        return face, noface


    tagged = []
    age_counter = {
        "m": [ 0 for i in range(9) ],
        "f": [ 0 for i in range(9) ],
    }

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
        age_tag = 0
        text = (f"f: {count_f} "
                f"m: {count_m} "
                f"(total: {count_f + count_m})\n\n"
                +age_table(age_counter, age_tag))
        frame_face, frame_noface = redraw(img, faces, locations, color, text)
        cv2.imshow("Tagger", frame_face)
        key = chr(cv2.waitKey()).lower()
        while key not in "q mf":
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
            if key in "12345678":
                age_tag = int(key)
                text = (f"f: {count_f} "
                        f"m: {count_m} "
                        f"(total: {count_f + count_m})\n\n"
                        +age_table(age_counter, age_tag))
                frame_face, frame_noface = redraw(img, faces, locations, color, text)
                toggle_face = True
                cv2.imshow("Tagger", frame_face)
            key = chr(cv2.waitKey()).lower()
        if key == "q":
            break
        if key == " ":
            continue
        if key == "m":
            tag = tags_male
            count_m += 1
            age_counter["m"][age_tag] += 1
        if key == "f":
            tag = tags_female
            count_f += 1
            age_counter["f"][age_tag] += 1
        tagged.append(TaggedFace(tag, age_tag, filename, img))
    for t in tagged:
        print(t)
    cv2.destroyAllWindows()
    return tagged


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
    print(f"Using phantom {phantom.__version__}")
    if FLAG_TAG:
        print("Tagging...")
        tagged = tag()
        if FLAG_SAVE:
            save_tagged(tagged, PATH_TAGFILE)
    else:
        tagged = load_tagged(PATH_TAGFILE)
    if FLAG_TRAIN:
        train(tagged)


if __name__ == "__main__":
    main()
