import cv2
import datetime
import numpy as np
import phantom
import pickle


from common import load_tagged, TaggedFace
from sklearn import svm, metrics


PATH_TAGGED = "tagged_faces.csv"


def normalize_gender(x):
    if x > 0.3:
        return 1
    if x < -0.3:
        return -1
    return 0


# instead of having a main() function we run the code as-is so that we can
# access the variables from running it with python -i
print(f"Using phantom {phantom.__version__}")
print("Loading tagged data...")
t0 = datetime.datetime.now()
tagged = load_tagged(PATH_TAGGED)
print("Preparing data...")
t1 = datetime.datetime.now()
total = 0
wrong_age = 0
wrong_gen = 0
for t in tagged:
    enc = phantom.faces.encode(t.img)[0]
    t.enc = enc
    tag = t.tag
    age_tag = t.age_tag
    gender = phantom.faces.estimate_gender(enc)
    age = phantom.faces.estimate_age(enc)[0]
    if normalize_gender(gender) != tag:
        wrong_gen += 1
    if age != age_tag:
        wrong_age += 1
    total += 1
print("Done testing!")
print(f"    Total:{total:18}")
print(f"    Acc. gender:{100*(total - wrong_gen)/total:12}")
print(f"    Acc. age:{100*(total - wrong_age)/total:15}")

print(f"Times:")
print(f"    Loading : {t1 - t0}")
