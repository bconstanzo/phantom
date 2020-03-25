from .faces import (
    # first, the classes
    Shape, Shape5p, Shape68p, Face, Atlas,
    # and now the functions
    detect, landmark, encode, compare, estimate_age, estimate_gender,
    normalize_landmark
    # and not much more for now
)

from . import align

age_tags = [
    # the elements of the tuples are:
    #    (lower age, higher age, description text)
    (-1,  -1, "none"),   # this one is a formality, it isn't used
    (0,    3, "baby"),
    (4,    9, "child"),
    (10,  13, "preteen"),
    (14,  17, "teen"),
    (18,  25, "young"),
    (26,  40, "young adult"),
    (41,  59, "adult"),
    (60,  99, "elder"),
]
