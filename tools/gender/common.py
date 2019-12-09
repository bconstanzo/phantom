
# age tags
# we define 8 ranges (plus one empty at position 0) fr
age_tags = [
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

tags_female = +1
tags_male   = -1


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
    def __init__(self, tag, age_tag, path, img):
        self.tag = tag
        self.age_tag = age_tag
        self.path = path
        self.img = img
    
    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(tag={self.tag}, age_tag={self.age_tag} path={self.path})")
