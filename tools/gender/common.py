
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


def load_tagged(path):
    """
    Loads the tagged image data from a CSV file.

    :return: list of TaggedFace
    """
    tagged = []
    with open(path) as filehandle:
        _header = filehandle.readline()
        for line in filehandle:
            path, tag, age_tag = line.strip().rsplit(",", 2)
            tag = int(tag)
            age_tag = int(age_tag)
            img = cv2.imread(path)
            tagged.append(TaggedFace(tag, age_tag, path, img))
    return tagged


def save_tagged(tagged, path):
    """
    Saves the tagged image data to a CSV file.

    :param tagged: list of TaggedFace
    :param path: file path to save to
    """
    header = "path,tag,age_tag\n"
    with open(path, "w") as filehandle:
        filehandle.write(header)
        for t in tagged:
            filehandle.write(f"{t.path},{t.tag},{t.age_tag}\n")
    return None

